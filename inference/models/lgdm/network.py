import torch
import torch.nn as nn
import torch.nn.functional as F

import clip
import ruamel.yaml as yaml

from inference.models.grasp_model import LanguageGraspModel

import inference.models.lgdm.albef.utils as utils
from inference.models.lgdm.albef.models.tokenization_bert import BertTokenizer
from inference.models.lgdm.albef.models.model_retrieval import ALBEF

filter_sizes = [32, 16, 8, 8, 16, 32]
kernel_sizes = [9, 5, 3, 3, 5, 9]
strides = [3, 2, 2, 2, 2, 3]


class LGDM(LanguageGraspModel):
    """
    GG-CNN
    Equivalient to the Keras Model used in the RSS Paper (https://arxiv.org/abs/1804.05172)
    """
    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0, clip_version='ViT-B/32'):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, filter_sizes[0], kernel_sizes[0], stride=strides[0], padding=3)
        self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_sizes[1], stride=strides[1], padding=2)
        self.conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_sizes[2], stride=strides[2], padding=1)
        self.convt1 = nn.ConvTranspose2d(filter_sizes[2], filter_sizes[3], kernel_sizes[3], stride=strides[3], padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(filter_sizes[3], filter_sizes[4], kernel_sizes[4], stride=strides[4], padding=2, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(filter_sizes[4], filter_sizes[5], kernel_sizes[5], stride=strides[5], padding=5, output_padding=1)

        self.y_flatten = nn.Sequential(
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.Linear(1024, 2888),
            nn.GELU(),
            # nn.Linear(2048, 2888),
            # nn.GELU(),
        )
        
        self.pos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.cos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.sin_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.width_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)

        # Setup language modality
        self.clip_version = clip_version
        self.lang_model = self._load_and_freeze_clip(self.clip_version)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)
        self._init_albef()

    def _init_albef(self):
        import argparse

        class WrapperArgument:
            def __init__(self):
                pass

            def add_attribute(self, name, value):
                setattr(self, name, value)

        args = WrapperArgument()
        args.config = 'inference/models/lgdm/albef/configs/Grounding.yaml'
        args.gradcam_mode = 'itm'
        args.block_num = 8
        args.text_encoder = 'bert-base-uncased'
        args.device = 'cuda'
        args.world_size = 1
        args.dist_url = 'env://'
        args.distributed = True

        utils.init_distributed_mode(args)

        config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
        self.tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

        self.albef = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=self.tokenizer)

    def forward(self, x, img, t, query, alpha, idx, prompt=None):
        # Encode text
        device = img.device
        text_input = self.tokenizer(query, padding='longest', max_length=30, return_tensors="pt").to(device)  
        image_atts, y = self.albef(img, text_input, alpha, idx)
        self.full_image_atts = self._process_attention_mask(image_atts=image_atts).to(device)
        r_channel, g_channel, b_channel = img[:, 0], img[:, 1], img[:, 2]
        r_channel, g_channel, b_channel = r_channel * self.full_image_atts, g_channel * self.full_image_atts, b_channel * self.full_image_atts
        r_channel, g_channel, b_channel = r_channel.unsqueeze(1), g_channel.unsqueeze(1), b_channel.unsqueeze(1)
        img = torch.cat([r_channel, g_channel, b_channel], dim=1)
        
        y = y[:, 0].unsqueeze(1)
        y = self.y_flatten(y)
        y = y.view(-1, 8, 19, 19)

        img = F.relu(self.conv1(img))
        img = F.relu(self.conv2(img))
        img = F.relu(self.conv3(img))

        # Combine textual features with the visual features
        # img = torch.clone(img).detach() + y

        img = F.relu(self.convt1(img))
        img = F.relu(self.convt2(img))
        img = F.relu(self.convt3(img))

        pos_output = self.pos_output(img)
        cos_output = self.cos_output(img)
        sin_output = self.sin_output(img)
        width_output = self.width_output(img)

        self.guiding_point = pos_output
        # Combine noise features from forward process to the guiding region
        pos_output = x + pos_output
        self.pos_output_str, self.cos_output_str, self.sin_output_str, self.width_output_str = pos_output, cos_output, sin_output, width_output
        
        return pos_output

    def _process_attention_mask(self, image_atts):
        bs, _ = image_atts.data.shape
        W, H, w, h = 224, 224, 14, 14
        ps = W // w
        image_atts = image_atts[:, 1:].view(-1, w, h).float()
        # image_atts.requires_grad = True

        # Initialize the larger tensor (224x224) with zeros
        full_image_atts = torch.zeros(bs, W, H)

        # Iterate through the smaller tensor and fill the corresponding 16x16 blocks
        for i in range(bs):  # Loop through each channel
            for j in range(w):  # Loop through each row in the smaller tensor
                for k in range(h):  # Loop through each column in the smaller tensor
                    # Fill the 16x16 block with the value from the smaller tensor
                    full_image_atts[i, j*16:(j+1)*16, k*16:(k+1)*16] = image_atts[i, j, k]
        
        return full_image_atts

    def compute_loss(self, yc, pos_pred, cos_pred, sin_pred, width_pred):
        y_pos, y_cos, y_sin, y_width = yc[0], yc[1], yc[2], yc[3]

        # if sample is None:
        #     pos_pred, cos_pred, sin_pred, width_pred = self.pos_output_str, self.cos_output_str, self.sin_output_str, self.width_output_str
        # else:
        #     pos_pred = sample
        #     cos_pred, sin_pred, width_pred = self.cos_output_str, self.sin_output_str, self.width_output_str

        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        # Get contrastive loss
        # contr_loss = self._get_contrastive_loss(self.full_image_atts.to(y_pos.device), y_pos)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss,
                # 'contr_loss': contr_loss,
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }
    
    def _get_contrastive_loss(self, x, y, temperature=1.0):
        # Normalize the vectors along the channel dimension
        x = F.normalize(x, dim=0, p=2)
        y = F.normalize(y, dim=0, p=2)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(x, y, dim=0) / temperature
        
        # Contrastive loss
        contrastive_loss = -torch.log(torch.exp(similarity) / torch.sum(torch.exp(similarity), dim=0))
        
        return torch.mean(contrastive_loss)

    def _load_and_freeze_clip(self, clip_version, device=None):
        clip_model, clip_preprocess = clip.load(clip_version, device=device,
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def _encode_text(self, raw_text, device=None):
        # raw_text - list (batch_size length) of strings with input text prompts
        max_text_len = 20 # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.lang_model.encode_text(texts).float()
