import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from inference.models.grasp_model import LanguageGraspModel, GraspModel
from .mobile_vit import get_model


class RAGT(GraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=18, dropout=False, prob=0.0):
        super(RAGT, self).__init__()
        self.mobile_vit = get_model()

        # Upsampling layers to increase spatial dimensions
        self.upsample_layers = nn.Sequential(
            nn.Upsample(scale_factor=33, mode='bilinear', align_corners=False),
            nn.ReLU()
        )

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)
    
    def get_predicted_grasp(self, x_in):
        x = self.mobile_vit(x_in)
        x = self.upsample_layers(x)
        
        return x

    def forward(self, x_in):
        x = self.mobile_vit(x_in)
        x = self.upsample_layers(x)
        x = x[:,:,:225, :225]

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output


class LRAGT(LanguageGraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=18, dropout=False, prob=0.0, clip_version='ViT-B/32'):
        super(LRAGT, self).__init__()
        self.mobile_vit = get_model()

        # Upsampling layers to increase spatial dimensions
        self.upsample_layers = nn.Sequential(
            nn.Upsample(scale_factor=33, mode='bilinear', align_corners=False),
            nn.ReLU()
        )

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.y_flatten = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 225),
            nn.GELU(),
        )

        # Setup language modality
        self.clip_version = clip_version
        self.lang_model = self._load_and_freeze_clip(self.clip_version)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in, prompt, query):
        x = self.mobile_vit(x_in)
        x = self.upsample_layers(x)
        x = x[:,:,:225, :225]

        # Encode text
        device = x.device
        y_feats = self._encode_text(query, device=device)
        y_feats = self.y_flatten(y_feats)
        y_feats = y_feats.unsqueeze(2).expand(-1, -1, 225).unsqueeze(1).expand(-1, 18, -1, -1)

        x = torch.clone(x).detach() + y_feats

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

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