"""
original code from apple:
https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit.py
"""

from typing import Optional, Tuple, Union, Dict
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from inference.models.ragt.transformer import TransformerEncoder
from inference.models.ragt.model_config import get_config

from inference.models.ragt.Anchor import num_anchors


def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvLayer(nn.Module):
    """
    Applies a 2D convolution over an input

    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
        kernel_size (Union[int, Tuple[int, int]]): Kernel size for convolution.
        stride (Union[int, Tuple[int, int]]): Stride for convolution. Default: 1
        groups (Optional[int]): Number of groups in convolution. Default: 1
        bias (Optional[bool]): Use bias. Default: ``False``
        use_norm (Optional[bool]): Use normalization layer after convolution. Default: ``True``
        use_act (Optional[bool]): Use activation layer after convolution (or convolution and normalization).
                                Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        For depth-wise convolution, `groups=C_{in}=C_{out}`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        use_norm: Optional[bool] = True,
        use_act: Optional[bool] = True,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)

        padding = (
            int((kernel_size[0] - 1) / 2),
            int((kernel_size[1] - 1) / 2),
        )

        block = nn.Sequential()

        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=bias
        )

        block.add_module(name="conv", module=conv_layer)

        if use_norm:
            norm_layer = nn.BatchNorm2d(num_features=out_channels, momentum=0.1)
            block.add_module(name="norm", module=norm_layer)

        if use_act:
            act_layer = nn.SiLU()
            block.add_module(name="act", module=act_layer)

        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class InvertedResidual(nn.Module):
    """
    This class implements the inverted residual block, as described in `MobileNetv2 <https://arxiv.org/abs/1801.04381>`_ paper

    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        stride (int): Use convolutions with a stride. Default: 1
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        skip_connection (Optional[bool]): Use skip-connection. Default: True

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        If `in_channels =! out_channels` and `stride > 1`, we set `skip_connection=False`

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Union[int, float],
        skip_connection: Optional[bool] = True,
    ) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super().__init__()

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1
                ),
            )

        block.add_module(
            name="conv_3x3",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim
            ),
        )

        block.add_module(
            name="red_1x1",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                use_act=False,
                use_norm=True,
            ),
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.stride = stride
        self.use_res_connect = (
            self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class MobileViTBlock(nn.Module):
    """
    This class defines the `MobileViT block <https://arxiv.org/abs/2110.02178?context=cs.LG>`_

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        transformer_dim (int): Input dimension to the transformer unit
        ffn_dim (int): Dimension of the FFN block
        n_transformer_blocks (int): Number of transformer blocks. Default: 2
        head_dim (int): Head dimension in the multi-head attention. Default: 32
        attn_dropout (float): Dropout in multi-head attention. Default: 0.0
        dropout (float): Dropout rate. Default: 0.0
        ffn_dropout (float): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (int): Patch height for unfolding operation. Default: 8
        patch_w (int): Patch width for unfolding operation. Default: 8
        transformer_norm_layer (Optional[str]): Normalization layer in the transformer block. Default: layer_norm
        conv_ksize (int): Kernel size to learn local representations in MobileViT block. Default: 3
        no_fusion (Optional[bool]): Do not combine the input and output feature maps. Default: False
    """

    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        ffn_dim: int,
        n_transformer_blocks: int = 2,
        head_dim: int = 32,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        patch_h: int = 8,
        patch_w: int = 8,
        conv_ksize: Optional[int] = 3,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        conv_3x3_in = ConvLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1
        )
        conv_1x1_in = ConvLayer(
            in_channels=in_channels,
            out_channels=transformer_dim,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False
        )

        conv_1x1_out = ConvLayer(
            in_channels=transformer_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1
        )
        conv_3x3_out = ConvLayer(
            in_channels=2 * in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1
        )

        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        global_rep = [
            TransformerEncoder(
                embed_dim=transformer_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout
            )
            for _ in range(n_transformer_blocks)
        ]
        global_rep.append(nn.LayerNorm(transformer_dim))
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def unfolding(self, x: Tensor) -> Tuple[Tensor, Dict]:
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = patch_w * patch_h
        batch_size, in_channels, orig_h, orig_w = x.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] -> [B * C * n_h, p_h, n_w, p_w]
        x = x.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
        x = x.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        x = x.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] -> [B, P, N, C]
        x = x.transpose(1, 3)
        # [B, P, N, C] -> [BP, N, C]
        x = x.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return x, info_dict

    def folding(self, x: Tensor, info_dict: Dict) -> Tensor:
        n_dim = x.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
            x.shape
        )
        # [BP, N, C] --> [B, P, N, C]
        x = x.contiguous().view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        )

        batch_size, pixels, num_patches, channels = x.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] -> [B, C, N, P]
        x = x.transpose(1, 3)
        # [B, C, N, P] -> [B*C*n_h, n_w, p_h, p_w]
        x = x.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B*C*n_h, n_w, p_h, p_w] -> [B*C*n_h, p_h, n_w, p_w]
        x = x.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] -> [B, C, H, W]
        x = x.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            x = F.interpolate(
                x,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False,
            )
        return x

    def forward(self, x: Tensor) -> Tensor:
        res = x

        fm = self.local_rep(x)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)

        # learn global representations
        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)

        # [B x Patch x Patches x C] -> [B x C x Patches x Patch]
        fm = self.folding(x=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)

        fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm


# 将四倍下采样特征与五倍下采样特征融合
# 将模型改为单分支输出预测
# S版本，4倍下采样输出128通道，5倍下采样输出160通道
class ResidualConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 middle_channels: int,
                 ) -> None:
        super().__init__()
        self.res_conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=middle_channels,
                      kernel_size=(3, 3), stride=(1, 1), padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=middle_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=middle_channels, out_channels=in_channels,
                      kernel_size=(3, 3), stride=(1, 1), padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=in_channels),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        x = F.silu(self.res_conv_block(x) + x)
        return x


class ConvSet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 middle_channels: int,
                 out_channels: int,
                 ) -> None:
        super(ConvSet, self).__init__()
        self.convset = nn.Sequential()
        self.convset.add_module(name='res_conv_block1',
                                module=ResidualConvBlock(in_channels=in_channels, middle_channels=middle_channels))
        self.convset.add_module(name='res_conv_block2',
                                module=ResidualConvBlock(in_channels=in_channels, middle_channels=middle_channels))
        self.convset.add_module(name='res_conv_block3',
                                module=ResidualConvBlock(in_channels=in_channels, middle_channels=middle_channels))
        self.convset.add_module(name='res_conv_block4',
                                module=ResidualConvBlock(in_channels=in_channels, middle_channels=middle_channels))
        self.convset.add_module(name='conv_3x3',
                                module=nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                 kernel_size=(3, 3), stride=(1, 1), padding=1,
                                                 bias=False))
        self.convset.add_module(name='bn', module=nn.BatchNorm2d(num_features=out_channels))
        self.convset.add_module(name='SiLU', module=nn.SiLU())

    def forward(self, x: Tensor) -> Tensor:
        x = self.convset(x)
        return x


class ConvDownSampling(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ) -> None:
        super().__init__()
        self.convdownsampling = nn.Sequential()
        # 3x3卷积
        self.convdownsampling.add_module(name='conv1_3x3', module=nn.Conv2d(in_channels=in_channels,
                                                                            out_channels=2*in_channels,
                                                                            kernel_size=(3, 3),
                                                                            stride=(2, 2),
                                                                            padding=1,
                                                                            bias=False))
        self.convdownsampling.add_module(name='bn1', module=nn.BatchNorm2d(num_features=2*in_channels))
        self.convdownsampling.add_module(name='SiLU1', module=nn.SiLU())
        # 3x3卷积
        self.convdownsampling.add_module(name='conv2_3x3', module=nn.Conv2d(in_channels=2*in_channels,
                                                                            out_channels=out_channels,
                                                                            kernel_size=(3, 3),
                                                                            stride=(1, 1),
                                                                            padding=1,
                                                                            bias=False))
        self.convdownsampling.add_module(name='bn2', module=nn.BatchNorm2d(num_features=out_channels))
        self.convdownsampling.add_module(name='SiLU2', module=nn.SiLU())

    def forward(self, x: Tensor) -> Tensor:
        x = self.convdownsampling(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 max_len: int,
                 ) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # # pe: [max_len, d_model] -> [1, max_len, d_model] -> [max_len, 1, d_model]
        # pe = pe.unsqueeze(0).transpose(0, 1)
        # pe: [max_len, d_model] -> [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        # # x: [max_len, N, d_model]
        # return x + self.pe[:x.size(0), :]
        # x: [N, max_len, d_model]
        # broadcast: pe: [1, max_len, d_model] -> [N, max_len, d_model]
        return x + self.pe[:, :x.size(1), :]


class GlobalTransformerEncoder(nn.Module):
    def __init__(self,
                 # in_channels: int,
                 d_model: int,
                 n_head: int,
                 ffn_dim: int,
                 encoder_layers: int,
                 ) -> None:
        super().__init__()

        # if in_channels != d_model:
        #     # 调整维度到d_model
        #     self.conv1_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=d_model,
        #                             kernel_size=(1, 1), stride=(1, 1), padding=0)
        #     # 调整维度到in_channels
        #     self.conv2_1x1 = nn.Conv2d(in_channels=d_model, out_channels=in_channels,
        #                             kernel_size=(1, 1), stride=(1, 1), padding=0)
            
        # 位置编码
        self.position_encoding = PositionalEncoding(d_model=d_model, max_len=169)
        # Transformer Encoder全局表征
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head,
                                                                    dim_feedforward=ffn_dim,
                                                                    batch_first=True,
                                                                    norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer,
                                                         num_layers=encoder_layers, norm=nn.LayerNorm(d_model))
        
    def sequentialize(self, x: Tensor) -> Tuple[Tensor, dict]:
        N, C, H, W = x.shape
        # [N C H W] -> [N H W C]
        x = x.permute(0, 2, 3, 1)
        # [N H W C] -> [N H*W C]
        x = x.reshape(N, H*W, C)
        shape_dict = {
            'origin_N': N,
            'origin_C': C,
            'origin_H': H,
            'origin_W': W,
        }
        return x, shape_dict

    def unsequentialize(self, x: Tensor, dim_dict: dict) -> Tensor:
        # [N H*W C] -> [N H W C]
        x = x.contiguous().view(dim_dict['origin_N'], dim_dict['origin_H'], dim_dict['origin_W'], dim_dict['origin_C'])
        # [N H W C] -> [N C H W]
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, x: Tensor) -> Tensor:
        # x = self.conv1_1x1(x)
        x, shape_dict = self.sequentialize(x)
        x = self.position_encoding(x)
        x = self.transformer_encoder(x)
        x = self.unsequentialize(x, dim_dict=shape_dict)
        # x = self.conv2_1x1(x)
        # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x


# class TransposeConv(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels
#                  ):
#         super(TransposeConv, self).__init__()
#         self.transpose_conv = nn.Sequential()
#         self.transpose_conv.add_module(name='conv_3x3', module=nn.Conv2d(in_channels=in_channels,
#                                                                          out_channels=in_channels,
#                                                                          kernel_size=(3, 3), stride=(1, 1), padding=1,
#                                                                          bias=False))
#         self.transpose_conv.add_module(name='bn1', module=nn.BatchNorm2d(in_channels))
#         self.transpose_conv.add_module(name='SiLU1', module=nn.SiLU())
#         self.transpose_conv.add_module(name='conv_1x1', module=nn.Conv2d(in_channels=in_channels,
#                                                                          out_channels=out_channels,
#                                                                          kernel_size=(1, 1), stride=(1, 1), padding=0,
#                                                                          bias=False))
#         self.transpose_conv.add_module(name='bn2', module=nn.BatchNorm2d(out_channels))
#         self.transpose_conv.add_module(name='SiLU2', module=nn.SiLU())
#         self.transpose_conv.add_module(name='transpose_conv', module=nn.ConvTranspose2d(in_channels=out_channels,
#                                                                                         out_channels=out_channels,
#                                                                                         kernel_size=(2, 2),
#                                                                                         stride=(2, 2), padding=(0, 0),
#                                                                                         output_padding=(0, 0)))
#         self.transpose_conv.add_module(name='bn3', module=nn.BatchNorm2d(out_channels))
#         self.transpose_conv.add_module(name='SiLU3', module=nn.SiLU())

#     def forward(self, x):
#         x = self.transpose_conv(x)
#         return x


# class SpatialPyramidPoolingFast(nn.Module):
#     def __init__(
#             self,
#             in_channels
#     ):
#         super(SpatialPyramidPoolingFast, self).__init__()
#         # 降维1*1卷积
#         self.conv1_1_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//4,
#                                    kernel_size=(1, 1), stride=(1, 1),
#                                    bias=False)
#         self.bn1 = nn.BatchNorm2d(num_features=in_channels//4)
#         self.SiLU1 = nn.SiLU()
#         # 最大池化
#         self.MaxPool_5_5 = nn.MaxPool2d(kernel_size=(5, 5), stride=(1, 1), padding=2)
#         self.MaxPool1_3_3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1)
#         self.MaxPool2_3_3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1)
#         # 特征融合
#         self.conv2_1_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
#                                    kernel_size=(1, 1), stride=(1, 1),
#                                    bias=False)
#         self.bn2 = nn.BatchNorm2d(num_features=in_channels)
#         self.SiLU2 = nn.SiLU()

#     def forward(self, x):
#         x1 = self.conv1_1_1(x)
#         x1 = self.bn1(x1)
#         x1 = self.SiLU1(x1)
#         x2 = self.MaxPool_5_5(x1)
#         x3 = self.MaxPool1_3_3(x2)
#         x4 = self.MaxPool2_3_3(x3)
#         x = torch.concat([x1, x2, x3, x4], dim=1)
#         x = self.conv2_1_1(x)
#         x = self.bn2(x)
#         x = self.SiLU2(x)
#         return x


class Detector(nn.Module):
    def __init__(self,
                 in_channels: int,
                 middle_channels: int,
                 out_channels: int,
                 ) -> None:
        super(Detector, self).__init__()
        self.detector = nn.Sequential()
        self.detector.add_module(name='res_conv_block1',
                                 module=ResidualConvBlock(in_channels=in_channels, middle_channels=middle_channels))
        self.detector.add_module(name='res_conv_block2',
                                 module=ResidualConvBlock(in_channels=in_channels, middle_channels=middle_channels))
        self.detector.add_module(name='res_conv_block3',
                                 module=ResidualConvBlock(in_channels=in_channels, middle_channels=middle_channels))
        self.detector.add_module(name='res_conv_block4',
                                 module=ResidualConvBlock(in_channels=in_channels, middle_channels=middle_channels))
        self.detector.add_module(name='res_conv_block5',
                                 module=ResidualConvBlock(in_channels=in_channels, middle_channels=middle_channels))
        # 输出
        self.detector.add_module(name='conv2_1x1', module=nn.Conv2d(in_channels=in_channels,
                                                                    out_channels=out_channels,
                                                                    kernel_size=(1, 1), stride=(1, 1), padding=0,
                                                                    bias=True))

    def forward(self, x: Tensor) -> Tensor:
        x = self.detector(x)
        return x


class MobileViT(nn.Module):
    """
    This class implements the `MobileViT architecture <https://arxiv.org/abs/2110.02178?context=cs.LG>`_
    """
    def __init__(self, model_cfg: Dict, num_classes: int = 1000):
        super().__init__()

        image_channels = 3
        out_channels = 16

        self.conv_1 = ConvLayer(
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2
        )

        self.layer_1, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer1"])
        self.layer_2, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer2"])
        self.layer_3, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer3"])
        self.layer_4, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer4"])
        self.layer_5, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer5"])

        # exp_channels = min(model_cfg["last_layer_exp_factor"] * out_channels, 960)
        # self.conv_1x1_exp = ConvLayer(
        #     in_channels=out_channels,
        #     out_channels=exp_channels,
        #     kernel_size=1
        # )
        #
        # self.classifier = nn.Sequential()
        # self.classifier.add_module(name="global_pool", module=nn.AdaptiveAvgPool2d(1))
        # self.classifier.add_module(name="flatten", module=nn.Flatten())
        # if 0.0 < model_cfg["cls_dropout"] < 1.0:
        #     self.classifier.add_module(name="dropout", module=nn.Dropout(p=model_cfg["cls_dropout"]))
        # self.classifier.add_module(name="fc", module=nn.Linear(in_features=exp_channels, out_features=num_classes))

        # 信息融合、检测模块
        self.conv_downsampling1 = ConvDownSampling(in_channels=96, out_channels=128)
        self.convset1 = ConvSet(in_channels=256, middle_channels=128, out_channels=120)
        self.conv_downsampling2 = ConvDownSampling(in_channels=120, out_channels=160)
        self.convset2 = ConvSet(in_channels=320, middle_channels=160, out_channels=240)
        self.transformer_encoder = GlobalTransformerEncoder(d_model=240, n_head=4, ffn_dim=480, encoder_layers=4)
        self.detector = Detector(in_channels=240, middle_channels=120, out_channels=num_anchors*6)

        # weight init
        self.apply(self.init_parameters)

    def _make_layer(self, input_channel, cfg: Dict) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(input_channel=input_channel, cfg=cfg)
        else:
            return self._make_mobilenet_layer(input_channel=input_channel, cfg=cfg)

    @staticmethod
    def _make_mobilenet_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channels

        return nn.Sequential(*block), input_channel

    @staticmethod
    def _make_mit_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        stride = cfg.get("stride", 1)
        block = []

        if stride == 2:
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4)
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        num_heads = cfg.get("num_heads", 4)
        head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            raise ValueError("Transformer input dimension should be divisible by head dimension. "
                             "Got {} and {}.".format(transformer_dim, head_dim))

        block.append(MobileViTBlock(
            in_channels=input_channel,
            transformer_dim=transformer_dim,
            ffn_dim=ffn_dim,
            n_transformer_blocks=cfg.get("transformer_blocks", 1),
            patch_h=cfg.get("patch_h", 2),
            patch_w=cfg.get("patch_w", 2),
            dropout=cfg.get("dropout", 0.1),
            ffn_dropout=cfg.get("ffn_dropout", 0.0),
            attn_dropout=cfg.get("attn_dropout", 0.1),
            head_dim=head_dim,
            conv_ksize=3
        ))

        return nn.Sequential(*block), input_channel

    @staticmethod
    def init_parameters(m):
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)

        x = self.layer_3(x)  # 下采样率8，输出维度：52x52x96
        x1 = self.layer_4(x)  # 下采样率16，输出维度：26x26x128
        x2 = self.layer_5(x1)  # 下采样率32，输出维度：13x13x160

        x = self.conv_downsampling1(x)
        x = torch.concat([x, x1], dim=1)
        x = self.convset1(x)
        x = self.conv_downsampling2(x)
        x = torch.concat([x, x2], dim=1)
        x = self.convset2(x)
        x = self.transformer_encoder(x)
        x = self.detector(x)
        return x


def mobile_vit_xx_small(num_classes: int = 1000):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xxs.pt
    config = get_config("xx_small")
    m = MobileViT(config, num_classes=num_classes)
    return m


def mobile_vit_x_small(num_classes: int = 1000):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xs.pt
    config = get_config("x_small")
    m = MobileViT(config, num_classes=num_classes)
    return m


def mobile_vit_small(num_classes: int = 1000):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt
    config = get_config("small")
    m = MobileViT(config, num_classes=num_classes)
    return m


def get_model():
    model = mobile_vit_small()
    return model
