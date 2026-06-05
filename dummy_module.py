import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import random
import argparse
from torchvision.ops import DeformConv2d

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
class RadarConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=5, padding=2, dilation=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=5, padding=2, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


class RadarDepthwiseSeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels),
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, padding=2, groups=128),
            nn.Conv2d(128, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


class RadarDeformConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2

        # offset
        #  kernel_size = 3, offset channels = 2 * 3 * 3 = 18
        # kernel_size = 5, offset channels = 2 * 5 * 5 = 50
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            padding=padding
        )

        #  deformable convolution
        self.deform_conv = DeformConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(inplace=True)

        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)

        x = self.deform_conv(x, offset)
        x = self.bn(x)
        x = self.act(x)

        return x


class RadarRADRAEDepthwiseSeparableEncoder(nn.Module):
    """
    Encode RAD and RAE tensors separately with depthwise separable conv blocks.
    RAD: [B, D, R, A] -> [B, C, R, A]
    RAE: [B, E, R, A] -> [B, C, R, A]
    """
    def __init__(self, d_in, e_in, feature_channels=64):
        super().__init__()

        self.rad_encoder = nn.Sequential(
            RadarDepthwiseSeparableConvBlock(
                in_channels=d_in,
                out_channels=128
            ),
            RadarDepthwiseSeparableConvBlock(
                in_channels=128,
                out_channels=feature_channels
            )
        )

        self.rae_encoder = nn.Sequential(
            RadarDepthwiseSeparableConvBlock(
                in_channels=e_in,
                out_channels=128
            ),
            RadarDepthwiseSeparableConvBlock(
                in_channels=128,
                out_channels=feature_channels
            )
        )

    def forward(self, rad: torch.Tensor, rae: torch.Tensor):
        rad_feat = self.rad_encoder(rad)
        rae_feat = self.rae_encoder(rae)

        return rad_feat, rae_feat
    
class RadarRADRAEEncoder(nn.Module):
    """
    Encode RAD and RAE tensors separately.
    RAD: [B, D, R, A] -> [B, C, R, A]
    RAE: [B, E, R, A] -> [B, C, R, A]
    """
    def __init__(self, d_in, e_in, feature_channels=64):
        super().__init__()

        self.rad_encoder = nn.Sequential(
            RadarConvBlock(
                in_channels=d_in,
                out_channels=128
            ),
            RadarConvBlock(
                in_channels=128,
                out_channels=feature_channels
            )
        )

        self.rae_encoder = nn.Sequential(
            RadarConvBlock(
                in_channels=e_in,
                out_channels=128
            ),
            RadarConvBlock(
                in_channels=128,
                out_channels=feature_channels
            )
        )
    
    def forward(self, rad: torch.Tensor, rae: torch.Tensor):

        """
        rad: [B, D, R, A]
        rae: [B, E, R, A]
        """

        rad_feat = self.rad_encoder(rad)
        rae_feat = self.rae_encoder(rae)

        return rad_feat, rae_feat


class Radar3DTensorDecoder(nn.Module):
    def __init__(
        self,
        input_channels,
        num_boxes,
        box_dim,
        num_classes,          # foreground classes only
        hidden_channels=128,
        pooled_size=(16, 16)
    ):
        super().__init__()

        self.num_boxes = num_boxes
        self.box_dim = box_dim
        self.num_classes = num_classes
        self.total_classes = num_classes + 1  # add background

        self.decoder_backbone = nn.Sequential(
            RadarConvBlock(
                in_channels=input_channels,
                out_channels=hidden_channels
            ),
            RadarConvBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels
            ),
        )

        self.pool = nn.AdaptiveAvgPool2d(pooled_size)

        pooled_h, pooled_w = pooled_size
        flatten_dim = hidden_channels * pooled_h * pooled_w

        self.box_head = nn.Linear(
            in_features=flatten_dim,
            out_features=num_boxes * box_dim
        )

        self.cls_head = nn.Linear(
            in_features=flatten_dim,
            out_features=num_boxes * self.total_classes
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, R, A]

        return:
            box_pred:   [B, num_boxes, box_dim]
            cls_pred: [B, num_boxes, num_classes + 1]
        """

        B = x.shape[0]

        feat = self.decoder_backbone(x)
        feat = self.pool(feat)
        feat = feat.flatten(start_dim=1)

        box_pred = self.box_head(feat)
        box_pred = box_pred.view(B, self.num_boxes, self.box_dim)

        cls_pred = self.cls_head(feat)
        cls_pred = cls_pred.view(B, self.num_boxes, self.total_classes)

        return {
            "box_pred": box_pred,
            "cls_pred": cls_pred,
        }


class Radar3DTensorDepthwiseSeparableDecoder(nn.Module):
    def __init__(
        self,
        input_channels,
        num_boxes,
        box_dim,
        num_classes,
        hidden_channels=128,
        pooled_size=(16, 16)
    ):
        super().__init__()

        self.num_boxes = num_boxes
        self.box_dim = box_dim
        self.num_classes = num_classes
        self.total_classes = num_classes + 1

        self.decoder_backbone = nn.Sequential(
            RadarDepthwiseSeparableConvBlock(
                in_channels=input_channels,
                out_channels=hidden_channels
            ),
            RadarDepthwiseSeparableConvBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels
            ),
        )

        self.pool = nn.AdaptiveAvgPool2d(pooled_size)

        pooled_h, pooled_w = pooled_size
        flatten_dim = hidden_channels * pooled_h * pooled_w

        self.box_head = nn.Linear(
            in_features=flatten_dim,
            out_features=num_boxes * box_dim
        )

        self.cls_head = nn.Linear(
            in_features=flatten_dim,
            out_features=num_boxes * self.total_classes
        )

    def forward(self, x: torch.Tensor):
        B = x.shape[0]

        feat = self.decoder_backbone(x)
        feat = self.pool(feat)
        feat = feat.flatten(start_dim=1)

        box_pred = self.box_head(feat)
        box_pred = box_pred.view(B, self.num_boxes, self.box_dim)

        cls_pred = self.cls_head(feat)
        cls_pred = cls_pred.view(B, self.num_boxes, self.total_classes)

        return {
            "box_pred": box_pred,
            "cls_pred": cls_pred,
        }
    
class MVRSS3DModel(nn.Module):
    """
    Complete RAD + RAE detection model.

    Input:
        rad: [B, D, R, A]
        rae: [B, E, R, A]

    Output:
        box_pred:   [B, num_boxes, box_dim]
        cls_pred: [B, num_boxes, num_classes + 1]

    Note:
        num_classes = foreground classes only
        background class is added inside decoder
    """

    def __init__(
        self,
        d_in,
        e_in,
        num_boxes,
        box_dim,
        num_classes,              # foreground classes only
        feature_channels=64,
        fusion_hidden_channels=64,
        decoder_hidden_channels=128,
        pooled_size=(16, 16)
    ):
        super().__init__()

        self.num_boxes = num_boxes
        self.box_dim = box_dim
        self.num_classes = num_classes
        self.total_classes = num_classes + 1

        # Encoder:
        # RAD: [B, D, R, A] -> [B, C, R, A]
        # RAE: [B, E, R, A] -> [B, C, R, A]
        self.encoder = RadarRADRAEEncoder(
            d_in=d_in,
            e_in=e_in,
            feature_channels=feature_channels
        )

        # Fusion:
        # concat: [B, C, R, A] + [B, C, R, A] -> [B, 2C, R, A]
        # fusion: [B, 2C, R, A] -> [B, C, R, A]
        self.fusion = nn.Sequential(
            nn.Conv2d(
                in_channels=feature_channels * 2,
                out_channels=fusion_hidden_channels,
                kernel_size=1
            ),
            nn.BatchNorm2d(fusion_hidden_channels),
            nn.LeakyReLU(inplace=True),

            RadarConvBlock(
                in_channels=fusion_hidden_channels,
                out_channels=feature_channels
            )
        )

        # Decoder:
        # [B, C, R, A] -> box_pred and cls_pred
        # cls_pred: [B, num_boxes, num_classes + 1]
        self.decoder = Radar3DTensorDecoder(
            input_channels=feature_channels,
            num_boxes=num_boxes,
            box_dim=box_dim,
            num_classes=num_classes,       # foreground only
            hidden_channels=decoder_hidden_channels,
            pooled_size=pooled_size
        )

    def forward(self, rad: torch.Tensor, rae: torch.Tensor):
        """
        rad: [B, D, R, A]
        rae: [B, E, R, A]
        """

        # Encode RAD and RAE separately.
        # rad_feat: [B, C, R, A]
        # rae_feat: [B, C, R, A]
        rad_feat, rae_feat = self.encoder(rad, rae)

        # Fuse features.
        # [B, C, R, A] + [B, C, R, A] -> [B, 2C, R, A]
        fused_feat = torch.cat([rad_feat, rae_feat], dim=1)

        # [B, 2C, R, A] -> [B, C, R, A]
        fused_feat = self.fusion(fused_feat)

        # Decode detection results.
        outputs = self.decoder(fused_feat)

        return outputs

class MVRSS3DModelDeform(MVRSS3DModel):
    def __init__(
        self,
        d_in,
        e_in,
        num_boxes,
        box_dim,
        num_classes,
        feature_channels=64,
        fusion_hidden_channels=64,
        decoder_hidden_channels=128,
        pooled_size=(8, 8)
    ):
        super().__init__(
            d_in=d_in,
            e_in=e_in,
            num_boxes=num_boxes,
            box_dim=box_dim,
            num_classes=num_classes,
            feature_channels=feature_channels,
            fusion_hidden_channels=fusion_hidden_channels,
            decoder_hidden_channels=decoder_hidden_channels,
            pooled_size=pooled_size
        )

        # Override fusion:
        # baseline 5x5 fusion block + one deformable conv refinement
        self.fusion = nn.Sequential(
            nn.Conv2d(
                in_channels=feature_channels * 2,
                out_channels=fusion_hidden_channels,
                kernel_size=1
            ),
            nn.BatchNorm2d(fusion_hidden_channels),
            nn.LeakyReLU(inplace=True),

            # Keep the original stable 5x5 radar convolution
            RadarConvBlock(
                in_channels=fusion_hidden_channels,
                out_channels=feature_channels
            ),

            # Add deformable convolution after normal convolution
            RadarDeformConvBlock(
                in_channels=feature_channels,
                out_channels=feature_channels,
                kernel_size=3
            )
        )


class MVRSS3DModelDeformDepthwiseSeparable(nn.Module):
    def __init__(
        self,
        d_in,
        e_in,
        num_boxes,
        box_dim,
        num_classes,
        feature_channels=64,
        fusion_hidden_channels=64,
        decoder_hidden_channels=128,
        pooled_size=(4, 4)
    ):
        super().__init__()

        self.num_boxes = num_boxes
        self.box_dim = box_dim
        self.num_classes = num_classes
        self.total_classes = num_classes + 1

        self.encoder = RadarRADRAEDepthwiseSeparableEncoder(
            d_in=d_in,
            e_in=e_in,
            feature_channels=feature_channels
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                in_channels=feature_channels * 2,
                out_channels=fusion_hidden_channels,
                kernel_size=1
            ),
            nn.BatchNorm2d(fusion_hidden_channels),
            nn.LeakyReLU(inplace=True),
            RadarDepthwiseSeparableConvBlock(
                in_channels=fusion_hidden_channels,
                out_channels=feature_channels
            ),
            RadarDeformConvBlock(
                in_channels=feature_channels,
                out_channels=feature_channels,
                kernel_size=3
            )
        )

        self.decoder = Radar3DTensorDepthwiseSeparableDecoder(
            input_channels=feature_channels,
            num_boxes=num_boxes,
            box_dim=box_dim,
            num_classes=num_classes,
            hidden_channels=decoder_hidden_channels,
            pooled_size=pooled_size
        )

    def forward(self, rad: torch.Tensor, rae: torch.Tensor):
        rad_feat, rae_feat = self.encoder(rad, rae)
        fused_feat = torch.cat([rad_feat, rae_feat], dim=1)
        fused_feat = self.fusion(fused_feat)
        outputs = self.decoder(fused_feat)

        return outputs
