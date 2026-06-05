import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import random
import argparse

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class RadarMultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )

        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )

        self.branch7 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(96, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        x7 = self.branch7(x)

        x = torch.cat([x3, x5, x7], dim=1)
        x = self.fuse(x)

        return x
    
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
    
class RadarRADRAEEncoder(nn.Module):
    """
    Encode RAD and RAE tensors separately.
    RAD: [B, D, R, A] -> [B, C, R, A]
    RAE: [B, E, R, A] -> [B, C, R, A]
    """
    def __init__(self, d_in, e_in, feature_channels=64):
        super().__init__()

        self.rad_encoder = nn.Sequential(
            RadarMultiScaleBlock(
                in_channels=d_in,
                out_channels=128
            ),
            RadarMultiScaleBlock(
                in_channels=128,
                out_channels=feature_channels
            )
        )

        self.rae_encoder = nn.Sequential(
            RadarMultiScaleBlock(
                in_channels=e_in,
                out_channels=128
            ),
            RadarMultiScaleBlock(
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
        pooled_size=(8, 8)
    ):
        super().__init__()

        self.num_boxes = num_boxes
        self.box_dim = box_dim
        self.num_classes = num_classes
        self.total_classes = num_classes + 1  # add background

        self.decoder_backbone = nn.Sequential(
            RadarMultiScaleBlock(
                in_channels=input_channels,
                out_channels=hidden_channels
            ),
            RadarMultiScaleBlock(
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
    
class MVRSS3DModel2(nn.Module):
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
        pooled_size=(8, 8)
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

            RadarMultiScaleBlock(
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
