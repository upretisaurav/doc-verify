"""The fused image+text model, the image-only baselines, and the three task heads they share."""

import timm
import torch
from torch import nn

from doc_verify.data import KIND_CLASSES, TYPE_CLASSES
from doc_verify.windows import CLIP_TEXT_DIM, TEXT_WINDOWS

VIT_NAME = 'vit_base_patch16_224'


class Heads(nn.Module):
    """Document type (10 Tobacco classes), authenticity (1 logit) and inconsistency kind (5 classes)."""

    def __init__(self, dim):
        super().__init__()
        # not named `type`: nn.Module already has a .type() method, which would shadow the layer
        self.type_head = nn.Linear(dim, len(TYPE_CLASSES))
        self.auth_head = nn.Linear(dim, 1)
        self.kind_head = nn.Linear(dim, len(KIND_CLASSES))

    def forward(self, x):
        return {'type': self.type_head(x), 'auth': self.auth_head(x).squeeze(-1), 'kind': self.kind_head(x)}


class ViTEncoder(nn.Module):
    """ImageNet-pretrained ViT-B/16 returning its 197 per-patch token states (768-d), normalised internally."""

    def __init__(self, pretrained=True):
        super().__init__()
        self.vit = timm.create_model(VIT_NAME, pretrained=pretrained, num_classes=0)
        config = self.vit.pretrained_cfg
        self.register_buffer('mean', torch.tensor(config['mean']).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(config['std']).view(1, 3, 1, 1))
        self.dim = self.vit.num_features

    def forward(self, image):
        return self.vit.forward_features((image - self.mean) / self.std)


class FusionHead(nn.Module):
    """
    Cross-attention fusion. The query sequence is a learned [FUSE] token plus every text token from every window
    (projected, with a learned window-position embedding); it self-attends, so values in different windows can be
    compared, and cross-attends to the image patch tokens. The [FUSE] output feeds the heads.
    """

    def __init__(self, image_dim, text_dim=CLIP_TEXT_DIM, dim=512, layers=2, attention_heads=8,
                 n_windows=TEXT_WINDOWS, dropout=0.1):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, dim)
        self.text_proj = nn.Linear(text_dim, dim)
        self.window_embedding = nn.Embedding(n_windows, dim)
        self.fuse_token = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerDecoderLayer(dim, attention_heads, dim_feedforward=4 * dim, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, image_tokens, text, text_mask):
        batch, n_windows, tokens, _ = text.shape
        windows = torch.arange(n_windows, device=text.device)
        text = self.text_proj(text) + self.window_embedding(windows)[None, :, None, :]
        text = text.reshape(batch, n_windows * tokens, -1)
        valid = text_mask.reshape(batch, n_windows * tokens).bool()

        query = torch.cat([self.fuse_token.expand(batch, -1, -1), text], dim=1)
        padding = torch.cat([torch.zeros(batch, 1, dtype=torch.bool, device=text.device), ~valid], dim=1)
        fused = self.decoder(query, self.image_proj(image_tokens), tgt_key_padding_mask=padding)
        return self.norm(fused[:, 0])


class FusionModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.image_encoder = ViTEncoder(pretrained)
        self.fusion = FusionHead(self.image_encoder.dim)
        self.heads = Heads(self.fusion.dim)

    def backbone_parameters(self):
        return self.image_encoder.parameters()

    def forward(self, image, text, text_mask):
        return self.heads(self.fusion(self.image_encoder(image), text, text_mask))


class ImageOnlyViT(nn.Module):
    """Baseline: the same pretrained ViT-B/16, its [CLS] token straight into the same heads (no text)."""

    def __init__(self, pretrained=True):
        super().__init__()
        self.image_encoder = ViTEncoder(pretrained)
        self.heads = Heads(self.image_encoder.dim)

    def backbone_parameters(self):
        return self.image_encoder.parameters()

    def forward(self, image, text=None, text_mask=None):
        return self.heads(self.image_encoder(image)[:, 0])


class ResNet18Scratch(nn.Module):
    """Baseline: ResNet18 trained from random weights on document type only."""

    def __init__(self, pretrained=False):
        super().__init__()
        import torchvision

        self.resnet = torchvision.models.resnet18(weights=None)
        self.resnet.fc = nn.Identity()
        self.heads = Heads(512)

    def backbone_parameters(self):
        return self.resnet.parameters()

    def forward(self, image, text=None, text_mask=None):
        return self.heads(self.resnet(image))


MODELS = {'fused': FusionModel, 'image_vit': ImageOnlyViT, 'resnet18': ResNet18Scratch}
