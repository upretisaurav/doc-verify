"""Training with the masked multi-task loss, early stopping on validation, and validation predictions."""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from doc_verify.data import MISSING, PageDataset, load_pages
from doc_verify.evaluate import head_metrics, selection_score
from doc_verify.models import MODELS


@dataclass
class TrainConfig:
    model: str = 'fused'
    lambda_auth: float = 1.0
    lambda_kind: float = 1.0
    use_type: bool = True
    epochs: int = 15
    patience: int = 3
    batch_size: int = 32
    lr_backbone: float = 2e-5
    lr_head: float = 1e-4
    weight_decay: float = 0.05
    seed: int = 42
    num_workers: int = 2
    pretrained: bool = True


def pick_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def masked_losses(outputs, batch, config):
    """
    L = L_type + lambda_auth * L_auth + lambda_kind * L_kind, each term averaged only over the pages in the batch
    that carry that label (Tobacco pages have no authenticity/kind label, SAD-Syn pages no type label).
    A term with no labelled pages in the batch contributes 0.
    """
    zero = outputs['type'].sum() * 0.0
    losses = {}

    mask = batch['type_label'] != MISSING
    losses['type'] = F.cross_entropy(outputs['type'][mask], batch['type_label'][mask]) if mask.any() else zero

    mask = batch['auth_label'] != MISSING
    losses['auth'] = (F.binary_cross_entropy_with_logits(outputs['auth'][mask], batch['auth_label'][mask].float())
                      if mask.any() else zero)

    mask = batch['kind_label'] != MISSING
    losses['kind'] = F.cross_entropy(outputs['kind'][mask], batch['kind_label'][mask]) if mask.any() else zero

    losses['total'] = (config.use_type * losses['type'] + config.lambda_auth * losses['auth']
                       + config.lambda_kind * losses['kind'])
    return losses


def forward(model, batch, device):
    text = batch['text'].to(device) if 'text' in batch else None
    text_mask = batch['text_mask'].to(device) if 'text_mask' in batch else None
    return model(batch['image'].to(device), text, text_mask)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device == 'cuda'):
            outputs = forward(model, batch, device)
        auth_score = torch.sigmoid(outputs['auth'].float()).cpu().numpy()
        rows.append(pd.DataFrame({
            'page_index': batch['page_index'].numpy(),
            'type_label': batch['type_label'].numpy(),
            'auth_label': batch['auth_label'].numpy(),
            'kind_label': batch['kind_label'].numpy(),
            'type_pred': outputs['type'].argmax(-1).cpu().numpy(),
            'auth_score': auth_score,
            'auth_pred': (auth_score >= 0.5).astype(int),
            'kind_pred': outputs['kind'].argmax(-1).cpu().numpy(),
        }))
    return pd.concat(rows, ignore_index=True)


def train(config, out_dir, data_root=None, text_cache=None, pages=None, device=None):
    """
    Train one model and keep the epoch with the best validation selection score.
    Writes config.json, history.csv, best.pt and val_predictions.csv to out_dir. Returns the best metrics.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or pick_device()

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    pages = load_pages(('train', 'val'), data_root) if pages is None else pages
    if not config.lambda_auth and not config.lambda_kind:
        pages = pages[pages.type_label != MISSING]
    if not config.use_type:
        pages = pages[pages.type_label == MISSING]

    uses_text = config.model == 'fused'
    datasets = {split: PageDataset(pages[pages.split == split], data_root, text_cache if uses_text else None)
                for split in ('train', 'val')}
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {
        'train': DataLoader(datasets['train'], batch_size=config.batch_size, shuffle=True, generator=generator,
                            num_workers=config.num_workers, persistent_workers=config.num_workers > 0),
        'val': DataLoader(datasets['val'], batch_size=config.batch_size, num_workers=config.num_workers),
    }

    model = MODELS[config.model](pretrained=config.pretrained).to(device)
    backbone = {id(p) for p in model.backbone_parameters()}
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.parameters() if id(p) in backbone], 'lr': config.lr_backbone},
        {'params': [p for p in model.parameters() if id(p) not in backbone], 'lr': config.lr_head},
    ], weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')

    (out_dir / 'config.json').write_text(json.dumps(asdict(config), indent=2))
    history, best, stale = [], None, 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        sums = {}
        for batch in loaders['train']:
            batch = {k: v.to(device) if k.endswith('label') else v for k, v in batch.items()}
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device == 'cuda'):
                losses = masked_losses(forward(model, batch, device), batch, config)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses['total']).backward()
            scaler.step(optimizer)
            scaler.update()
            for name, value in losses.items():
                sums[name] = sums.get(name, 0.0) + float(value)
        scheduler.step()

        predictions = predict(model, loaders['val'], device)
        metrics = head_metrics(predictions)
        score = selection_score(metrics)
        row = {'epoch': epoch, **{f'train_{k}_loss': v / len(loaders['train']) for k, v in sums.items()},
               **{f'val_{k}': v for k, v in metrics.items()}, 'val_selection': score}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / 'history.csv', index=False)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)

        if best is None or score > best['val_selection']:
            best, stale = row, 0
            torch.save(model.state_dict(), out_dir / 'best.pt')
            predictions.to_csv(out_dir / 'val_predictions.csv', index=False)
        else:
            stale += 1
            if stale >= config.patience:
                break

    (out_dir / 'best_metrics.json').write_text(json.dumps(best, indent=2))
    return best
