"""Train DeepLabv3+ on WeedsGalore (3 classes, 5-band input), select it on validation mIoU, evaluate on test and save
per-pixel class probabilities for the decision layer. Settings follow Celikkan et al. 2025 (4.1.1) and their code:
Adam lr 1e-3, unweighted cross-entropy, rot90 / flip / Gaussian jitter (sd 1/25) augmentation, inputs scaled to [0, 1].
Mixed precision is used to fit 600x600 tiles on a T4. Run on Colab GPU:

  python -m src.weeds.train --data <dir with weedsgalore_<split>.npz> --out <dir> [--stage train|test|all] [--smoke]

Training resumes from <out>/last.pt and stops safely before --budget-hours.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.weeds.data import CLASSES3, PAPER_TEST_3CLASS_MSI, SEMANTIC_CLASSES, load_npz, to_3class
from src.weeds.model import DeepLabV3Plus


class TileDataset(Dataset):
    def __init__(self, d, augment):
        self.x, self.y, self.augment = d['x'], to_3class(d['y']), augment

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = self.x[i].astype(np.float32) / 65535.0
        lab = self.y[i]
        if self.augment:
            k = np.random.randint(4)
            img, lab = np.rot90(img, k, axes=(1, 2)), np.rot90(lab, k)
            flip = np.random.randint(3)  # 0 none, 1 vertical, 2 horizontal
            if flip == 1:
                img, lab = img[:, ::-1, :], lab[::-1, :]
            elif flip == 2:
                img, lab = img[:, :, ::-1], lab[:, ::-1]
            img = img + np.random.randn(*img.shape).astype(np.float32) / 25
        return (torch.from_numpy(np.ascontiguousarray(img)),
                torch.from_numpy(np.ascontiguousarray(lab).astype(np.int64)))


def seed_worker(worker_id):
    # each DataLoader worker would otherwise inherit the same numpy RNG state and repeat the same augmentations
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def confusion(pred, lab, n=3):
    """cm[true, pred] accumulated over all pixels."""
    return torch.bincount((lab * n + pred).flatten(), minlength=n * n).reshape(n, n)


def iou_from_confusion(cm, names=CLASSES3):
    """Per-class IoU and mIoU in %. A class absent from both labels and predictions gets NaN and is left out of the
    mIoU (torchmetrics, used by the authors, would count it as 0; every class occurs in the test split)."""
    cm = cm.double()
    tp = cm.diag()
    denom = cm.sum(0) + cm.sum(1) - tp
    iou = torch.where(denom > 0, tp / denom.clamp(min=1), torch.full_like(tp, float('nan')))
    res = {c: float(100 * v) for c, v in zip(names, iou)}
    res['miou'] = float(100 * torch.nanmean(iou))
    return res


@torch.no_grad()
def evaluate(model, d, dev, batch=2, keep_probs=False, limit=None, classes=3, bands=None, amp=True):
    """classes: 3 (labels merged by to_3class) or 6 (semantic labels); bands: use only the first `bands` input bands
    (3 = R, G, B); amp: float16 autocast on GPU as in training, False = float32 as in the authors' evaluate.py.
    keep_probs (3 classes only) also returns P(crop), P(weed) per pixel in float16."""
    if keep_probs and classes != 3:
        raise ValueError('probabilities are kept for the 3-class setting only')
    model.eval()
    n = len(d['x']) if limit is None else min(limit, len(d['x']))
    y = to_3class(d['y'][:n]) if classes == 3 else d['y'][:n]
    cm = torch.zeros(classes, classes, dtype=torch.long, device=dev)
    probs = []
    for i in range(0, n, batch):
        x = torch.from_numpy(d['x'][i:i + batch, :bands].astype(np.float32) / 65535.0).to(dev)
        lab = torch.from_numpy(y[i:i + batch].astype(np.int64)).to(dev)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp and dev.type == 'cuda'):
            logits = model(x)
        p = torch.softmax(logits.float(), dim=1)
        cm += confusion(p.argmax(1), lab, classes)
        if keep_probs:
            probs.append(p[:, 1:].half().cpu().numpy())  # P(crop), P(weed)
    res = iou_from_confusion(cm.cpu(), CLASSES3 if classes == 3 else SEMANTIC_CLASSES)
    res['confusion'] = cm.cpu().tolist()
    return (res, np.concatenate(probs)) if keep_probs else res


def torch_load(path, dev):
    try:
        return torch.load(path, map_location=dev, weights_only=False)  # our own checkpoint, not third-party
    except TypeError:
        return torch.load(path, map_location=dev)


def grad_scaler():
    try:
        return torch.amp.GradScaler('cuda')
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def train(args, dev):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'train_done.json').exists():
        print('training already finished:', (out / 'train_done.json').read_text())
        return
    tr, va = load_npz(Path(args.data) / 'weedsgalore_train.npz'), load_npz(Path(args.data) / 'weedsgalore_val.npz')
    model = DeepLabV3Plus(5, 3, pretrained_backbone=not args.no_pretrained).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = grad_scaler()
    start, best, bad, history = 0, -1.0, 0, []
    if (out / 'last.pt').exists():
        ck = torch_load(out / 'last.pt', dev)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        scaler.load_state_dict(ck['scaler'])
        start, best, bad, history = ck['epoch'] + 1, ck['best_miou'], ck['bad'], ck['history']
        print(f'resumed after epoch {ck["epoch"]} (best val mIoU {best:.2f})')
    if (out / 'best.json').exists():
        # best.pt may be newer than last.pt (last.pt is written every few epochs): never let a worse model replace it
        best = max(best, json.loads((out / 'best.json').read_text())['val_miou'])
    loader = DataLoader(TileDataset(tr, augment=True), batch_size=args.batch, shuffle=True, drop_last=True,
                        num_workers=args.workers, worker_init_fn=seed_worker, pin_memory=dev.type == 'cuda',
                        generator=torch.Generator().manual_seed(args.seed + start))
    def save_last(epoch):
        tmp = out / 'last.tmp'
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
                    'epoch': epoch, 'best_miou': best, 'bad': bad, 'history': history}, tmp)
        tmp.replace(out / 'last.pt')

    t0, finished, reason, epoch = time.time(), False, '', None
    for epoch in range(start, args.epochs):
        model.train()
        te, losses = time.time(), []
        for it, (x, y) in enumerate(loader):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            if args.max_iters and it + 1 >= args.max_iters:
                break
        val = evaluate(model, va, dev, limit=args.limit_val)
        secs = time.time() - te
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'val_miou': val['miou'],
                        **{f'val_{c}': val[c] for c in CLASSES3}, 'seconds': secs})
        if val['miou'] > best:
            best, bad = val['miou'], 0
            tmp = out / 'best.tmp'
            torch.save(model.state_dict(), tmp)
            tmp.replace(out / 'best.pt')
            (out / 'best.json').write_text(json.dumps({'epoch': epoch, 'val_miou': best}))
        else:
            bad += 1
        if (epoch + 1) % args.ckpt_every == 0:
            save_last(epoch)
        (out / 'history.json').write_text(json.dumps(history, indent=1))
        mem = f' | peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB' if dev.type == 'cuda' else ''
        print(f'epoch {epoch:3d} | loss {history[-1]["train_loss"]:.4f} | val mIoU {val["miou"]:.2f} '
              f'(crop {val["crop"]:.1f}, weed {val["weed"]:.1f}) | best {best:.2f} | {secs:.0f} s{mem}', flush=True)
        if bad >= args.patience:
            finished, reason = True, f'early stopping: no val improvement for {args.patience} epochs'
            break
        if time.time() - t0 + 1.5 * secs > args.budget_hours * 3600:
            reason = f'time budget of {args.budget_hours} h reached: rerun to resume'
            break
    else:
        finished, reason = True, f'reached {args.epochs} epochs'
    if epoch is not None:
        save_last(epoch)  # always checkpoint on exit, whatever stopped the loop
    print(reason)
    if finished and not args.smoke:
        (out / 'train_done.json').write_text(json.dumps({'reason': reason, 'best_val_miou': best,
                                                         'epochs_run': len(history), 'batch': args.batch,
                                                         'lr': args.lr, 'seed': args.seed}))


def test(args, dev):
    out = Path(args.out)
    model = DeepLabV3Plus(5, 3, pretrained_backbone=False).to(dev)
    model.load_state_dict(torch_load(out / 'best.pt', dev))
    report = {}
    for split in ('val', 'test'):
        d = load_npz(Path(args.data) / f'weedsgalore_{split}.npz')
        res, probs = evaluate(model, d, dev, keep_probs=True, limit=args.limit_val)
        n = len(probs)
        np.savez_compressed(out / f'probs_{split}.npz', ids=d['ids'][:n], p_crop=probs[:, 0], p_weed=probs[:, 1])
        report[split] = res
        print(f'{split}: mIoU {res["miou"]:.2f} | background {res["background"]:.2f}, crop {res["crop"]:.2f}, '
              f'weed {res["weed"]:.2f}')
    report['paper_test_3class_msi'] = PAPER_TEST_3CLASS_MSI
    report['test_minus_paper_miou'] = report['test']['miou'] - PAPER_TEST_3CLASS_MSI['miou']
    (out / 'segmentation_metrics.json').write_text(json.dumps(report, indent=1))
    print(f'test mIoU {report["test"]["miou"]:.2f} vs paper {PAPER_TEST_3CLASS_MSI["miou"]:.2f} '
          f'(diff {report["test_minus_paper_miou"]:+.2f})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stage', choices=['train', 'test', 'all'], default='all')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--batch', type=int, default=2, help='authors\' code default; the paper reports 8 on an A100')
    ap.add_argument('--ckpt-every', type=int, default=5, help='write last.pt (about 0.5 GB) every N epochs and on exit')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--budget-hours', type=float, default=3.5)
    ap.add_argument('--max-iters', type=int, default=0, help='stop each epoch after this many iterations (smoke test)')
    ap.add_argument('--limit-val', type=int, default=None, help='evaluate only the first N tiles (smoke test)')
    ap.add_argument('--smoke', action='store_true', help='1 epoch of 2 iterations, 2 evaluation tiles')
    ap.add_argument('--no-pretrained', action='store_true', help='random backbone (local tests only; skips download)')
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.max_iters, args.limit_val, args.patience = 1, 2, 2, 1
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', dev, torch.cuda.get_device_name(0) if dev.type == 'cuda' else '')
    if args.stage in ('train', 'all'):
        train(args, dev)
    if args.stage in ('test', 'all'):
        test(args, dev)


if __name__ == '__main__':
    main()
