"""Evaluate the DeepLabv3+ checkpoints released by the WeedsGalore authors (ckpts.zip on the dataset DOI page) with
the evaluation code of this project (src.weeds.train.evaluate, the function that scored our own runs). Two uses:
check our evaluation against the paper (Celikkan et al. 2025, Tables 3 and 4), and run the decision layer on the
authors' own model next to ours.

Their network is VainF's DeepLabV3Plus with the first convolution replaced to take `in_channels` bands (their
src/nets/deeplabv3plus and src/evaluate.py); src.weeds.model has the same layers under other names, so the state dict
is renamed, never converted, and loaded with strict=True. Inputs are scaled to [0, 1] and RGB uses the first three
bands (R, G, B), as in their dataset class. Each checkpoint is evaluated twice:
  float32  as in the authors' evaluate.py: the numbers to compare with the paper;
  float16  autocast, as our runs were evaluated: shows what float16 inference alone changes, and gives the
           probabilities for the decision layer, so it treats both models alike.
Run on Colab GPU (on CPU both modes are float32):

  python -m src.weeds.authors --data <dir with weedsgalore_<split>.npz> --ckpts <dir with dlv3p_*.pth> --out <dir>

For each checkpoint <out>/<name>/ gets segmentation_metrics.json ('val' and 'test' from float16 inference, 'fp32' with
both splits in float32); 3-class checkpoints also get probs_val.npz and probs_test.npz in the format of
src.weeds.train, so `python -m src.weeds.analyze --out <out>/<name>` works on them.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.weeds.analyze import clean
from src.weeds.data import BANDS, PAPER_TEST_3CLASS_MSI, load_npz
from src.weeds.model import DeepLabV3Plus
from src.weeds.train import evaluate

CKPT_URL = 'https://doidata.gfz.de/weedsgalore_e_celikkan_2024/ckpts.zip'
CKPT_ZIP_BYTES = 886908486  # size served by GFZ on 2026-11-05
# deterministic checkpoints in ckpts.zip: name -> (input bands, classes); the dlv3p_do_* (MC dropout) are not used
CKPTS = {'dlv3p_msi_3': (5, 3), 'dlv3p_rgb_3': (3, 3), 'dlv3p_msi_6': (5, 6), 'dlv3p_rgb_6': (3, 6)}
# Celikkan et al. 2025, DeepLabv3+, test IoU in %: Table 3 (3 classes) and Table 4 (6 classes)
PAPER = {
    'dlv3p_msi_3': PAPER_TEST_3CLASS_MSI,
    'dlv3p_rgb_3': {'miou': 79.33, 'background': 97.97, 'crop': 67.93, 'weed': 72.08},
    'dlv3p_msi_6': {'miou': 55.52, 'background': 98.37, 'maize': 73.03, 'amaranth': 76.17, 'barnyard_grass': 53.55,
                    'quickweed': 21.11, 'weed_other': 10.86},
    'dlv3p_rgb_6': {'miou': 50.81, 'background': 97.94, 'maize': 71.46, 'amaranth': 74.32, 'barnyard_grass': 45.95,
                    'quickweed': 6.26, 'weed_other': 8.92},
}
# their parameter-name prefixes -> ours (src.weeds.model)
RENAMES = [('backbone.conv1.', 'stem.0.'), ('backbone.bn1.', 'stem.1.'), ('backbone.layer', 'layer'),
           ('classifier.project.', 'low_proj.'), ('classifier.aspp.convs.', 'aspp.branches.'),
           ('classifier.aspp.project.', 'aspp.project.'), ('classifier.classifier.', 'classifier.')]


def rename_state_dict(sd):
    out = {}
    for k, v in sd.items():
        name = k[len('module.'):] if k.startswith('module.') else k
        for old, new in RENAMES:
            if name.startswith(old):
                out[new + name[len(old):]] = v
                break
        else:
            raise KeyError(f'unexpected parameter name in checkpoint: {k}')
    return out


def load_checkpoint(path, bands, classes, dev):
    # weights_only: a third-party file must not be able to run code when unpickled
    sd = torch.load(path, map_location='cpu', weights_only=True)
    model = DeepLabV3Plus(bands, classes, pretrained_backbone=False)
    model.load_state_dict(rename_state_dict(sd), strict=True)
    return model.to(dev).eval()


def per_class(res):
    return {k: v for k, v in res.items() if k != 'confusion'}


def run(data_dir, ckpt_dir, out_dir, dev):
    if BANDS[:3] != ('R', 'G', 'B'):
        raise ValueError('RGB checkpoints expect the first three bands to be R, G, B')
    missing = [n for n in CKPTS if not (Path(ckpt_dir) / f'{n}.pth').exists()]
    if missing:
        raise FileNotFoundError(f'checkpoints not found in {ckpt_dir}: {missing}')
    splits = {s: load_npz(Path(data_dir) / f'weedsgalore_{s}.npz') for s in ('val', 'test')}
    summary = {}
    for name, (bands, classes) in CKPTS.items():
        model = load_checkpoint(Path(ckpt_dir) / f'{name}.pth', bands, classes, dev)
        out = Path(out_dir) / name
        out.mkdir(parents=True, exist_ok=True)
        report = {'fp32': {}}
        for split, d in splits.items():
            report['fp32'][split] = evaluate(model, d, dev, classes=classes, bands=bands, amp=False)
            if classes == 3:
                res, probs = evaluate(model, d, dev, keep_probs=True, classes=3, bands=bands, amp=True)
                np.savez_compressed(out / f'probs_{split}.npz', ids=d['ids'], p_crop=probs[:, 0], p_weed=probs[:, 1])
            else:
                res = evaluate(model, d, dev, classes=classes, bands=bands, amp=True)
            report[split] = res
        paper = PAPER[name]
        report['paper_test'] = paper
        report['fp32_test_minus_paper_miou'] = report['fp32']['test']['miou'] - paper['miou']
        report['test_minus_paper_miou'] = report['test']['miou'] - paper['miou']
        (out / 'segmentation_metrics.json').write_text(json.dumps(clean(report), indent=1))
        summary[name] = {'bands': bands, 'classes': classes, 'paper': paper,
                         'fp32_test': per_class(report['fp32']['test']), 'fp16_test': per_class(report['test']),
                         'fp32_test_minus_paper_miou': report['fp32_test_minus_paper_miou'],
                         'fp16_test_minus_paper_miou': report['test_minus_paper_miou']}
        t32, t16 = report['fp32']['test'], report['test']
        print(f'{name}: test mIoU float32 {t32["miou"]:.2f}, float16 {t16["miou"]:.2f}, paper {paper["miou"]:.2f} '
              f'(float32 - paper {report["fp32_test_minus_paper_miou"]:+.2f}) | float32 / paper: ' +
              ', '.join(f'{c} {t32[c]:.2f} / {v:.2f}' for c, v in paper.items() if c != 'miou'))
        del model
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / 'authors_vs_paper.json').write_text(json.dumps(clean(summary), indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpts', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', dev, torch.cuda.get_device_name(0) if dev.type == 'cuda' else '')
    run(args.data, args.ckpts, args.out, dev)


if __name__ == '__main__':
    main()
