"""Read the WeedsGalore dataset zip and convert each split into compact arrays.

Band order R, G, B, NIR, RE and the 3-class merge (0 background, 1 maize, 2 any weed) follow the authors' loader
(github.com/GFZ/weedsgalore, src/datasets/weedsgalore/weedsgalore.py). Images stay uint16; dividing by 65535 gives the
[0, 1] scaling that their matplotlib-based reader produces.
"""
import io
import zipfile

import numpy as np
from PIL import Image

ROOT = 'weedsgalore-dataset'
BANDS = ('R', 'G', 'B', 'NIR', 'RE')
SPLITS = ('train', 'val', 'test')
DATES = ('2023-05-25', '2023-05-30', '2023-06-06', '2023-06-15')
PRE_HERBICIDE_DATES = DATES[:3]  # herbicide was applied on 12 June 2023 (Celikkan et al. 2025, 3.1)
EXPECTED_TILES = {'train': 104, 'val': 26, 'test': 26}
EXPECTED_TEST_LOCATIONS = 8  # 2 field patches x 4 sampled points (Celikkan et al. 2025, 3.2-3.3)
TILE = 600
SEMANTIC_CLASSES = ('background', 'maize', 'amaranth', 'barnyard_grass', 'quickweed', 'weed_other')
CLASSES3 = ('background', 'crop', 'weed')
# Celikkan et al. 2025, Table 3: DeepLabv3+ with 5-band (MSI) input, 3-class setting, test IoU in %
PAPER_TEST_3CLASS_MSI = {'miou': 82.90, 'background': 98.45, 'crop': 72.93, 'weed': 77.31}


def split_ids(zf, split):
    return zf.read(f'{ROOT}/splits/{split}.txt').decode().split()


def tile_date(tile_id):
    return tile_id[:10]


def read_png(zf, name):
    return np.array(Image.open(io.BytesIO(zf.read(name))))


def read_tile(zf, tile_id):
    """(5, 600, 600) uint16 image in BANDS order and (600, 600) uint8 semantic mask with values 0-5."""
    base = f'{ROOT}/{tile_date(tile_id)}'
    bands = [read_png(zf, f'{base}/images/{tile_id}_{b}.png') for b in BANDS]
    for b, a in zip(BANDS, bands):
        if a.dtype != np.uint16:  # checked before stacking: an 8-bit band would otherwise be scaled by 65535 silently
            raise ValueError(f'{tile_id} band {b}: expected 16-bit PNG, got {a.dtype}')
    mask = read_png(zf, f'{base}/semantics/{tile_id}.png')
    return np.stack(bands), mask.astype(np.uint8)


def location_blocks(ids):
    """Group the tile ids of a split file into field locations. In the val and test files each location is written
    as a block of consecutive ids with decreasing dates (checked visually for the test split on 2026-10-29: shared
    field markers and tyre tracks across dates). Returns a list of blocks."""
    blocks, cur = [], []
    for t in ids:
        if cur and tile_date(t) >= tile_date(cur[-1]):
            blocks.append(cur)
            cur = []
        cur.append(t)
    if cur:
        blocks.append(cur)
    return blocks


def to_3class(mask):
    """0 background, 1 maize, 2 weed (all weed species merged), as in the authors' binary_label."""
    out = mask.copy()
    out[out > 1] = 2
    return out


def load_split(zip_path, split):
    with zipfile.ZipFile(zip_path) as zf:
        ids = split_ids(zf, split)
        tiles = [read_tile(zf, t) for t in ids]
    x = np.stack([t[0] for t in tiles])
    y = np.stack([t[1] for t in tiles])
    check_split(split, ids, x, y)
    return {'ids': np.array(ids), 'x': x, 'y': y}


def check_split(split, ids, x, y):
    n = EXPECTED_TILES[split]
    if len(ids) != n or x.shape != (n, len(BANDS), TILE, TILE) or y.shape != (n, TILE, TILE):
        raise ValueError(f'{split}: unexpected shapes ids={len(ids)} x={x.shape} y={y.shape}')
    if x.dtype != np.uint16 or y.max() > len(SEMANTIC_CLASSES) - 1:
        raise ValueError(f'{split}: unexpected dtype {x.dtype} or label values up to {y.max()}')
    if not set(tile_date(t) for t in ids) <= set(DATES):
        raise ValueError(f'{split}: unknown acquisition date in ids')


def build_npz(zip_path, out_dir):
    """Write weedsgalore_<split>.npz (ids, x uint16, y uint8 semantic) for each split; returns the paths."""
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in SPLITS:
        d = load_split(zip_path, split)
        paths[split] = out_dir / f'weedsgalore_{split}.npz'
        np.savez_compressed(paths[split], **d)
        print(f'{split}: {len(d["ids"])} tiles, x {d["x"].shape} {d["x"].dtype}, saved {paths[split]}')
    return paths


def load_npz(path):
    with np.load(path) as d:
        return {'ids': d['ids'].astype(str), 'x': d['x'], 'y': d['y']}
