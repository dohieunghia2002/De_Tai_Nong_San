"""Export every WeedsGalore tile as a viewable 8-bit JPEG panel (true colour, NIR false colour, label mask) plus one
overview sheet per acquisition date, for showing the dataset to readers without remote-sensing tools. Reads the dataset
zip only; brightness is a linear stretch between dataset-wide 1st and 99th percentiles per band, so dates stay
comparable. Labels in the images are in Vietnamese (the audience); fonts come from Windows (Arial covers Vietnamese).

  python scripts/export_weeds_images.py [--zip data/weedsgalore/weedsgalore-dataset.zip] [--out data/weedsgalore/anh_xem_truoc]
"""
import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from src.weeds.data import ROOT, SPLITS, read_tile, split_ids  # noqa: E402

FONT, FONT_BOLD = 'C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/arialbd.ttf'
R, G, B, NIR = 0, 1, 2, 3  # band indices in src.weeds.data.BANDS order
CLASSES = [  # WeedsGalore semantic classes 0-5: legend text and colour
    ('nền đất', (45, 45, 45)),
    ('ngô', (60, 200, 60)),
    ('dền (amaranth)', (235, 50, 50)),
    ('cỏ lồng vực (barnyard grass)', (255, 215, 0)),
    ('cúc dại (quickweed)', (235, 70, 235)),
    ('cỏ khác', (0, 200, 235)),
]
SPLIT_VI = {'train': 'tập huấn luyện (train)', 'val': 'tập kiểm định (val)', 'test': 'tập kiểm tra (test)'}
TILE, GAP, THUMB = 600, 12, 220
README = '''Bộ dữ liệu WeedsGalore: ảnh xem trước của toàn bộ 156 ô ảnh

Nguồn dữ liệu: Celikkan và cộng sự, WACV 2025 (IEEE/CVF).
Tải dữ liệu gốc: https://doidata.gfz.de/weedsgalore_e_celikkan_2024/
Giấy phép: Creative Commons Attribution 4.0 (CC BY 4.0), bản đầy đủ trong LICENSE.txt.

Thay đổi so với dữ liệu gốc: dữ liệu gốc là 5 dải phổ riêng (ảnh PNG 16 bit, trình xem ảnh thông thường hiển thị gần
như đen). Ở đây các dải được ghép thành ảnh màu 8 bit và kéo giãn độ sáng (tuyến tính, giữa phân vị 1% và 99% của toàn
bộ dữ liệu, từng dải), chỉ để xem. Dữ liệu gốc không thay đổi.

Bối cảnh:
- Một ruộng ngô ở Potsdam (Đức), năm 2023. Drone DJI P4 Multispectral bay ở độ cao 5 m, mỗi điểm ảnh 2,5 mm.
- 4 đợt bay: 25/5, 30/5, 6/6 (mỗi đợt 48 ô ảnh) và 15/6 (12 ô ảnh). Ruộng được phun thuốc trừ cỏ ngày 12/6.
- Mỗi ô ảnh 600 x 600 điểm ảnh, tương đương 1,5 m x 1,5 m ngoài ruộng.
- Mỗi ô ảnh có 5 dải phổ: đỏ (R), lục (G), lam (B), cận hồng ngoại (NIR), rìa đỏ (RE), và một mặt nạ nhãn 6 lớp.
- Chia tập: 104 ô huấn luyện (train), 26 ô kiểm định (val), 26 ô kiểm tra (test), chia theo vị trí trên ruộng.

Cách xem:
- 0_tong_quan_<ngày>.jpg: toàn bộ ô ảnh của một đợt bay (ảnh màu thật, thu nhỏ), kèm số hiệu và tập dữ liệu.
- Thư mục <ngày>/: mỗi ô ảnh một file, độ phân giải đầy đủ, gồm 3 hình cạnh nhau:
  1. ảnh màu thật (ghép 3 dải R, G, B);
  2. ảnh màu giả (ghép 3 dải NIR, R, G, cách hiển thị thường dùng trong viễn thám; thực vật hiện màu đỏ);
  3. nhãn: nền đất xám đậm, ngô xanh lá, dền đỏ, cỏ lồng vực vàng, cúc dại tím hồng, cỏ khác xanh lơ.
- Trong bài toán của đề tài, mọi loài cỏ được gộp thành một lớp "cỏ dại".
'''


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def date_vi(tile_id):
    y, m, d = tile_id[:10].split('-')
    return f'{d}/{m}/{y}'


def band_limits(zf, ids):
    """Dataset-wide 1st / 99th percentile per band, on every 10th pixel of every tile."""
    sample = np.stack([read_tile(zf, t)[0][:, ::10, ::10] for t in ids])
    return np.percentile(sample, 1, axis=(0, 2, 3)), np.percentile(sample, 99, axis=(0, 2, 3))


def composite(x, bands, lo, hi):
    out = [(x[b].astype(np.float32) - lo[b]) / (hi[b] - lo[b]) for b in bands]
    return Image.fromarray((np.clip(np.stack(out, -1), 0, 1) * 255).round().astype(np.uint8))


def label_image(mask):
    return Image.fromarray(np.array([c for _, c in CLASSES], dtype=np.uint8)[mask])


def panel(tile_id, split, x, mask, lo, hi):
    weed = 100 * float((mask >= 2).mean())
    head, title, legend = 56, 40, 52
    w = 3 * TILE + 4 * GAP
    img = Image.new('RGB', (w, head + title + TILE + legend), 'white')
    d = ImageDraw.Draw(img)
    d.text((GAP, 14), f'{tile_id}   |   bay ngày {date_vi(tile_id)}   |   {SPLIT_VI[split]}   |   ô ảnh 1,5 m × 1,5 m'
                      f'   |   cỏ dại chiếm {weed:.1f}% diện tích'.replace('.', ','), fill='black', font=font(24, True))
    views = [('Ảnh màu thật (R, G, B)', composite(x, (R, G, B), lo, hi)),
             ('Ảnh màu giả (NIR, R, G): thực vật hiện màu đỏ', composite(x, (NIR, R, G), lo, hi)),
             ('Nhãn (mặt nạ phân vùng)', label_image(mask))]
    for i, (name, im) in enumerate(views):
        x0 = GAP + i * (TILE + GAP)
        d.text((x0, head + 6), name, fill='black', font=font(22))
        img.paste(im, (x0, head + title))
    lx, ly = GAP, head + title + TILE + 14
    for name, colour in CLASSES:
        d.rectangle([lx, ly, lx + 24, ly + 24], fill=colour, outline='black')
        d.text((lx + 32, ly), name, fill='black', font=font(20))
        lx += 32 + int(d.textlength(name, font=font(20))) + 34
    return img


def overview(date, tiles):
    """tiles: list of (tile_id, split, true-colour PIL image), sorted by id."""
    cols = 8
    rows = -(-len(tiles) // cols)
    head, cap = 70, 30
    img = Image.new('RGB', (GAP + cols * (THUMB + GAP), head + rows * (THUMB + cap + GAP)), 'white')
    d = ImageDraw.Draw(img)
    d.text((GAP, 18), f'WeedsGalore, đợt bay {date_vi(date)}: {len(tiles)} ô ảnh (mỗi ô 1,5 m × 1,5 m), ảnh màu thật',
           fill='black', font=font(28, True))
    for k, (tile_id, split, im) in enumerate(tiles):
        x0, y0 = GAP + (k % cols) * (THUMB + GAP), head + (k // cols) * (THUMB + cap + GAP)
        img.paste(im.resize((THUMB, THUMB), Image.LANCZOS), (x0, y0))
        d.text((x0, y0 + THUMB + 3), f'{tile_id[11:]}  ({split})', fill='black', font=font(20))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', default=str(ROOT_DIR / 'data/weedsgalore/weedsgalore-dataset.zip'))
    ap.add_argument('--out', default=str(ROOT_DIR / 'data/weedsgalore/anh_xem_truoc'))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip) as zf:
        split_of = {t: s for s in SPLITS for t in split_ids(zf, s)}
        ids = sorted(split_of)
        lo, hi = band_limits(zf, ids)
        print('stretch limits per band (1st / 99th percentile):', lo.tolist(), hi.tolist())
        by_date = {}
        for t in ids:
            x, mask = read_tile(zf, t)
            (out / t[:10]).mkdir(exist_ok=True)
            panel(t, split_of[t], x, mask, lo, hi).save(out / t[:10] / f'{t}.jpg', quality=88)
            by_date.setdefault(t[:10], []).append((t, split_of[t], composite(x, (R, G, B), lo, hi)))
        (out / 'LICENSE.txt').write_bytes(zf.read(f'{ROOT}/LICENSE.txt'))
    for date, tiles in by_date.items():
        overview(date, tiles).save(out / f'0_tong_quan_{date}.jpg', quality=85)
    (out / 'README.txt').write_text(README, encoding='utf-8')
    n = sum(len(v) for v in by_date.values())
    print(f'{n} tile panels and {len(by_date)} overview sheets written to {out}')


if __name__ == '__main__':
    main()
