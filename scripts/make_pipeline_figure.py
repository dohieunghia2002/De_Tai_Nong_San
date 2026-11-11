"""Draw the pipeline figure of the decision support system on one real test tile: multispectral input, predicted
segmentation (output 1) and per-cell spray decisions (output 2), with example explanations. Decisions are recomputed
as in src.weeds.analyze for the primary setting (2-fold cross-fit by location on the pre-herbicide test tiles) and
checked against the run's go_no_go.json before drawing. Text in the figure is Vietnamese (the audience).

  python scripts/make_pipeline_figure.py [--run results/weeds_pilot_b8] [--tile 2023-05-30_0720]
"""
import argparse
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT_DIR), str(ROOT_DIR / 'scripts')]
from export_weeds_images import B, G, R, band_limits, composite, date_vi, font  # noqa: E402
from src.weeds.analyze import ALPHA, BETA, PRIMARY, primary_decisions, test_location_folds  # noqa: E402
from src.weeds.data import BANDS, SPLITS, TILE, load_split, split_ids, to_3class  # noqa: E402
from src.weeds.decision import DEFER, NO_SPRAY, SPRAY, cell_table, decision_metrics  # noqa: E402

GSD_M = 0.0025  # ground size of one pixel: 2.5 mm (Celikkan et al. 2025)
SEG = [('nền đất', (45, 45, 45)), ('ngô', (60, 200, 60)), ('cỏ dại', (235, 50, 50))]
DECISION = {SPRAY: ('PHUN', (215, 48, 39)), NO_SPRAY: ('KHÔNG PHUN', (49, 130, 189)),
            DEFER: ('CHƯA CHẮC', (245, 170, 0))}
DARK, GREY, TAG = (30, 30, 30), (110, 110, 110), (52, 73, 94)
IMG, GAP_X, MARGIN = 440, 270, 40
COLS = [MARGIN + i * (IMG + GAP_X) for i in range(3)]
W, H = COLS[-1] + IMG + MARGIN, 1100
Y_TAG, Y_TITLE, Y_IMG = 92, 128, 210


def num(v, digits=2):
    return f'{v:.{digits}f}'.replace('.', ',')


def pct(v):
    return num(100 * v)


def same(a, b):
    """Equal up to rounding, treating None (NaN written by analyze.clean) and NaN as the same missing value."""
    missing = [x is None or (isinstance(x, float) and math.isnan(x)) for x in (a, b)]
    return all(missing) if any(missing) else math.isclose(a, b, abs_tol=1e-9)


def decisions(zip_path, run):
    """Per-cell conformal_3way decisions on the pre-herbicide test tiles, with the thresholds applied to each cell,
    computed by the same cross-fit as src.weeds.analyze and checked against the run's go_no_go.json."""
    test = load_split(zip_path, 'test')
    with np.load(run / 'probs_test.npz') as p:
        if list(p['ids'].astype(str)) != list(test['ids']):
            raise ValueError('probabilities and labels are not in the same tile order')
        p_crop, p_weed = p['p_crop'].astype(np.float32), p['p_weed'].astype(np.float32)
    cells = cell_table(test['ids'], to_3class(test['y']), p_weed, PRIMARY['cell_size'])
    t = primary_decisions(cells, test_location_folds(list(test['ids'])))

    # the figure must show the same decisions as the reported results
    m = decision_metrics(t['decision'].to_numpy(), t['w'].to_numpy() >= PRIMARY['tau'], t['w'].to_numpy(),
                         t['crop'].to_numpy())
    rep = json.loads((run / 'go_no_go.json').read_text())
    reported = {'spray_reduction': rep['checks']['C2_spray_reduction']['value'],
                'deferred': rep['checks']['C3_deferred']['value'],
                'missed_need_rate': rep['info']['missed_need_rate'],
                'false_spray_rate': rep['info']['false_spray_rate']}
    for k, v in reported.items():
        if not same(m[k], v):
            raise ValueError(f'{k}: recomputed {m[k]} differs from go_no_go.json {v}')
    return test, p_crop, p_weed, t


def centred(d, cx, y, text, f, fill=DARK, **kw):
    d.text((cx - d.textlength(text, font=f) / 2, y), text, font=f, fill=fill, **kw)


def arrow(d, x0, x1, y):
    d.line([(x0, y), (x1 - 18, y)], fill=TAG, width=6)
    d.polygon([(x1, y), (x1 - 24, y - 14), (x1 - 24, y + 14)], fill=TAG)


def step_box(d, cx, bottom, lines, half_width):
    """Processing step drawn just above its arrow, so the arrow stays fully visible."""
    f_b, f = font(21, True), font(18)
    h = 34 + 26 * (len(lines) - 1) + 16
    d.rounded_rectangle([cx - half_width, bottom - h, cx + half_width, bottom], radius=14, fill=(236, 241, 246),
                        outline=TAG, width=3)
    y = bottom - h + 12
    for i, line in enumerate(lines):
        centred(d, cx, y, line, f_b if i == 0 else f)
        y += 30 if i == 0 else 26


def dashed_rect(d, box, dash=14, gap=9, width=3, fill=GREY):
    x0, y0, x1, y1 = box
    for a, b, horizontal, at in ((x0, x1, True, y0), (x0, x1, True, y1), (y0, y1, False, x0), (y0, y1, False, x1)):
        s = a
        while s < b:
            e = min(s + dash, b)
            d.line([(s, at), (e, at)] if horizontal else [(at, s), (at, e)], fill=fill, width=width)
            s = e + gap


def header(d, x, tag, title, suffix):
    f_tag = font(20, True)
    w = d.textlength(tag, font=f_tag)
    d.rounded_rectangle([x, Y_TAG, x + w + 24, Y_TAG + 32], radius=8, fill=TAG)
    d.text((x + 12, Y_TAG + 4), tag, font=f_tag, fill='white')
    d.text((x, Y_TITLE), title, font=font(25, True), fill=DARK)
    d.text((x + d.textlength(title + ' ', font=font(25, True)), Y_TITLE + 4), suffix, font=font(21), fill=GREY)


def captions(d, x, y, lines):
    for i, line in enumerate(lines):
        d.text((x, y + 27 * i), line, font=font(20), fill=DARK)


def legend(d, x, y, items):
    f = font(19)
    for name, colour in items:
        d.rectangle([x, y + 2, x + 22, y + 24], fill=colour, outline='black')
        d.text((x + 30, y), name, font=f, fill=DARK)
        x += 30 + d.textlength(name, font=f) + 26


def decision_map(rgb, cells, cell_px):
    """Field image with each grid cell tinted by its decision, numbered, and labelled with its predicted weed cover."""
    base = np.asarray(rgb, dtype=np.float32)
    tint = base.copy()
    n = base.shape[0] // cell_px
    for _, c in cells.iterrows():
        r, k = divmod(int(c['cell']), n)
        tint[r * cell_px:(r + 1) * cell_px, k * cell_px:(k + 1) * cell_px] = DECISION[c['decision']][1]
    im = Image.fromarray((0.55 * base + 0.45 * tint).round().astype(np.uint8))
    d = ImageDraw.Draw(im)
    for i in range(1, n):
        d.line([(i * cell_px, 0), (i * cell_px, n * cell_px)], fill='white', width=4)
        d.line([(0, i * cell_px), (n * cell_px, i * cell_px)], fill='white', width=4)
    for _, c in cells.iterrows():
        r, k = divmod(int(c['cell']), n)
        d.text((k * cell_px + 12, r * cell_px + 8), str(int(c['cell']) + 1), font=font(30, True), fill='white',
               stroke_width=3, stroke_fill='black')
        # one word per line, so 'KHÔNG PHUN' and 'CHƯA CHẮC' fit inside a cell
        lines = [(w, font(31, True)) for w in DECISION[c['decision']][0].split()] + [(f'cỏ {pct(c["s"])}%', font(27))]
        cx, y = (k + 0.5) * cell_px, (r + 0.5) * cell_px - 36 * len(lines) / 2 + 10
        for text, f in lines:
            centred(d, cx, y, text, f, fill='white', stroke_width=3, stroke_fill='black')
            y += 36
    return im


def explanation(c):
    """One sentence that states the rule of decision.decide_conformal for this cell: SPRAY iff s >= lam_lo and
    s > lam_hi, NO_SPRAY iff s < lam_lo and s <= lam_hi, DEFER otherwise. When the thresholds cross (lam_hi < lam_lo)
    the binding ones are lam_lo for spraying and lam_hi for not spraying, otherwise the reverse. lam_lo can be -inf
    and lam_hi +inf when calibration has too few cells; such a threshold never decides, so it only appears for DEFER."""
    lo, hi, s, d = c['lam_lo'], c['lam_hi'], c['s'], c['decision']
    crossed = hi < lo
    head = f'Ô {int(c["cell"]) + 1}, {DECISION[d][0]}: cỏ dại dự đoán chiếm {pct(s)}% diện tích ô'
    if d == SPRAY:
        return f'{head}, từ ngưỡng phun {pct(lo)}% trở lên.' if crossed else f'{head}, vượt ngưỡng phun {pct(hi)}%.'
    if d == NO_SPRAY:
        return (f'{head}, không vượt ngưỡng không phun {pct(hi)}%.' if crossed
                else f'{head}, dưới ngưỡng không phun {pct(lo)}%.')
    if not (math.isfinite(lo) and math.isfinite(hi)):
        missing = ' và '.join(n for n, v in (('không phun', lo), ('phun', hi)) if not math.isfinite(v))
        return f'{head}; phần hiệu chỉnh có quá ít ô để đặt ngưỡng {missing}: cần người ra kiểm tra.'
    no_spray, spray = min(lo, hi), max(lo, hi)
    return (f'{head}, nằm giữa ngưỡng không phun {pct(no_spray)}% và ngưỡng phun {pct(spray)}%: '
            f'cần người ra kiểm tra.')


def model_label(run):
    done = run / 'train_done.json'
    if done.exists():
        cfg = json.loads(done.read_text())
        return f'mô hình đề tài tự huấn luyện (batch {cfg.get("batch", "?")}, seed {cfg.get("seed", "?")})'
    return f'mô hình công bố của tác giả ({run.name})'


def g(v):
    return f'{v:g}'.replace('.', ',')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', default=str(ROOT_DIR / 'data/weedsgalore/weedsgalore-dataset.zip'))
    ap.add_argument('--run', default=str(ROOT_DIR / 'results/weeds_pilot_b8'))
    ap.add_argument('--tile', default='2023-05-30_0720')
    ap.add_argument('--out', default=str(ROOT_DIR / 'data/weedsgalore/so_do_pipeline.png'))
    args = ap.parse_args()
    run = Path(args.run)
    test, p_crop, p_weed, t = decisions(args.zip, run)
    i = list(test['ids']).index(args.tile)
    cells = t[t['tile'] == args.tile]
    if cells.empty:
        raise ValueError(f'{args.tile} is not a pre-herbicide test tile')
    miou = json.loads((run / 'segmentation_metrics.json').read_text())['test']['miou']
    cell_m, tile_m = PRIMARY['cell_size'] * GSD_M, TILE * GSD_M

    # dataset-wide stretch, as in the preview images (export_weeds_images.py), so the tile looks the same in both
    with zipfile.ZipFile(args.zip) as zf:
        lo, hi = band_limits(zf, sorted(t for s in SPLITS for t in split_ids(zf, s)))
    rgb = composite(test['x'][i], (R, G, B), lo, hi)
    pred = np.argmax(np.stack([1 - p_crop[i] - p_weed[i], p_crop[i], p_weed[i]]), axis=0)
    seg = Image.fromarray(np.array([c for _, c in SEG], dtype=np.uint8)[pred])
    dmap = decision_map(rgb, cells, PRIMARY['cell_size'])

    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)
    d.text((MARGIN, 26), 'Hệ thống hỗ trợ ra quyết định phun thuốc trừ cỏ từ ảnh UAV đa phổ: sơ đồ xử lý',
           font=font(34, True), fill=DARK)

    # input: the RGB composite in front of four offset layers standing for the other bands
    for k in range(4, 0, -1):
        o = 10 * k
        d.rectangle([COLS[0] + o, Y_IMG - o, COLS[0] + IMG + o, Y_IMG + IMG - o], fill=(215 - 22 * k,) * 3,
                    outline='white', width=2)
    img.paste(rgb.resize((IMG, IMG), Image.LANCZOS), (COLS[0], Y_IMG))
    img.paste(seg.resize((IMG, IMG), Image.LANCZOS), (COLS[1], Y_IMG))
    img.paste(dmap.resize((IMG, IMG), Image.LANCZOS), (COLS[2], Y_IMG))

    header(d, COLS[0], 'INPUT', 'Ảnh drone đa phổ', '')
    header(d, COLS[1], 'OUTPUT 1', 'Kết quả phân vùng', '(segmentation)')
    header(d, COLS[2], 'OUTPUT 2', 'Kết quả phân lớp từng ô', '(classification)')
    mid = Y_IMG + IMG / 2
    arrow(d, COLS[0] + IMG + 52, COLS[1] - 12, mid)
    arrow(d, COLS[1] + IMG + 12, COLS[2] - 12, mid)
    step_box(d, (COLS[0] + IMG + 40 + COLS[1]) / 2, mid - 18, ['Mô hình học sâu', 'DeepLabv3+', 'phân vùng ảnh'], 100)
    step_box(d, (COLS[1] + IMG + COLS[2]) / 2, mid - 18,
             ['Mô-đun quyết định', f'chia ô {g(cell_m)} m × {g(cell_m)} m,', 'so % cỏ dự đoán', 'với ngưỡng conformal'],
             115)

    y_cap = Y_IMG + IMG + 18
    captions(d, COLS[0], y_cap, [f'{len(BANDS)} dải phổ: {", ".join(BANDS)} (ảnh 16 bit)',
                                 f'{TILE} × {TILE} điểm ảnh ≈ {g(tile_m)} m × {g(tile_m)} m'])
    captions(d, COLS[1], y_cap, ['Mỗi điểm ảnh: nền đất / ngô / cỏ dại, kèm', 'xác suất → bản đồ vị trí cỏ dại'])
    captions(d, COLS[2], y_cap, [f'Mỗi ô {g(cell_m)} m × {g(cell_m)} m: phun / không phun /',
                                 'chưa chắc → bản đồ phun cho cả ruộng'])
    legend(d, COLS[1], y_cap + 66, SEG)
    legend(d, COLS[2], y_cap + 66, [v for v in DECISION.values()])

    # explanation examples: one cell per decision present on this tile
    y0 = y_cap + 126
    box = [MARGIN, y0, W - MARGIN, y0 + 206]
    dashed_rect(d, box)
    d.text((MARGIN + 20, y0 + 16), 'Lời giải thích theo quy tắc ngưỡng. Ví dụ trên ô ảnh này:',
           font=font(22, True), fill=DARK)
    examples = [cells[cells['decision'] == SPRAY].nlargest(1, 's'), cells[cells['decision'] == DEFER].head(1),
                cells[cells['decision'] == NO_SPRAY].nsmallest(1, 's')]
    y = y0 + 54
    for ex in examples:
        if ex.empty:
            continue
        c = ex.iloc[0]
        d.rectangle([MARGIN + 22, y + 3, MARGIN + 42, y + 23], fill=DECISION[c['decision']][1], outline='black')
        d.text((MARGIN + 54, y), explanation(c), font=font(20), fill=DARK)
        y += 34
    d.text((MARGIN + 22, y + 6), 'Mỗi ô PHUN và CHƯA CHẮC còn có giải thích "bỏ những vùng cỏ nào thì quyết định đổi", '
                                 'đã kiểm chứng bằng mô hình (hình giải thích phản thực).', font=font(20), fill=GREY)

    f_note = font(17)
    y = box[3] + 16
    d.text((MARGIN, y), f'Minh họa trên ô ảnh {args.tile} (tập test, bay ngày {date_vi(args.tile)}) với '
                        f'{model_label(run)}, mIoU test {num(miou)}%. Ngưỡng conformal hiệu chỉnh trên các vị trí test '
                        f'khác, α = {g(ALPHA)}, β = {g(BETA)}.', font=f_note, fill=GREY)
    d.text((MARGIN, y + 24), f'Ô cần phun nếu cỏ dại chiếm từ {g(100 * PRIMARY["tau"])}% diện tích ô (ngưỡng minh '
                             'họa, chưa có nguồn). Dữ liệu: WeedsGalore (Celikkan và cộng sự, WACV 2025), giấy phép '
                             'CC BY 4.0.', font=f_note, fill=GREY)
    out = Path(args.out)
    img.save(out)
    print(f'saved {out} ({out.stat().st_size / 1e6:.1f} MB); decisions on {args.tile}:',
          {DECISION[k][0]: int((cells['decision'] == k).sum()) for k in DECISION})


if __name__ == '__main__':
    main()
