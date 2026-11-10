"""Draw the counterfactual explanations of a few spraying decisions, as checked by the model (results/weeds_cfcheck).

For each example cell, one row: the tile around it; the cell with the patches the explanation names; the model's
P(weed) map of the cell; the cell's predicted weed area after removing 0, 1, 2, ... of those patches (the model's
rescoring next to the fast estimate of src.weeds.counterfactual) against the thresholds; the spectral bands behind
the score (integrated gradients from bare soil, results/weeds_xai); and the explanation in words. The examples are
picked by fixed rules (pick_examples), not by eye. Before drawing, the cell scores of the saved probability map are
checked against the Colab run and the patch rankings against results/weeds_counterfactual. Text is Vietnamese (the
audience).

  python scripts/make_counterfactual_figure.py [--out data/weedsgalore/hinh_giai_thich_phan_thuc.png]
"""
import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT_DIR), str(ROOT_DIR / 'scripts')]
from export_weeds_images import B, G, R, band_limits, composite, date_vi  # noqa: E402
from src.weeds.analyze import PRIMARY  # noqa: E402
from src.weeds.counterfactual import GSD_CM, PATCH  # noqa: E402
from src.weeds.data import BANDS, SPLITS, TILE, read_tile, split_ids  # noqa: E402
from src.weeds.decision import DEFER, NO_SPRAY, SPRAY  # noqa: E402

CELL = PRIMARY['cell_size']
CELL_CM2 = (CELL * GSD_CM) ** 2  # 2500 cm^2: a 50 cm x 50 cm cell
PATCH_CM = PATCH * GSD_CM
# the decision colours of make_pipeline_figure.py, with the amber darkened to #c98500: as a set they pass the
# dataviz palette checks on white (lightness band, chroma, colour-blind and normal-vision separation, 3:1 contrast)
DECISION = {SPRAY: ('PHUN', '#d7301f'), NO_SPRAY: ('KHÔNG PHUN', '#3182bd'), DEFER: ('CHƯA CHẮC', '#c98500')}
INK, INK2, MUTED, GRID, BAR = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#2a78d6'
SCORE_TOL = 1e-5


def num(v, digits=0):
    return f'{v:.{digits}f}'.replace('.', ',')


def pick_examples(cells):
    """The example cells, by fixed rules on the model-checked table (cfcheck_cells.csv): (key, row) pairs.
      fragile  a SPRAY cell that one removed patch already takes off SPRAY, the one with the most predicted weed;
      defer    a DEFER cell that one removed patch takes to NO_SPRAY (the usual case), the one with the most weed;
      short    a SPRAY cell of the last flight before spraying where the estimate falls short of the model by two
               patches or more, the median by the model's k.
    Cells whose decision changes back while patches are removed are left out: their story is not one sentence."""
    steady = ~cells['reverts_to_spray'].astype(bool) & ~cells['reverts_from_no_spray'].astype(bool)
    spray, defer = cells['decision_rescored'] == SPRAY, cells['decision_rescored'] == DEFER
    fragile = cells[spray & steady & (cells['k_not_spray_model'] == 1)]
    defer1 = cells[defer & steady & (cells['k_no_spray_model'] == 1)]
    short = cells[spray & steady & (cells['date'] == cells['date'].max())
                  & (cells['k_not_spray_model'] - cells['k_not_spray_estimate'] >= 2)]
    short = short.sort_values(['k_not_spray_model', 'tile', 'cell'])
    picks = [('fragile', fragile.sort_values(['s_model', 'tile', 'cell'], ascending=[False, True, True]).head(1)),
             ('defer', defer1.sort_values(['s_model', 'tile', 'cell'], ascending=[False, True, True]).head(1)),
             ('short', short.iloc[[(len(short) - 1) // 2]] if len(short) else short)]
    return [(key, rows.iloc[0]) for key, rows in picks if len(rows)]


def thresholds_cm2(c):
    """(no-spray threshold, spray threshold) in cm^2 of predicted weed. Under decision.decide_conformal a cell is
    SPRAY iff s >= lam_lo and s > lam_hi, NO_SPRAY iff s < lam_lo and s <= lam_hi: whichever way the thresholds lie,
    the larger one bounds SPRAY and the smaller one NO_SPRAY."""
    lo, hi = c['lam_lo'], c['lam_hi']
    return min(lo, hi) * CELL_CM2, max(lo, hi) * CELL_CM2


def removed_cm2(steps, k):
    """Predicted weed area in the first k ranked patches: s(0) - estimate(k), in cm^2."""
    s = steps.set_index('k')['s_estimate']
    return (s.loc[0] - s.loc[k]) * CELL_CM2


def text_lines(key, c, steps, add_to_spray_cm2):
    """Three lines: why this decision, what would change it (checked by the model), bands and the true label."""
    name = DECISION[int(c['decision_rescored'])][0]
    weed = c['s_model'] * CELL_CM2
    no_thr, sp_thr = thresholds_cm2(c)
    where = (f'từ ngưỡng phun {num(sp_thr)} cm² trở lên' if c['decision_rescored'] == SPRAY else
             f'nằm giữa ngưỡng không phun {num(no_thr)} cm² và ngưỡng phun {num(sp_thr)} cm²')
    why = (f'Vì sao {name}: mô hình thấy khoảng {num(weed)} cm² cỏ trong ô 50 cm × 50 cm '
           f'({num(100 * c["s_model"], 1)}% diện tích), {where}.')
    k_no = int(c['k_no_spray_model'])
    if c['decision_rescored'] == SPRAY:
        k = int(c['k_not_spray_model'])
        after = int(steps.set_index('k').loc[k, 'decision_model'])
        then = (f'chuyển thẳng sang KHÔNG PHUN' if after == NO_SPRAY else
                f'chuyển sang {DECISION[after][0]}; bỏ {k_no} vùng thì KHÔNG PHUN')
        what = (f'Đổi gì thì quyết định đổi (mô hình đã chấm lại): bỏ {patch_words(k)} ({num(removed_cm2(steps, k))} '
                f'cm² cỏ dự đoán) thì ô {then}. Ước lượng nhanh: {int(c["k_not_spray_estimate"])} vùng.')
    else:
        what = (f'Đổi gì thì quyết định đổi (mô hình đã chấm lại): bỏ {patch_words(k_no)} '
                f'({num(removed_cm2(steps, k_no), 1)} cm² cỏ dự đoán) thì ô thành KHÔNG PHUN; thêm khoảng '
                f'{num(add_to_spray_cm2, 1)} cm² cỏ thì PHUN (theo quy tắc ngưỡng). Người kiểm tra chỉ cần ra xem '
                f'{"vùng" if k_no == 1 else f"{k_no} vùng"} này.')
    cm2 = band_cm2(c)
    up = sorted((b for b in BANDS if cm2[b] > 0), key=cm2.get, reverse=True)[:2]
    down = min(BANDS, key=cm2.get)
    bands = ' và '.join(f'{b} ({signed(cm2[b])} cm²)' for b in up) + ' đẩy lượng cỏ dự đoán lên nhiều nhất'
    if cm2[down] < 0:
        bands += f'; {down} kéo xuống nhiều nhất ({signed(cm2[down])} cm²)'
    need = c['w'] >= PRIMARY['tau']
    truth = ('ô thực sự cần phun' if need else 'ô thực ra không cần phun') + f' (ngưỡng minh họa {num(100 * PRIMARY["tau"])}%)'
    tail = f'Dải phổ (IG): {bands}. Nhãn thật: cỏ chiếm {num(100 * c["w"], 2)}% diện tích ô → {truth}.'
    return [why, what, tail]


NUMBERED = 9  # patches are numbered on the images up to this many


def patch_words(k):
    if k == 1:
        return 'vùng số 1'
    return f'{k} vùng số 1–{k}' if k <= NUMBERED else f'{k} vùng có viền'


def band_cm2(c):
    """Integrated gradients per band in cm^2 of predicted weed. They add up to s(tile) - s(bare soil), and the
    bare-soil score is about 0 (results/weeds_xai), so the bands share out the cell's predicted weed area."""
    return {b: c[f'ig_{b}'] * CELL_CM2 for b in BANDS}


def signed(v):
    return ('+' if v >= 0 else '−') + num(abs(v))


TITLES = {'fragile': 'Ví dụ {i}. Quyết định PHUN mong manh: bỏ một vùng cỏ 5 cm là đổi quyết định',
          'defer': 'Ví dụ {i}. Ô CHƯA CHẮC: chỉ cần ra kiểm tra một vùng 5 cm',
          'short': 'Ví dụ {i}. Ô nhiều cỏ ngày {date}: ước lượng nhanh thiếu {gap} vùng so với mô hình chấm lại'}


def patch_boxes(c):
    """The scanned patches of the cell, in the explanation's order, as (x, y) pixel corners inside the cell."""
    side, grid_cols = CELL // PATCH, TILE // PATCH
    r, col = divmod(int(c['cell']), TILE // CELL)
    out = []
    for p in str(c['patches']).split():
        i, j = divmod(int(p), grid_cols)
        out.append(((j - col * side) * PATCH, (i - r * side) * PATCH))
    return out


def draw_patches(ax, boxes, k_solid, k_dashed, number):
    import matplotlib.patheffects as pe
    from matplotlib.patches import Rectangle
    halo = [pe.withStroke(linewidth=4, foreground='black')]
    for n, (x, y) in enumerate(boxes[:k_dashed]):
        solid = n < k_solid
        rect = Rectangle((x - 0.5, y - 0.5), PATCH, PATCH, fill=False, edgecolor='white', linewidth=2,
                         linestyle='-' if solid else (0, (3, 2)))
        rect.set_path_effects(halo)
        ax.add_patch(rect)
        if number:
            ax.text(x + PATCH / 2 - 0.5, y + PATCH / 2 - 0.5, str(n + 1), color='white', fontsize=9, ha='center',
                    va='center', fontweight='bold', path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])


def clean_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_row(sub, i, key, c, steps, rgb, p_cell, add_to_spray_cm2):
    import matplotlib.patheffects as pe
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator
    name, colour = DECISION[int(c['decision_rescored'])]
    gap = int(c['k_not_spray_model'] - c['k_not_spray_estimate'])
    sub.suptitle(TITLES[key].format(i=i, date=date_vi(c['tile'])[:5], gap=gap), x=0.012, ha='left', fontsize=15,
                 fontweight='bold', color=INK)
    axs = sub.subplots(1, 5, gridspec_kw={'width_ratios': [1, 1, 1.12, 1.75, 1.05], 'wspace': 0.32})
    sub.subplots_adjust(left=0.012, right=0.99, top=0.84, bottom=0.27)
    r, col = divmod(int(c['cell']), TILE // CELL)
    k_not = int(c['k_not_spray_model'])
    k_no = int(c['k_no_spray_model'])

    # 1. the tile, with the grid and this cell
    ax = axs[0]
    ax.imshow(rgb)
    for t in range(1, TILE // CELL):
        ax.axhline(t * CELL - 0.5, color='white', lw=1)
        ax.axvline(t * CELL - 0.5, color='white', lw=1)
    box = Rectangle((col * CELL - 0.5, r * CELL - 0.5), CELL, CELL, fill=False, edgecolor=colour, linewidth=3.5)
    box.set_path_effects([pe.withStroke(linewidth=6, foreground='white')])
    ax.add_patch(box)
    ax.text(col * CELL + 10, r * CELL + 12, f'ô {int(c["cell"]) + 1}', color='white', fontsize=10,
            fontweight='bold', va='top', path_effects=[pe.withStroke(linewidth=3, foreground='black')])
    ax.set_title(f'Ô ảnh {c["tile"]} (1,5 m × 1,5 m)\nbay ngày {date_vi(c["tile"])}', fontsize=10.5, color=INK2)
    clean_axes(ax)

    # 2. the cell, with the patches the explanation names (solid: up to leaving SPRAY, dashed: on to NO_SPRAY)
    boxes = patch_boxes(c)
    solid = k_not if c['decision_rescored'] == SPRAY else 0
    ax = axs[1]
    ax.imshow(np.asarray(rgb)[r * CELL:(r + 1) * CELL, col * CELL:(col + 1) * CELL])
    draw_patches(ax, boxes, solid, k_no, number=k_no <= NUMBERED)
    ax.set_title(f'Ô {int(c["cell"]) + 1} phóng to (50 cm × 50 cm): {name}\nvùng cần bỏ, mỗi vùng {num(PATCH_CM)} cm × '
                 f'{num(PATCH_CM)} cm', fontsize=10.5, color=INK2)
    clean_axes(ax)

    # 3. the model's P(weed) inside the cell (one-hue sequential scale)
    ax = axs[2]
    im = ax.imshow(p_cell, cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
    draw_patches(ax, boxes, solid, k_no, number=k_no <= NUMBERED)
    ax.set_title('Xác suất cỏ dự đoán\n(mô hình phân vùng)', fontsize=10.5, color=INK2)
    clean_axes(ax)
    cb = sub.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(['0', '0,5', '1'])
    cb.ax.tick_params(labelsize=8.5, colors=INK2, length=2)
    cb.outline.set_visible(False)

    # 4. predicted weed area as patches are removed, against the decision zones
    ax = axs[3]
    k = steps['k'].to_numpy()
    model, est = steps['s_model'].to_numpy() * CELL_CM2, steps['s_estimate'].to_numpy() * CELL_CM2
    no_thr, sp_thr = thresholds_cm2(c)
    top = max(model.max(), est.max(), sp_thr) * 1.12
    crossed = c['lam_hi'] < c['lam_lo']  # then SPRAY iff s >= lam_lo and NO_SPRAY iff s <= lam_hi
    zones = ((sp_thr, top, SPRAY, f'PHUN ({"≥" if crossed else ">"} {num(sp_thr)} cm²)'),
             (no_thr, sp_thr, DEFER, 'CHƯA CHẮC'),
             (0, no_thr, NO_SPRAY, f'KHÔNG PHUN ({"≤" if crossed else "<"} {num(no_thr)} cm²)'))
    mark = k_not if c['decision_rescored'] == SPRAY else k_no
    right = mark >= 0.6 * k.max()  # the marker line is in the right part: labels go left of it
    named = True
    for lo_, hi_, d, text in zones:
        ax.axhspan(lo_, hi_, color=DECISION[d][1], alpha=0.13, lw=0)
        if hi_ - lo_ >= 0.1 * top:  # named inside the zone where it fits, at the right edge or left of the marker
            ax.text(mark - 0.3 if right else k.max() + 0.3, (lo_ + hi_) / 2, text + ' ', fontsize=9, color=INK2,
                    fontweight='bold', va='center', ha='right')
        else:
            named = False
    for y in (no_thr, sp_thr):
        ax.axhline(y, color=INK2, lw=0.8)
    if not named:  # thin zones: the thresholds in one line above them, on the left where the curve is high
        ax.text(0.2, sp_thr, f'ngưỡng phun {num(sp_thr)} cm², ngưỡng không phun {num(no_thr)} cm²', fontsize=8.5,
                color=INK2, va='bottom', ha='left')
    ax.plot(k, est, color=INK2, lw=1.8, ls=(0, (4, 2.5)))
    ax.plot(k, model, color=INK, lw=2, marker='o', ms=4.5)
    ax.axvline(mark, color=colour if c['decision_rescored'] == SPRAY else DECISION[NO_SPRAY][1], lw=1.4)
    label = f'bỏ {mark} vùng → hết PHUN' if c['decision_rescored'] == SPRAY else f'bỏ {mark} vùng → KHÔNG PHUN'
    ax.text(mark, top * 0.97, (label + ' ') if right else (' ' + label), fontsize=9, color=INK, va='top',
            ha='right' if right else 'left')
    ax.set_xlim(-0.3, k.max() + 0.3)
    ax.set_ylim(0, top)
    ax.set_xlabel('số vùng đã bỏ (vùng nhiều cỏ dự đoán nhất trước)', fontsize=9.5, color=INK2)
    ax.set_ylabel('cỏ dự đoán trong ô (cm²)', fontsize=9.5, color=INK2)
    # the two lines differ by style, and the title says which is which (an in-plot legend collides with the curves)
    ax.set_title('Lượng cỏ dự đoán khi bỏ dần từng vùng\nđường liền có chấm: mô hình chấm lại · đường đứt: ước lượng '
                 'nhanh', fontsize=10.5, color=INK2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(labelsize=8.5, colors=INK2)
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c3c2b7')

    # 5. bands behind the score: integrated gradients from bare soil, in cm^2 of predicted weed (they add up to it)
    ax = axs[4]
    cm2 = band_cm2(c)
    vals = np.array([cm2[b] for b in BANDS])
    y = np.arange(len(BANDS))[::-1]
    ax.barh(y, vals, color=BAR, height=0.62)
    ax.axvline(0, color='#c3c2b7', lw=1)
    span = vals.max() - min(vals.min(), 0)
    ax.set_xlim(min(vals.min(), 0) - (0.4 * span if vals.min() < 0 else 0.05 * span), vals.max() + 0.4 * span)
    for yy, v in zip(y, vals):
        ax.text(v + (0.03 if v >= 0 else -0.03) * span, yy, signed(v), va='center', ha='left' if v >= 0 else 'right',
                fontsize=8.5, color=INK)
    ax.set_yticks(y, BANDS, fontsize=9, color=INK2)
    ax.tick_params(axis='y', length=0)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f'Dải phổ đóng góp vào lượng cỏ dự đoán\n(IG so với đất trống, cm²; tổng {num(sum(vals))} cm²)',
                 fontsize=10.5, color=INK2)

    for n, line in enumerate(text_lines(key, c, steps, add_to_spray_cm2)):
        sub.text(0.012, 0.165 - 0.06 * n, line, fontsize=10.5, color=INK if n < 2 else INK2, ha='left', va='top')


def load_inputs(root):
    res = root / 'results'
    cells = pd.read_csv(res / 'weeds_cfcheck/cfcheck_cells.csv', keep_default_na=False, na_values=[''])
    steps = pd.read_csv(res / 'weeds_cfcheck/cfcheck_steps.csv')
    xai = pd.read_csv(res / 'weeds_xai/xai_cells.csv')
    local = pd.read_csv(res / 'weeds_counterfactual/counterfactual_cells.csv', keep_default_na=False, na_values=[''])
    cells = (cells.merge(xai[['tile', 'cell', 'w', 'ig_sum'] + [f'ig_{b}' for b in BANDS]], on=['tile', 'cell'])
             .merge(local[['tile', 'cell', 'patches', 'add_to_spray_cm2']].rename(columns={'patches': 'patches_local'}),
                    on=['tile', 'cell']))
    return cells, steps


def check_example(c, p_cell):
    """The saved probability map gives the Colab run's cell score, and both rankings name the same patches."""
    s = float(p_cell.astype(np.float64).mean())
    if abs(s - c['s_model']) > SCORE_TOL:
        raise ValueError(f'{c["tile"]} cell {c["cell"]}: score {s:.6f} from the saved map, {c["s_model"]:.6f} on Colab')
    a, b = str(c['patches']).split(), str(c['patches_local']).split()
    n = min(len(a), len(b))
    if a[:n] != b[:n]:
        raise ValueError(f'{c["tile"]} cell {c["cell"]}: the Colab and local patch rankings differ')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', default=str(ROOT_DIR / 'data/weedsgalore/weedsgalore-dataset.zip'))
    ap.add_argument('--probs', default=str(ROOT_DIR / 'results/weeds_authors/dlv3p_msi_3/probs_test.npz'))
    ap.add_argument('--out', default=str(ROOT_DIR / 'data/weedsgalore/hinh_giai_thich_phan_thuc.png'))
    args = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if any('arial' in f.name.lower() for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = 'Arial'  # as the other figures; it has the Vietnamese letters

    cells, steps = load_inputs(ROOT_DIR)
    examples = pick_examples(cells)
    with np.load(args.probs) as p:
        ids, p_weed = list(p['ids'].astype(str)), p['p_weed']
    with zipfile.ZipFile(args.zip) as zf:
        lo, hi = band_limits(zf, sorted(t for s in SPLITS for t in split_ids(zf, s)))  # as the preview images
        tiles = {c['tile']: read_tile(zf, c['tile'])[0] for _, c in examples}

    fig = plt.figure(figsize=(20, 5.0 * len(examples) + 1.3), dpi=110, facecolor='white')
    subs = fig.subfigures(len(examples) + 1, 1, height_ratios=[5.0] * len(examples) + [1.3])
    for i, ((key, c), sub) in enumerate(zip(examples, subs), start=1):
        r, col = divmod(int(c['cell']), TILE // CELL)
        p_cell = p_weed[ids.index(c['tile'])][r * CELL:(r + 1) * CELL, col * CELL:(col + 1) * CELL]
        check_example(c, p_cell)
        st = steps[(steps['tile'] == c['tile']) & (steps['cell'] == c['cell'])].sort_values('k')
        draw_row(sub, i, key, c, st, composite(tiles[c['tile']], (R, G, B), lo, hi), p_cell.astype(np.float32),
                 c['add_to_spray_cm2'])
        print(f'example {i} ({key}): {c["tile"]} cell {int(c["cell"]) + 1}, k model {c["k_not_spray_model"]:g} / '
              f'{c["k_no_spray_model"]:g}, estimate {c["k_not_spray_estimate"]:g} / {c["k_no_spray_estimate"]:g}')
    notes = [
        'Cách đọc: "bỏ một vùng" là thay vùng 5 cm × 5 cm đó bằng đất trống rồi cho mô hình chấm lại cả ô ảnh. Các '
        'vùng được bỏ theo thứ tự nhiều cỏ dự đoán nhất trước, đúng thứ tự đánh số; số vùng ghi trên hình là số mô '
        'hình đã kiểm chứng',
        'theo thứ tự đó, không phải số vùng ít nhất có thể. Viền liền: các vùng cần bỏ để ô hết PHUN; viền đứt: các '
        'vùng bỏ thêm để ô thành KHÔNG PHUN. "Ước lượng nhanh" chỉ trừ phần xác suất cỏ của các vùng bị bỏ, không chạy '
        'lại mô hình.',
        'Ví dụ được chọn theo quy tắc cố định (scripts/make_counterfactual_figure.py, hàm pick_examples), không chọn '
        'bằng mắt. Mô hình phân vùng: mô hình công bố của tác giả (DeepLabv3+, 5 dải phổ). Ngưỡng conformal hiệu chỉnh '
        'trên các vị trí test khác.',
        'Ngưỡng cần phun 1% diện tích ô chỉ là ngưỡng minh họa, chưa có nguồn. Dữ liệu: WeedsGalore (Celikkan và cộng '
        'sự, WACV 2025), giấy phép CC BY 4.0.']
    for n, line in enumerate(notes):
        subs[-1].text(0.012, 0.86 - 0.2 * n, line, fontsize=10, color=INK2, ha='left', va='top')
    out = Path(args.out)
    fig.savefig(out, dpi=110, facecolor='white')
    print(f'saved {out} ({out.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
