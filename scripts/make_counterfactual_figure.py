"""Draw the counterfactual explanations of a few spraying decisions, as checked by the model (results/weeds_cfcheck).

For each example cell, one row: the tile around it; the cell with the patches the explanation names; the model's
P(weed) map of the cell; the cell's predicted weed area after removing 0, 1, 2, ... of those patches (the model's
rescoring next to the fast estimate of src.weeds.counterfactual) against the thresholds; the spectral bands behind
the score (integrated gradients from bare soil, results/weeds_xai); and the explanation in words. The examples are
picked by fixed rules (pick_examples), not by eye. Before drawing, the weed pixels of the saved maps are checked
against the Colab run (check_example) and, where they agree, the patch rankings against results/weeds_counterfactual. Text is Vietnamese (the
audience); --lang en draws the detailed figure in English (decimal points), for the English version of the paper.

  python scripts/make_counterfactual_figure.py [--out data/weedsgalore/hinh_giai_thich_phan_thuc.png] [--lang en]
"""
import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT_DIR), str(ROOT_DIR / 'scripts')]
from export_weeds_images import B, G, R, band_limits, composite, date_en, date_vi, fixed_half_up  # noqa: E402
from src.weeds.analyze import PRIMARY  # noqa: E402
from src.weeds.counterfactual import GSD_CM, PATCH  # noqa: E402
from src.weeds.data import BANDS, SPLITS, TILE, read_tile, split_ids  # noqa: E402
from src.weeds.decision import DEFER, NO_SPRAY, SPRAY, predicted_weed  # noqa: E402

CELL = PRIMARY['cell_size']
CELL_CM2 = (CELL * GSD_CM) ** 2  # 2500 cm^2: a 50 cm x 50 cm cell
PATCH_CM = PATCH * GSD_CM
# the decision colours of make_pipeline_figure.py, with the amber darkened to #c98500: as a set they pass the
# dataviz palette checks on white (lightness band, chroma, colour-blind and normal-vision separation, 3:1 contrast)
DECISION = {SPRAY: ('PHUN', '#d7301f'), NO_SPRAY: ('KHÔNG PHUN', '#3182bd'), DEFER: ('CHƯA CHẮC', '#c98500')}
INK, INK2, MUTED, GRID, BAR = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#2a78d6'
# The saved maps (one GPU run of the model, two tiles a batch) and the Colab checks (another run, one tile a batch) can
# see a few weed pixels differently: 16 of the 216 cells of the main model differ, by 1 or 2 pixels (16/11/2026).
PIXEL_TOL = 2
LANG = 'vi'  # 'en' with --lang en (set_language); the demo imports this module and keeps Vietnamese


def T(vi, en):
    """The text in the figure's language."""
    return en if LANG == 'en' else vi


def set_language(lang):
    """Switch the figure's text to `lang` ('vi' or 'en'); main() calls it once, before drawing."""
    global LANG
    LANG = lang
    if lang == 'en':
        DECISION.update({SPRAY: ('SPRAY', DECISION[SPRAY][1]), NO_SPRAY: ('NO SPRAY', DECISION[NO_SPRAY][1]),
                         DEFER: ('DEFER', DECISION[DEFER][1])})
        TITLES.update({'fragile': 'Example {i}. Fragile SPRAY decision: removing one 5 cm patch of weed changes it',
                       'defer': 'Example {i}. DEFER cell: removing one 5 cm patch of weed makes it NO SPRAY',
                       'short': 'Example {i}. Weedy cell on {date}: the fast estimate falls {gap} patches short of the '
                                "model's rescoring"})


def patches_en(k):
    return f'{k} patch' if k == 1 else f'{k} patches'


def num(v, digits=0):
    """v with `digits` decimals, halves rounded up (export_weeds_images.fixed_half_up: 0,145 -> 0,15). The demo draws
    its curves with these helpers and writes its own numbers with the same rounding."""
    s = fixed_half_up(v, digits)
    return s if LANG == 'en' else s.replace('.', ',')


def pct(share, digits=2):
    """A share of a cell's pixels as a percentage with `digits` decimals (0,195 % -> 0,20 %)."""
    return num(100 * share, digits)


def cm2(v):
    """An area in cm^2: one decimal below 10 cm^2 (the thresholds at tau = 0.1 % are 2 to 5 cm^2), none above."""
    return num(v, 1 if abs(v) < 10 else 0)


def k_or_all(k):
    """A model-checked number of patches as an int; every patch of the cell where the decision never came (NaN)."""
    return int(k) if np.isfinite(k) else (CELL // PATCH) ** 2


def tau_words(tau):
    """The weed-cover threshold as a percentage with the digits it needs (0,1%, 0,06%)."""
    s = f'{100 * tau:g}%'
    return s if LANG == 'en' else s.replace('.', ',')


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
    if c['decision_rescored'] == SPRAY:
        where = T(f'từ ngưỡng phun {cm2(sp_thr)} cm² trở lên', f'at or above the spray threshold of {cm2(sp_thr)} cm²')
    else:
        where = T(f'nằm giữa ngưỡng không phun {cm2(no_thr)} cm² và ngưỡng phun {cm2(sp_thr)} cm²',
                  f'between the no-spray threshold of {cm2(no_thr)} cm² and the spray threshold of {cm2(sp_thr)} cm²')
    share = pct(c['s_model'])  # 2 digits: tau is 0.1 %, the thresholds 0.085-0.175 %
    why = T(f'Vì sao {name}: mô hình thấy khoảng {cm2(weed)} cm² cỏ trong ô 50 cm × 50 cm ({share}% diện tích), {where}.',
            f'Why {name}: the model sees about {cm2(weed)} cm² of weed in the 50 cm × 50 cm cell ({share}% of its area), '
            f'{where}.')
    never = not np.isfinite(c['k_no_spray_model'])  # every patch removed, and still not NO_SPRAY
    k_no = k_or_all(c['k_no_spray_model'])
    head = T('Đổi gì thì quyết định đổi (mô hình đã chấm lại): ', 'What changes the decision (rescored by the model): ')
    if c['decision_rescored'] == SPRAY:
        k = int(c['k_not_spray_model'])
        after = int(steps.set_index('k').loc[k, 'decision_model'])
        if after == NO_SPRAY:
            then = T('chuyển thẳng sang KHÔNG PHUN', 'goes straight to NO SPRAY')
        else:
            then = T(f'chuyển sang {DECISION[after][0]}; ', f'goes to {DECISION[after][0]}; ') + (
                T('bỏ hết các vùng của ô vẫn chưa thành KHÔNG PHUN',
                  'removing every patch of the cell still does not make it NO SPRAY') if never else
                T(f'bỏ {k_no} vùng thì KHÔNG PHUN', f'removing {patches_en(k_no)} makes it NO SPRAY'))
        removed, est = cm2(removed_cm2(steps, k)), int(c['k_not_spray_estimate'])
        what = head + T(f'bỏ {patch_words(k)} ({removed} cm² cỏ dự đoán) thì ô {then}. Ước lượng nhanh: {est} vùng.',
                        f'after removing {patch_words(k)} ({removed} cm² of predicted weed), the cell {then}. '
                        f'Fast estimate: {patches_en(est)}.')
    elif never:
        add = num(add_to_spray_cm2, 1)
        what = head + T('bỏ hết các vùng của ô vẫn chưa thành KHÔNG PHUN; '
                        f'thêm khoảng {add} cm² cỏ thì PHUN (theo quy tắc ngưỡng).',
                        'removing every patch of the cell still does not make it NO SPRAY; '
                        f'about {add} cm² more weed would make it SPRAY (threshold rule).')
    else:
        removed, add = cm2(removed_cm2(steps, k_no)), num(add_to_spray_cm2, 1)
        what = head + T(f'bỏ {patch_words(k_no)} ({removed} cm² cỏ dự đoán) thì ô thành KHÔNG PHUN; thêm khoảng '
                        f'{add} cm² cỏ thì PHUN (theo quy tắc ngưỡng). Người kiểm tra nên xem '
                        f'{"vùng" if k_no == 1 else f"{k_no} vùng"} này trước.',
                        f'removing {patch_words(k_no)} ({removed} cm² of predicted weed) makes the cell NO SPRAY; '
                        f'about {add} cm² more weed would make it SPRAY (threshold rule). A scout should check '
                        f'{"this patch" if k_no == 1 else "these patches"} first.')
    per_band = band_cm2(c)
    up = sorted((b for b in BANDS if per_band[b] > 0), key=per_band.get, reverse=True)[:2]
    down = min(BANDS, key=per_band.get)
    bands = T(' và ', ' and ').join(f'{b} ({signed(per_band[b])} cm²)' for b in up) + T(
        ' đẩy xác suất cỏ của ô lên nhiều nhất', " raise the cell's weed probability most" if len(up) > 1 else
        " raises the cell's weed probability most")
    if per_band[down] < 0:
        bands += T(f'; {down} kéo xuống nhiều nhất ({signed(per_band[down])} cm²)',
                   f'; {down} lowers it most ({signed(per_band[down])} cm²)')
    need = c['w'] >= PRIMARY['tau']
    tau = tau_words(PRIMARY['tau'])
    truth = (T('ô thực sự cần phun', 'the cell does need spraying') if need else
             T('ô thực ra không cần phun', 'the cell does not need spraying')) + T(f' (ngưỡng {tau})', f' (threshold {tau})')
    cover = pct(c['w'])
    tail = T(f'Dải phổ (IG): {bands}. Nhãn thật: cỏ chiếm {cover}% diện tích ô → {truth}.',
             f'Bands (IG): {bands}. True label: weed covers {cover}% of the cell → {truth}.')
    return [why, what, tail]


NUMBERED = 9  # patches are numbered on the images up to this many


def patch_words(k):
    if k == 1:
        return T('vùng số 1', 'patch 1')
    return T(f'{k} vùng số 1–{k}', f'patches 1–{k}') if k <= NUMBERED else T(f'{k} vùng có viền',
                                                                              f'the {k} outlined patches')


def band_cm2(c):
    """Integrated gradients per band, as cm^2: the cell's mean P(weed) times the cell area. They add up to
    p(tile) - p(bare soil), and the bare-soil P(weed) is about 0 (results/weeds_xai), so the bands share out the cell's
    weed probability, the smooth stand-in for its predicted weed area (IG needs a gradient, the pixel count has none)."""
    return {b: c[f'ig_{b}'] * CELL_CM2 for b in BANDS}


def signed(v):
    return ('+' if v >= 0 else '−') + num(abs(v))


TITLES = {'fragile': 'Ví dụ {i}. Quyết định PHUN mong manh: bỏ một vùng cỏ 5 cm là đổi quyết định',
          'defer': 'Ví dụ {i}. Ô CHƯA CHẮC: bỏ một vùng cỏ 5 cm là thành KHÔNG PHUN',
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


def draw_curve(ax, c, steps, estimate=True, fs=9, fs_axis=9.5, fs_tick=8.5, xlabel=None, ylabel=None):
    """The cell's predicted weed area (cm^2) after removing 0, 1, 2, ... of the named patches, as the model rescored
    it (and the fast estimate, if `estimate`), over the decision zones, with a line where the decision changes."""
    from matplotlib.ticker import MaxNLocator
    if xlabel is None:
        xlabel = T('số vùng đã bỏ (vùng nhiều cỏ dự đoán nhất trước)', 'patches removed (most predicted weed first)')
    if ylabel is None:
        ylabel = T('cỏ dự đoán trong ô (cm²)', 'predicted weed in the cell (cm²)')
    colour = DECISION[int(c['decision_rescored'])][1]
    k_not, k_no = k_or_all(c['k_not_spray_model']), k_or_all(c['k_no_spray_model'])
    k = steps['k'].to_numpy()
    model, est = steps['s_model'].to_numpy() * CELL_CM2, steps['s_estimate'].to_numpy() * CELL_CM2
    no_thr, sp_thr = thresholds_cm2(c)
    top = max(model.max(), est.max(), sp_thr) * 1.12
    crossed = c['lam_hi'] < c['lam_lo']  # then SPRAY iff s >= lam_lo and NO_SPRAY iff s <= lam_hi
    zones = ((sp_thr, top, SPRAY, f'{DECISION[SPRAY][0]} ({"≥" if crossed else ">"} {cm2(sp_thr)} cm²)'),
             (no_thr, sp_thr, DEFER, DECISION[DEFER][0]),
             (0, no_thr, NO_SPRAY, f'{DECISION[NO_SPRAY][0]} ({"≤" if crossed else "<"} {cm2(no_thr)} cm²)'))
    mark = k_not if c['decision_rescored'] == SPRAY else k_no
    right = mark >= 0.6 * k.max()  # the marker line is in the right part: labels go left of it
    named = True
    for lo_, hi_, d, text in zones:
        ax.axhspan(lo_, hi_, color=DECISION[d][1], alpha=0.13, lw=0)
        if hi_ - lo_ >= 0.1 * top:  # named inside the zone where it fits, at the right edge or left of the marker
            ax.text(mark - 0.3 if right else k.max() + 0.3, (lo_ + hi_) / 2, text + ' ', fontsize=fs, color=INK2,
                    fontweight='bold', va='center', ha='right')
        else:
            named = False
    for y in (no_thr, sp_thr):
        ax.axhline(y, color=INK2, lw=0.8)
    if not named:  # thin zones: the thresholds in one line above them, on the left where the curve is high
        ax.text(0.2, sp_thr, T(f'ngưỡng phun {cm2(sp_thr)} cm², ngưỡng không phun {cm2(no_thr)} cm²',
                               f'spray threshold {cm2(sp_thr)} cm², no-spray threshold {cm2(no_thr)} cm²'),
                fontsize=fs - 0.5, color=INK2, va='bottom', ha='left')
    if estimate:
        ax.plot(k, est, color=INK2, lw=1.8, ls=(0, (4, 2.5)))
    ax.plot(k, model, color=INK, lw=2, marker='o', ms=4.5)
    ax.axvline(mark, color=colour if c['decision_rescored'] == SPRAY else DECISION[NO_SPRAY][1], lw=1.4)
    label = (T(f'bỏ {mark} vùng → hết PHUN', f'remove {patches_en(mark)} → no longer SPRAY')
             if c['decision_rescored'] == SPRAY else T(f'bỏ {mark} vùng → KHÔNG PHUN', f'remove {patches_en(mark)} → NO SPRAY'))
    ax.text(mark, top * 0.97, (label + ' ') if right else (' ' + label), fontsize=fs, color=INK, va='top',
            ha='right' if right else 'left')
    ax.set_xlim(-0.3, k.max() + 0.3)
    ax.set_ylim(0, top)
    ax.set_xlabel(xlabel, fontsize=fs_axis, color=INK2)
    ax.set_ylabel(ylabel, fontsize=fs_axis, color=INK2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(labelsize=fs_tick, colors=INK2)
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c3c2b7')


def draw_row(sub, i, key, c, steps, rgb, p_cell, add_to_spray_cm2):
    import matplotlib.patheffects as pe
    from matplotlib.patches import Rectangle
    name, colour = DECISION[int(c['decision_rescored'])]
    gap = int(c['k_not_spray_model'] - c['k_not_spray_estimate'])
    sub.suptitle(TITLES[key].format(i=i, date=T(date_vi(c['tile'])[:5], date_en(c['tile'], year=False)), gap=gap),
                 x=0.012, ha='left', fontsize=15, fontweight='bold', color=INK)
    axs = sub.subplots(1, 5, gridspec_kw={'width_ratios': [1, 1, 1.12, 1.75, 1.05], 'wspace': 0.32})
    sub.subplots_adjust(left=0.012, right=0.99, top=0.84, bottom=0.27)
    r, col = divmod(int(c['cell']), TILE // CELL)
    k_not = k_or_all(c['k_not_spray_model'])
    k_no = k_or_all(c['k_no_spray_model'])

    # 1. the tile, with the grid and this cell
    ax = axs[0]
    ax.imshow(rgb)
    for t in range(1, TILE // CELL):
        ax.axhline(t * CELL - 0.5, color='white', lw=1)
        ax.axvline(t * CELL - 0.5, color='white', lw=1)
    box = Rectangle((col * CELL - 0.5, r * CELL - 0.5), CELL, CELL, fill=False, edgecolor=colour, linewidth=3.5)
    box.set_path_effects([pe.withStroke(linewidth=6, foreground='white')])
    ax.add_patch(box)
    n = int(c['cell']) + 1
    ax.text(col * CELL + 10, r * CELL + 12, T(f'ô {n}', f'cell {n}'), color='white', fontsize=10,
            fontweight='bold', va='top', path_effects=[pe.withStroke(linewidth=3, foreground='black')])
    ax.set_title(T(f'Ô ảnh {c["tile"]} (1,5 m × 1,5 m)\nbay ngày {date_vi(c["tile"])}',
                   f'Tile {c["tile"]} (1.5 m × 1.5 m)\nflown on {date_en(c["tile"])}'), fontsize=10.5, color=INK2)
    clean_axes(ax)

    # 2. the cell, with the patches the explanation names (solid: up to leaving SPRAY, dashed: on to NO_SPRAY)
    boxes = patch_boxes(c)
    solid = k_not if c['decision_rescored'] == SPRAY else 0
    ax = axs[1]
    ax.imshow(np.asarray(rgb)[r * CELL:(r + 1) * CELL, col * CELL:(col + 1) * CELL])
    draw_patches(ax, boxes, solid, k_no, number=k_no <= NUMBERED)
    side = num(PATCH_CM)
    ax.set_title(T(f'Ô {n} phóng to (50 cm × 50 cm): {name}\nvùng cần bỏ, mỗi vùng {side} cm × {side} cm',
                   f'Cell {n} enlarged (50 cm × 50 cm): {name}\npatches to remove, each {side} cm × {side} cm'),
                 fontsize=10.5, color=INK2)
    clean_axes(ax)

    # 3. the model's P(weed) inside the cell (one-hue sequential scale)
    ax = axs[2]
    im = ax.imshow(p_cell, cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
    draw_patches(ax, boxes, solid, k_no, number=k_no <= NUMBERED)
    ax.set_title(T('Xác suất cỏ dự đoán\n(mô hình phân vùng)', 'Predicted weed probability\n(segmentation model)'),
                 fontsize=10.5, color=INK2)
    clean_axes(ax)
    cb = sub.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(['0', T('0,5', '0.5'), '1'])
    cb.ax.tick_params(labelsize=8.5, colors=INK2, length=2)
    cb.outline.set_visible(False)

    # 4. predicted weed area as patches are removed, against the decision zones
    ax = axs[3]
    draw_curve(ax, c, steps, estimate=True)
    # the two lines differ by style, and the title says which is which (an in-plot legend collides with the curves)
    ax.set_title(T('Lượng cỏ dự đoán khi bỏ dần từng vùng\nđường liền có chấm: mô hình chấm lại · đường đứt: ước lượng '
                   'nhanh', 'Predicted weed as patches are removed\nsolid with dots: model rescoring · dashed: fast '
                   'estimate'), fontsize=10.5, color=INK2)

    # 5. bands behind the score: integrated gradients from bare soil, in cm^2 of predicted weed (they add up to it)
    ax = axs[4]
    per_band = band_cm2(c)
    vals = np.array([per_band[b] for b in BANDS])
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
    total = num(sum(vals))
    ax.set_title(T(f'Dải phổ đóng góp vào xác suất cỏ của ô\n(IG so với đất trống, cm²; tổng {total} cm²)',
                   f"Band contributions to the cell's weed probability\n(IG from bare soil, cm²; total {total} cm²)"),
                 fontsize=10.5, color=INK2)

    for n, line in enumerate(text_lines(key, c, steps, add_to_spray_cm2)):
        sub.text(0.012, 0.165 - 0.06 * n, line, fontsize=10.5, color=INK if n < 2 else INK2, ha='left', va='top')


SIMPLE_TITLES = {'fragile': 'Ví dụ {i}. Ô PHUN sát ngưỡng', 'defer': 'Ví dụ {i}. Ô CHƯA CHẮC',
                 'short': 'Ví dụ {i}. Ô nhiều cỏ'}


def simple_lines(c, steps):
    """Three short lines for the simple figure: the decision and why, what changes it (as the model rescored it),
    and the true label."""
    dec = int(c['decision_rescored'])
    no_thr, sp_thr = thresholds_cm2(c)
    weed = cm2(c['s_model'] * CELL_CM2)
    never = not np.isfinite(c['k_no_spray_model'])  # every patch removed, and still not NO_SPRAY
    k_no = k_or_all(c['k_no_spray_model'])
    if dec == SPRAY:
        why = f'Máy thấy khoảng {weed} cm² cỏ trong ô, từ ngưỡng phun {cm2(sp_thr)} cm² trở lên → PHUN.'
        k = int(c['k_not_spray_model'])
        after = int(steps.set_index('k').loc[k, 'decision_model'])
        more = '' if after == NO_SPRAY else ('; bỏ hết các vùng vẫn chưa thành KHÔNG PHUN' if never else
                                             f'; bỏ thêm vùng số {k_no} → KHÔNG PHUN' if k_no == k + 1 else
                                             f'; bỏ {patch_words(k_no)} → KHÔNG PHUN')
        if k >= 10:
            share = 100 * k * PATCH ** 2 / CELL ** 2
            change = (f'Phải bỏ tới {patch_words(k)} ({num(share)}% diện tích ô) mới hết PHUN '
                      f'(thành {DECISION[after][0]}{more}), nên quyết định phun khó đổi.')
        else:
            change = f'Bỏ {patch_words(k)} → {DECISION[after][0]}{more}.'
            if k == 1:
                change += ' Chỉ thiếu một đám cỏ nhỏ là quyết định đã đổi.'
    else:
        why = (f'Máy thấy khoảng {weed} cm² cỏ, nằm giữa ngưỡng không phun {cm2(no_thr)} cm² và ngưỡng phun '
               f'{cm2(sp_thr)} cm² → CHƯA CHẮC.')
        change = ('Bỏ hết các vùng của ô vẫn chưa thành KHÔNG PHUN.' if never else
                  f'Bỏ {patch_words(k_no)} → KHÔNG PHUN. Người kiểm tra nên xem '
                  f'{"vùng này" if k_no == 1 else "các vùng này"} trước.')
    need = c['w'] >= PRIMARY['tau']
    truth = (f'Thực tế (theo nhãn): cỏ chiếm {pct(c["w"])}% diện tích ô → '
             + ('ô thật sự cần phun.' if need else 'ô thực ra không cần phun.'))
    return [why, change, truth]


def put_lines(sub, lines, y, fontsize, step, width, colors):
    """Lines of text from y down, each wrapped at `width` characters; returns the y below the last line."""
    import textwrap
    for line, colour in zip(lines, colors):
        for part in textwrap.wrap(line, width):
            sub.text(0.03, y, part, fontsize=fontsize, color=colour, ha='left', va='top')
            y -= step
    return y


def draw_simple_row(sub, i, key, c, steps, rgb_cell, p_cell):
    """Simple layout for print: the cell with the model's weed tinted red and the named patches numbered, and the
    predicted weed area as they are removed, as the model rescored it; then three short lines."""
    dec = int(c['decision_rescored'])
    sub.suptitle(SIMPLE_TITLES[key].format(i=i) + f' (bay ngày {date_vi(c["tile"])})', x=0.03, ha='left',
                 fontsize=13, fontweight='bold', color=INK)
    axs = sub.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.5], 'wspace': 0.3})
    sub.subplots_adjust(left=0.03, right=0.97, top=0.84, bottom=0.36)
    k_no = k_or_all(c['k_no_spray_model'])

    ax = axs[0]
    alpha = 0.65 * np.clip(p_cell, 0, 1)[..., None]
    tinted = np.asarray(rgb_cell, dtype=np.float32) / 255 * (1 - alpha) + np.array([0.84, 0.19, 0.12]) * alpha
    ax.imshow(np.clip(tinted, 0, 1))  # float rounding can leave 1 + 1e-8
    draw_patches(ax, patch_boxes(c), k_no, k_no, number=k_no <= NUMBERED)
    ax.set_title(f'Ô 50 cm × 50 cm: {DECISION[dec][0]}', fontsize=10.5, color=INK2)
    clean_axes(ax)
    ax.set_xlabel('màu đỏ: chỗ máy cho là cỏ\n(càng đỏ, máy càng chắc)', fontsize=9.5, color=INK2)

    ax = axs[1]
    draw_curve(ax, c, steps, estimate=False, fs=9.5, fs_axis=10, fs_tick=9, xlabel='số vùng 5 cm đã bỏ',
               ylabel='lượng cỏ máy thấy (cm²)')
    ax.set_title('Bỏ dần từng vùng thì quyết định đổi ở đâu?', fontsize=10.5, color=INK2)

    put_lines(sub, simple_lines(c, steps), 0.235, 10.5, 0.06, 118, [INK, INK, INK2])


SIMPLE_NOTES = [
    'Cách đọc: "bỏ một vùng" là thay vùng 5 cm × 5 cm đó bằng đất trống trên ảnh rồi cho máy chấm lại. Các vùng được '
    'bỏ lần lượt, vùng nhiều cỏ nhất trước (số đánh trên ảnh là thứ tự này). Số vùng trên hình là số máy đã kiểm chứng '
    'theo thứ tự đó, không phải số ít nhất có thể.',
    'Ba ô được chọn theo quy tắc cố định, không chọn bằng mắt. Mô hình: mô hình công bố của tác giả (DeepLabv3+, '
    f'5 dải phổ). Ngưỡng "cần phun": cỏ chiếm từ {tau_words(PRIMARY["tau"])} diện tích ô (Longchamps và cộng sự, '
    '2014). Dữ liệu: WeedsGalore (Celikkan và cộng sự, WACV 2025), giấy phép CC BY 4.0.']


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


def check_example(c, w_cell):
    """The saved maps (w_cell: the cell's weed pixels from them) see the weed pixels of the Colab run up to PIXEL_TOL
    pixels; where they see as many, both rankings name the same patches (with a pixel more or less, two patches can
    swap). Returns the saved map's weed pixels minus Colab's."""
    n_saved, n_colab = int(w_cell.sum()), int(round(c['s_model'] * w_cell.size))
    if abs(n_saved - n_colab) > PIXEL_TOL:
        raise ValueError(f'{c["tile"]} cell {c["cell"]}: {n_saved} weed pixels in the saved map, {n_colab} on Colab')
    a, b = str(c['patches']).split(), str(c['patches_local']).split()
    n = min(len(a), len(b))
    if n_saved == n_colab and a[:n] != b[:n]:
        raise ValueError(f'{c["tile"]} cell {c["cell"]}: the Colab and local patch rankings differ')
    return n_saved - n_colab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', default=str(ROOT_DIR / 'data/weedsgalore/weedsgalore-dataset.zip'))
    ap.add_argument('--probs', default=str(ROOT_DIR / 'results/weeds_authors/dlv3p_msi_3/probs_test.npz'))
    ap.add_argument('--out', default=None, help='default: data/weedsgalore/hinh_giai_thich_phan_thuc.png, or '
                                                 'hinh_giai_thich_don_gian.png with --simple')
    ap.add_argument('--simple', action='store_true', help='simple layout for print: cell and curve only')
    ap.add_argument('--lang', choices=('vi', 'en'), default='vi', help='language of the text (en: detailed figure only)')
    args = ap.parse_args()
    if args.simple and args.lang == 'en':
        ap.error('--lang en draws the detailed figure only, not --simple')
    set_language(args.lang)
    if args.out is None:
        args.out = str(ROOT_DIR / 'data/weedsgalore' / ('hinh_giai_thich_don_gian.png' if args.simple
                                                        else 'hinh_giai_thich_phan_thuc.png'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if any('arial' in f.name.lower() for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = 'Arial'  # as the other figures; it has the Vietnamese letters

    cells, steps = load_inputs(ROOT_DIR)
    examples = pick_examples(cells)
    with np.load(args.probs) as p:
        ids, p_crop, p_weed = list(p['ids'].astype(str)), p['p_crop'], p['p_weed']
    with zipfile.ZipFile(args.zip) as zf:
        lo, hi = band_limits(zf, sorted(t for s in SPLITS for t in split_ids(zf, s)))  # as the preview images
        tiles = {c['tile']: read_tile(zf, c['tile'])[0] for _, c in examples}

    if args.simple:  # sized for an A4 page: 9 in wide, text at least 9 pt
        fig = plt.figure(figsize=(9, 4.6 * len(examples) + 1.3), dpi=200, facecolor='white')
        subs = fig.subfigures(len(examples) + 1, 1, height_ratios=[4.6] * len(examples) + [1.3])
    else:
        fig = plt.figure(figsize=(20, 5.0 * len(examples) + 1.3), dpi=110, facecolor='white')
        subs = fig.subfigures(len(examples) + 1, 1, height_ratios=[5.0] * len(examples) + [1.3])
    for i, ((key, c), sub) in enumerate(zip(examples, subs), start=1):
        r, col = divmod(int(c['cell']), TILE // CELL)
        row, sl = ids.index(c['tile']), (slice(r * CELL, (r + 1) * CELL), slice(col * CELL, (col + 1) * CELL))
        p_cell = p_weed[row][sl]  # (row: the tile's row in the maps; i stays the example's number in the titles)
        check_example(c, predicted_weed(p_crop[row][sl], p_cell))
        st = steps[(steps['tile'] == c['tile']) & (steps['cell'] == c['cell'])].sort_values('k')
        rgb = composite(tiles[c['tile']], (R, G, B), lo, hi)
        if args.simple:
            draw_simple_row(sub, i, key, c, st, np.asarray(rgb)[r * CELL:(r + 1) * CELL, col * CELL:(col + 1) * CELL],
                            p_cell.astype(np.float32))
        else:
            draw_row(sub, i, key, c, st, rgb, p_cell.astype(np.float32), c['add_to_spray_cm2'])
        print(f'example {i} ({key}): {c["tile"]} cell {int(c["cell"]) + 1}, k model {c["k_not_spray_model"]:g} / '
              f'{c["k_no_spray_model"]:g}, estimate {c["k_not_spray_estimate"]:g} / {c["k_no_spray_estimate"]:g}')
    notes = T([
        'Cách đọc: "bỏ một vùng" là thay vùng 5 cm × 5 cm đó bằng đất trống rồi cho mô hình chấm lại cả ô ảnh. Các '
        'vùng được bỏ theo thứ tự nhiều cỏ dự đoán nhất trước, đúng thứ tự đánh số; số vùng ghi trên hình là số mô '
        'hình đã kiểm chứng',
        'theo thứ tự đó, không phải số vùng ít nhất có thể. Viền liền: các vùng cần bỏ để ô hết PHUN; viền đứt: các '
        'vùng bỏ thêm để ô thành KHÔNG PHUN. "Ước lượng nhanh" chỉ trừ số điểm ảnh cỏ của các vùng bị bỏ, không chạy '
        'lại mô hình.',
        'Ví dụ được chọn theo quy tắc cố định (scripts/make_counterfactual_figure.py, hàm pick_examples), không chọn '
        'bằng mắt. Mô hình phân vùng: mô hình công bố của tác giả (DeepLabv3+, 5 dải phổ). Ngưỡng conformal hiệu chỉnh '
        'trên các vị trí test khác.',
        f'Ngưỡng cần phun: cỏ chiếm từ {tau_words(PRIMARY["tau"])} diện tích ô (trong khoảng 0,06–0,31% của Longchamps và '
        'cộng sự, 2014). Dữ liệu: WeedsGalore (Celikkan và cộng sự, WACV 2025), giấy phép CC BY 4.0.'], [
        'How to read: "removing a patch" replaces that 5 cm × 5 cm patch with bare soil and lets the model rescore the '
        'whole tile. Patches are removed in order of predicted weed, most first, as numbered; the number of patches '
        'shown is the one the model confirmed',
        'in that order, not the fewest possible. Solid outline: patches to remove for the cell to leave SPRAY; dashed: '
        'further patches for NO SPRAY. The "fast estimate" only subtracts the weed pixels of the removed patches, '
        'without rerunning the model.',
        'Examples picked by fixed rules (scripts/make_counterfactual_figure.py, function pick_examples), not by eye. '
        "Segmentation model: the dataset authors' published model (DeepLabv3+, 5 bands). Conformal thresholds "
        'calibrated on the other test locations.',
        f'A cell needs spraying if weed covers at least {tau_words(PRIMARY["tau"])} of it (within the 0.06–0.31% range '
        'of Longchamps et al., 2014). Data: WeedsGalore (Celikkan et al., WACV 2025), CC BY 4.0.'])
    if args.simple:
        put_lines(subs[-1], SIMPLE_NOTES, 0.94, 9, 0.13, 130, [INK2] * len(SIMPLE_NOTES))  # 0.17 in a line, as before
    else:
        for n, line in enumerate(notes):
            subs[-1].text(0.012, 0.86 - 0.2 * n, line, fontsize=10, color=INK2, ha='left', va='top')
    out = Path(args.out)
    fig.savefig(out, dpi=fig.dpi, facecolor='white')
    print(f'saved {out} ({out.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
