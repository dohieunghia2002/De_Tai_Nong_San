"""Matplotlib figures of the progress demo. Text is Vietnamese (the audience); decision colours and the patch and
curve drawing come from scripts/make_counterfactual_figure.py, so the demo looks like the report's figures."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import re  # noqa: E402  (after matplotlib.use, like every import here)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import core as C  # noqa: E402

if any('arial' in f.name.lower() for f in font_manager.fontManager.ttflist):
    plt.rcParams['font.family'] = 'Arial'  # has the Vietnamese letters, as the report's figures

CF = C.CF
INK, INK2 = CF.INK, CF.INK2
SEG3 = np.array([(45, 45, 45), (60, 200, 60), (235, 50, 50)], dtype=np.uint8)  # as scripts/make_pipeline_figure.py
SEG3_VI = ('nền đất', 'ngô', 'cỏ dại')
# error map: reference categorical slots 1 and 2 (blue, orange) for the two weed errors, muted grey for the rest
ERRORS = (('đúng', (242, 242, 239)), ('cỏ bị bỏ sót', (42, 120, 214)), ('nhận nhầm thành cỏ', (235, 104, 52)),
          ('nhầm nền đất / ngô', (137, 135, 129)))
HALO = [pe.withStroke(linewidth=3, foreground='black')]


def hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32) / 255


def clean(ax):
    CF.clean_axes(ax)


def legend_patches(fig, items, y=0.0, ncol=None, fontsize=10, wrong=None):
    """Colour boxes for (text, colour) items; `wrong` adds the X mark of a wrong decision with that text."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=np.asarray(c) / 255 if not isinstance(c, str) else c, edgecolor='#c3c2b7', label=t)
               for t, c in items]
    if wrong:
        handles.append(Line2D([], [], marker='X', color='white', markeredgecolor='black', markeredgewidth=1.2,
                              markersize=11, linestyle='None', label=wrong))
    fig.legend(handles=handles, loc='lower center', ncol=ncol or len(handles), frameon=False, fontsize=fontsize,
               bbox_to_anchor=(0.5, y))


def grid_lines(ax, cell, colour='white', lw=1):
    for t in range(1, C.TILE // cell):
        ax.axhline(t * cell - 0.5, color=colour, lw=lw)
        ax.axvline(t * cell - 0.5, color=colour, lw=lw)


# ---------------------------------------------------------------- dataset

def tile_views(x, mask, lo, hi):
    """True colour, NIR false colour, red edge alone and the 6-class label of one tile."""
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.6), dpi=90)
    red_edge = C.BANDS.index('RE')
    views = ((C.composite(x, (C.R, C.G, C.B), lo, hi), 'Ảnh màu thật (R, G, B)'),
             (C.composite(x, (C.NIR, C.R, C.G), lo, hi), 'Ảnh màu giả (NIR, R, G)\nthực vật hiện màu đỏ'),
             (C.composite(x, (red_edge,) * 3, lo, hi), 'Dải rìa đỏ (RE)'),
             (np.array([c for _, c in C.CLASSES], dtype=np.uint8)[mask], 'Nhãn của tác giả (6 lớp)'))
    for ax, (img, title) in zip(axs, views):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, color=INK2)
        clean(ax)
    legend_patches(fig, [(t, c) for t, c in C.CLASSES], y=-0.02, ncol=6)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.1, wspace=0.05)
    return fig


# ---------------------------------------------------------------- segmentation

def error_map(pred, true):
    out = np.zeros(true.shape + (3,), dtype=np.uint8)
    out[:] = ERRORS[0][1]
    out[(true == 2) & (pred != 2)] = ERRORS[1][1]
    out[(true != 2) & (pred == 2)] = ERRORS[2][1]
    out[(pred != true) & (true != 2) & (pred != 2)] = ERRORS[3][1]
    return out


def segmentation_views(rgb, true3, pred3, p_weed):
    """Image, prediction and P(weed); with the label (true3 not None) also the label and the errors."""
    if true3 is None:
        panels = ((rgb, 'Ảnh màu thật'), (SEG3[pred3], 'Mô hình dự đoán (3 lớp)'), (None, 'Xác suất cỏ của mô hình'))
    else:
        panels = ((rgb, 'Ảnh màu thật'), (SEG3[true3], 'Nhãn thật (3 lớp)'), (SEG3[pred3], 'Mô hình dự đoán'),
                  (None, 'Xác suất cỏ của mô hình'), (error_map(pred3, true3), 'Chỗ mô hình sai'))
    fig, axs = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.9), dpi=85)
    for ax, (img, title) in zip(axs, panels):
        if img is None:
            im = ax.imshow(p_weed.astype(np.float32), cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, ticks=[0, 0.5, 1])
            cb.ax.set_yticklabels(['0', '0,5', '1'])
            cb.outline.set_visible(False)
        else:
            ax.imshow(img, interpolation='nearest')
        ax.set_title(title, fontsize=11, color=INK2)
        clean(ax)
    items = list(zip(SEG3_VI, SEG3)) + ([] if true3 is None else list(ERRORS))
    legend_patches(fig, items, y=-0.02, ncol=len(items))
    fig.subplots_adjust(left=0.01, right=0.98, top=0.88, bottom=0.1, wspace=0.08)
    return fig


# ---------------------------------------------------------------- decisions

def _tinted(rgb, decisions, cell, alpha):
    """RGB tile with each cell tinted by its decision colour."""
    img = np.asarray(rgb, dtype=np.float32) / 255
    per_row = C.TILE // cell
    c = cell * img.shape[0] / C.TILE  # cell side in pixels of this (possibly shrunk) image

    def edge(i):
        return int(round(i * c))

    for k, d in enumerate(decisions):
        r, col = divmod(k, per_row)
        sl = (slice(edge(r), edge(r + 1)), slice(edge(col), edge(col + 1)))
        img[sl] = img[sl] * (1 - alpha) + hex_rgb(C.DECISION_COLOR[int(d)]) * alpha
    return np.clip(img, 0, 1)


def _cell_marks(ax, decisions, need, cell, scale, lw=2.2):
    """Cell borders in the decision colour, and an X on cells whose decision is wrong for the label (need: None or
    missing values when the label is unknown)."""
    per_row = C.TILE // cell
    c = cell / scale
    if need is None:
        need = [None] * len(decisions)
    for k, (d, n) in enumerate(zip(decisions, need)):
        r, col = divmod(k, per_row)
        x0, y0 = col * c - 0.5, r * c - 0.5
        ax.add_patch(Rectangle((x0 + lw / 2, y0 + lw / 2), c - lw, c - lw, fill=False,
                               edgecolor=C.DECISION_COLOR[int(d)], linewidth=lw))
        if n is None or pd.isna(n):
            continue
        if (d == C.NO_SPRAY and n) or (d == C.SPRAY and not n):
            m = 0.28 * c
            for xs, ys in (((x0 + m, x0 + c - m), (y0 + m, y0 + c - m)), ((x0 + m, x0 + c - m), (y0 + c - m, y0 + m))):
                ax.plot(xs, ys, color='white', lw=lw * 1.6, solid_capstyle='round',
                        path_effects=[pe.withStroke(linewidth=lw * 3, foreground='black')])


DECISION_LEGEND = [(C.DECISION_VI[d], C.DECISION_COLOR[d]) for d in (C.SPRAY, C.DEFER, C.NO_SPRAY)]


def field_overview(panels, dates, n_locations, cell, titles):
    """Rows: flight dates; columns: field locations. panels[(date, location)] = (rgb thumbnail, decisions, need);
    titles[location] = column title."""
    fig, axs = plt.subplots(len(dates), n_locations, figsize=(2.05 * n_locations, 2.25 * len(dates) + 0.7), dpi=90,
                            squeeze=False)
    for i, date in enumerate(dates):
        for j in range(n_locations):
            ax = axs[i][j]
            clean(ax)
            p = panels.get((date, j + 1))
            if p is None:
                ax.text(0.5, 0.5, 'không có ảnh', ha='center', va='center', color=INK2, fontsize=9,
                        transform=ax.transAxes)
                continue
            rgb, dec, need = p
            scale = C.TILE / rgb.shape[0]
            ax.imshow(_tinted(rgb, dec, cell, 0.32), interpolation='bilinear')
            _cell_marks(ax, dec, need, cell, scale, lw=1.6)
            if i == 0:
                ax.set_title(titles[j + 1], fontsize=10, color=INK2)
            if j == 0:
                ax.set_ylabel(C.date_vi(date + '_')[:5], fontsize=11, color=INK2, rotation=0, ha='right',
                              va='center', labelpad=8)
    legend_patches(fig, DECISION_LEGEND, y=0.0, fontsize=10, wrong='quyết định sai so với nhãn')
    fig.subplots_adjust(left=0.05, right=0.995, top=0.9, bottom=0.1, wspace=0.04, hspace=0.06)
    return fig


def tile_decisions(rgb, rows, policy, cell, tau, title):
    """One tile: cells tinted by decision, with predicted (and, when the label is known, true) weed cover in each."""
    fig, ax = plt.subplots(figsize=(7.2, 7.6), dpi=90)
    labelled = rows['w'].notna().all()
    dec, need = rows[policy].to_numpy(), (rows['need'].to_numpy() if labelled else None)
    ax.imshow(_tinted(rgb, dec, cell, 0.25))
    grid_lines(ax, cell, colour='white', lw=0.8)
    _cell_marks(ax, dec, need, cell, 1, lw=3)
    per_row = C.TILE // cell
    small = cell < 200
    for k, (_, r) in enumerate(rows.iterrows()):
        rr, col = divmod(k, per_row)
        cx, cy = col * cell + cell / 2, rr * cell + cell / 2
        lines = [f'ô {k + 1}', C.DECISION_VI[int(r[policy])]] if small else [
            f'ô {k + 1}: {C.DECISION_VI[int(r[policy])]}',
            f'máy thấy {pct(r["s"], 2)}'] + (
            [f'nhãn thật {pct(r["w"], 2)}' + (' (cần phun)' if r['need'] else '')] if labelled else [])
        ax.text(cx, cy - (0 if small else cell * 0.36), '\n'.join(lines), ha='center', va='top' if not small else 'center',
                fontsize=8 if small else 9.5, color='white', fontweight='bold', path_effects=HALO, linespacing=1.4)
    ax.set_title(title, fontsize=11.5, color=INK2)
    clean(ax)
    legend_patches(fig, DECISION_LEGEND, y=0.0, ncol=4, fontsize=9.5, wrong='sai so với nhãn' if labelled else None)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.06)
    return fig


def pct(v, digits=1):
    return f'{100 * v:.{digits}f}%'.replace('.', ',')


# ---------------------------------------------------------------- explanations

def cell_panels(rgb, true3, p_weed_tile, c, colour, k_solid, k_dashed):
    """Tile with the cell boxed; the cell with the patches the explanation names; the model's P(weed) there; and the
    true label of the cell when it is known (true3 not None)."""
    cell = C.PRIMARY['cell_size']
    r, col = divmod(int(c['cell']), C.TILE // cell)
    sl = (slice(r * cell, (r + 1) * cell), slice(col * cell, (col + 1) * cell))
    boxes = CF.patch_boxes(c) if isinstance(c.get('patches'), str) and c['patches'] else []
    number = k_dashed <= CF.NUMBERED
    n = 3 if true3 is None else 4
    fig, axs = plt.subplots(1, n, figsize=(4.25 * n, 4.7), dpi=90, gridspec_kw={'width_ratios': [1, 1, 1.12, 1][:n]})
    ax = axs[0]
    ax.imshow(rgb)
    grid_lines(ax, cell)
    box = Rectangle((col * cell - 0.5, r * cell - 0.5), cell, cell, fill=False, edgecolor=colour, linewidth=3.5)
    box.set_path_effects([pe.withStroke(linewidth=6, foreground='white')])
    ax.add_patch(box)
    dated = re.match(r'\d{4}-\d{2}-\d{2}', str(c['tile']))
    ax.set_title(f'Ô ảnh {c["tile"]} (1,5 m × 1,5 m)\n' + (f'bay ngày {C.date_vi(c["tile"])}, ' if dated else '')
                 + f'ô {int(c["cell"]) + 1}', fontsize=10.5, color=INK2)
    ax = axs[1]
    ax.imshow(np.asarray(rgb)[sl])
    CF.draw_patches(ax, boxes, k_solid, k_dashed, number=number)
    ax.set_title('Ô phóng to (50 cm × 50 cm)\nvùng 5 cm × 5 cm mà lời giải thích chỉ ra', fontsize=10.5, color=INK2)
    ax = axs[2]
    im = ax.imshow(p_weed_tile[sl].astype(np.float32), cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
    CF.draw_patches(ax, boxes, k_solid, k_dashed, number=number)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(['0', '0,5', '1'])
    cb.outline.set_visible(False)
    ax.set_title('Xác suất cỏ của mô hình\n(càng đỏ, máy càng chắc là cỏ)', fontsize=10.5, color=INK2)
    if true3 is not None:
        ax = axs[3]
        ax.imshow(SEG3[true3[sl]], interpolation='nearest')
        ax.set_title('Nhãn thật của ô\n(xám: đất, xanh: ngô, đỏ: cỏ)', fontsize=10.5, color=INK2)
    for a in axs:
        clean(a)
    fig.subplots_adjust(left=0.01, right=0.98, top=0.82, bottom=0.02, wspace=0.12)
    return fig


def attribution_panels(rgb, ig, grad_cam, cell_idx, colour):
    """Integrated gradients and Grad-CAM of one cell's decision, over the whole tile, with the cell boxed."""
    cell = C.PRIMARY['cell_size']
    r, col = divmod(int(cell_idx), C.TILE // cell)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.9), dpi=90)
    lim = float(np.abs(ig).max()) or 1.0
    pool = C.TILE / ig.shape[0]
    panels = ((rgb, None, 'Ảnh màu thật', None),
              (ig, dict(cmap='RdBu_r', vmin=-lim, vmax=lim), 'Integrated Gradients (IG)\nđỏ: đẩy lượng cỏ lên · xanh: kéo xuống', None),
              (grad_cam, dict(cmap='Greys', vmin=0, vmax=float(grad_cam.max()) or 1.0),
               'Grad-CAM\ncàng đậm, càng "quan trọng"', None))
    for ax, (img, kw, title, _) in zip(axs, panels):
        if kw is None:
            ax.imshow(img)
            s = 1
        else:
            ax.imshow(img.astype(np.float32), interpolation='nearest', **kw)
            s = pool
        c = cell / s
        for t in range(1, C.TILE // cell):
            ax.axhline(t * c - 0.5, color='#c3c2b7', lw=0.6)
            ax.axvline(t * c - 0.5, color='#c3c2b7', lw=0.6)
        box = Rectangle((col * c - 0.5, r * c - 0.5), c, c, fill=False, edgecolor=colour, linewidth=3)
        box.set_path_effects([pe.withStroke(linewidth=5, foreground='white')])
        ax.add_patch(box)
        ax.set_title(title, fontsize=10.5, color=INK2)
        clean(ax)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.02, wspace=0.06)
    return fig


def removal_curve(c, steps, estimate_only=False):
    """Predicted weed area of the cell as its ranked patches are removed: the model's rescoring and the fast estimate,
    or the estimate alone (estimate_only: steps carry the estimate in s_model)."""
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=90)
    CF.draw_curve(ax, c, steps, estimate=not estimate_only)
    second = ('đường liền: ước lượng nhanh (chưa cho mô hình chấm lại)' if estimate_only else
              'đường liền có chấm: mô hình chấm lại · đường đứt: ước lượng nhanh')
    ax.set_title('Lượng cỏ dự đoán khi bỏ dần từng vùng (vùng nhiều cỏ nhất trước)\n' + second, fontsize=10.5,
                 color=INK2)
    fig.tight_layout()
    return fig
