"""Streamlit side of the demo: cached loaders around core, number formatting (Vietnamese decimal comma) and small
shared widgets. Large arrays are cached as resources (shared, never mutated); tables as data."""
import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import core as C
import figures as F

SOURCE = ('Dữ liệu: WeedsGalore (Celikkan và cộng sự, IEEE/CVF WACV 2025), giấy phép CC BY 4.0. '
          'Ngưỡng "cần phun" τ = 1% diện tích ô chỉ là ngưỡng minh họa, chưa có tài liệu làm căn cứ.')


# ---------------------------------------------------------------- formatting

def num(v, digits=0):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '–'
    return f'{v:,.{digits}f}'.replace(',', ' ').replace('.', ',')


def pct(v, digits=1):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '–'
    return f'{100 * v:.{digits}f}%'.replace('.', ',')


def iou_text(v, digits=1):
    """IoU in % for display; a class absent from both the label and the prediction has none."""
    return f'{num(float(v), digits)}%' if math.isfinite(float(v)) else 'không có (lớp này không xuất hiện)'


@st.cache_data(max_entries=128, show_spinner=False)
def _figure_png(name, args, kwargs):
    fig = getattr(F, name)(*args, **kwargs)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=fig.dpi, bbox_inches='tight', facecolor='white')  # as st.pyplot saves it
    plt.close(fig)
    return buf.getvalue()


def figure(name, *args, width='stretch', **kwargs):
    """Show figures.<name>(*args, **kwargs). Each figure is drawn once per distinct input and kept as a PNG: a rerun
    caused by another widget (choosing a cell, moving a slider) shows it again at once instead of drawing it anew.
    width='content' shows a small figure at its own size (no larger than the column) instead of stretching it."""
    st.image(_figure_png(name, args, kwargs), width=width)


def band_chart(values, x_title, digits=1, suffix='', height=220, extra_tooltip=None):
    """One horizontal bar per band with its value written at the end of the bar (left of a negative bar); the axis
    leaves room for the labels so none is cut. values: one number per band, in BANDS order."""
    import altair as alt
    d = pd.DataFrame({'band': list(C.BANDS), 'value': np.asarray(values, dtype=float)})
    d['text'] = [('+' if v >= 0 else '−') + num(abs(float(v)), digits) + suffix for v in d['value']]
    lo, hi = min(float(d['value'].min()), 0.0), max(float(d['value'].max()), 0.0)
    pad = 0.25 * ((hi - lo) or 1.0)
    base = alt.Chart(d).encode(
        y=alt.Y('band:N', sort=list(C.BANDS), title=None, axis=alt.Axis(labelFontSize=12)),
        x=alt.X('value:Q', title=x_title, scale=alt.Scale(domain=[lo - (pad if lo < 0 else 0), hi + pad], nice=False)))
    tips = [alt.Tooltip('band:N', title='dải'), alt.Tooltip('text:N', title='giá trị')] + (extra_tooltip or [])
    bars = base.mark_bar(color='#2a78d6', cornerRadiusEnd=4, size=22).encode(tooltip=tips)
    pos = base.transform_filter('datum.value >= 0').mark_text(align='left', dx=4, color='#0b0b0b').encode(text='text:N')
    neg = base.transform_filter('datum.value < 0').mark_text(align='right', dx=-4, color='#0b0b0b').encode(text='text:N')
    st.altair_chart((bars + pos + neg).properties(height=height), width='stretch')


def need_data():
    """Stop the page with a message when the dataset zip is missing."""
    if not C.ZIP.exists():
        st.error(f'Không tìm thấy bộ dữ liệu `{C.ZIP}`. Tải `weedsgalore-dataset.zip` (xem README.md) và đặt vào '
                 '`data/weedsgalore/`.')
        st.stop()


def model_picker(key='model', label='Mô hình phân vùng', models=None, help=None):
    models = models or C.available_models()
    if not models:
        st.error('Chưa có bản đồ xác suất `probs_*.npz` trong `results/` (file này không có trên GitHub; giải nén gói '
                 'kết quả tải về từ Colab).')
        st.stop()
    return st.selectbox(label, models, format_func=lambda k: C.MODELS[k]['label'], key=key, help=help)


def footer():
    st.divider()
    st.caption(SOURCE)


# ---------------------------------------------------------------- cached loaders

@st.cache_data(show_spinner=False)
def dataset_ids():
    return C.dataset_ids()


@st.cache_data(persist='disk', show_spinner='Lần chạy đầu: đang tính độ sáng chung cho ảnh (khoảng 20 giây, chỉ một lần)…')
def band_limits():
    lo, hi = C.dataset_band_limits()
    return lo.tolist(), hi.tolist()


@st.cache_resource(show_spinner=False, max_entries=48)
def tile(tile_id):
    """(bands, semantic mask) of one tile."""
    return C.read_bands(tile_id)


@st.cache_resource(show_spinner=False, max_entries=64)
def rgb(tile_id):
    lo, hi = map(np.asarray, band_limits())
    return np.asarray(C.composite(tile(tile_id)[0], (C.R, C.G, C.B), lo, hi))


@st.cache_resource(show_spinner=False, max_entries=64)
def thumbnail(tile_id, step=3):
    return rgb(tile_id)[::step, ::step]


@st.cache_resource(show_spinner=False)
def probs(key, split):
    ids, p_crop, p_weed = C.load_probs(key, split)
    C.check_order(ids, dataset_ids()[split])
    return ids, p_crop, p_weed


@st.cache_resource(show_spinner=False)
def masks(split):
    """Semantic masks of a split, in the order of its split file."""
    return C.read_masks(dataset_ids()[split])


@st.cache_data(show_spinner=False)
def cells(key, split, cell):
    ids, _, p_weed = probs(key, split)
    return C.cell_table(ids, C.to_3class(masks(split)[:len(ids)]), p_weed, cell)


@st.cache_data(show_spinner=False)
def locations():
    return C.locations(dataset_ids()['test'])


@st.cache_data(show_spinner=False)
def decide(key, cell, tau, alpha, beta, scope, calibration):
    _, fold_of = locations()
    return C.decide(cells(key, 'test', cell), cells(key, 'val', cell), fold_of, tau, alpha, beta, C.SCOPES[scope],
                    calibration)


@st.cache_data(show_spinner=False)
def saved_metrics(key):
    return C.saved_metrics(key)


@st.cache_data(show_spinner=False)
def explanation_cells():
    return C.explanation_cells()


@st.cache_data(show_spinner=False)
def no_spray_cells():
    return C.no_spray_cells()


@st.cache_resource(show_spinner=False)
def xai_maps():
    return C.load_xai_maps()


@st.cache_data(show_spinner=False)
def xai_cells():
    return pd.read_csv(C.RESULTS / 'weeds_xai/xai_cells.csv')


@st.cache_data(show_spinner=False)
def read_json(rel):
    return C.read_json(C.RESULTS / rel)


def decision_badge(d):
    colour = C.DECISION_COLOR[int(d)]
    return (f'<span style="background:{colour};color:white;padding:2px 10px;border-radius:4px;font-weight:600">'
            f'{C.DECISION_VI[int(d)]}</span>')
