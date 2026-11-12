"""Streamlit side of the demo: cached loaders around core, number formatting (Vietnamese decimal comma) and small
shared widgets. Large arrays are cached as resources (shared, never mutated); tables as data."""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import core as C

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


def show(fig):
    st.pyplot(fig, clear_figure=True)  # width='stretch' by default
    plt.close(fig)


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
