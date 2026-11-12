"""Run the system on one tile, module by module: input (a dataset tile or five uploaded band PNGs) -> 1. find weed
(the authors' model on CPU) -> 2. decide every cell -> 3. explain one cell. Heavy steps (model check, integrated
gradients) run only on request, with their expected time shown first."""
import hashlib

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import core as C
import figures as F
import pipeline as PL
import ui

CF = C.CF
CELL, TAU = PL.CELL, PL.TAU
CELL_CM2 = C.cell_cm2(CELL)
SPLIT_VI = {'train': 'tập huấn luyện', 'val': 'tập chọn mô hình (val)', 'test': 'tập kiểm tra'}
# Integrated gradients on CPU, measured on the user's laptop (i7-1165G7, 4 threads, 12/11/2026): a forward pass of a
# 600 x 600 tile took 2.7 s; two IG steps on a 300 x 300 crop took 4.1 s including the closing forward pass of two
# images, i.e. a step (forward + backward) about twice a forward pass of the same size, and the process grew by about
# 0.4 GB. A tile has four times the pixels: about 1.5 GB more memory. Both constants round these up.
IG_STEP_FACTOR = 3     # seconds of one IG step / seconds of one forward pass
IG_RAM_GB = 2          # free memory to have before computing IG on a tile

ui.need_data()
st.title('Chạy thử Hệ thống hỗ trợ ra quyết định phun thuốc trừ cỏ')
st.markdown('Đưa vào **một ô ảnh drone 5 dải phổ** (1,5 m × 1,5 m). Hệ thống chạy lần lượt ba module, và trang hiện '
            'kết quả của từng module:\n\n'
            '**Ảnh 5 dải** → **① Tìm cỏ** (mạng DeepLabv3+) → **② Quyết định từng ô 50 cm** (hai ngưỡng conformal) → '
            '**③ Giải thích** (vì sao, đổi gì thì quyết định đổi, dải phổ nào quan trọng)')


# ---------------------------------------------------------------- model

if C.MAIN_MODEL not in C.available_models():
    st.error('Thiếu bản đồ xác suất đã lưu của mô hình chính (`results/weeds_authors/dlv3p_msi_3/probs_*.npz`, không có '
             'trên GitHub): hai ngưỡng conformal được hiệu chỉnh trên đó. Giải nén gói kết quả Colab vào `results/`.')
    st.stop()
if not PL.CKPT.exists():
    if PL.CKPT_ZIP.exists():
        try:
            with st.spinner('Lấy mô hình dlv3p_msi_3 ra khỏi ckpts.zip (một lần)…'):
                PL.extract_checkpoint()
        except ValueError as e:
            st.error(str(e))
            st.stop()
    else:
        st.error(f'Chưa có mô hình của tác giả. Tải `ckpts.zip` (887 MB) từ {PL.CKPT_URL} (hoặc từ Google Drive, thư mục '
                 '`De_Tai_Nong_San/weeds_pilot/authors_ckpts/`) và đặt vào `data/weedsgalore/ckpts/`.')
        st.stop()


@st.cache_resource(show_spinner='Đang nạp mô hình phân vùng…')
def model():
    return PL.load_model()


@st.cache_data(max_entries=8, show_spinner=False)
def segment(x):
    return PL.segment(model(), x)


@st.cache_data(max_entries=16, show_spinner=False)
def check_cell(x, row):
    return PL.model_check(model(), x, pd.Series(row))


@st.cache_data(max_entries=16, show_spinner=False)
def attribute_cell(x, cell_idx):
    return PL.attributions(model(), x, cell_idx)


@st.cache_data(max_entries=8, show_spinner=False)
def sample_files(tile_id):
    return PL.sample_files(tile_id)


# ---------------------------------------------------------------- input

st.header('Đầu vào: một ô ảnh 5 dải phổ')
ids = ui.dataset_ids()
source = st.segmented_control('Nguồn ảnh', ['dataset', 'upload'], default='dataset', required=True,
                              format_func={'dataset': 'Chọn ô ảnh có sẵn trong WeedsGalore',
                                           'upload': 'Upload 5 file ảnh (R, G, B, NIR, RE)'}.get)
cropped = False
if source == 'dataset':
    examples = CF.pick_examples(ui.explanation_cells()[0])
    first = examples[0][1]['tile'] if examples else ids['test'][0]
    a, b, c = st.columns([1.2, 2.5, 1.6], vertical_alignment='bottom')
    split = a.selectbox('Tập', ['test', 'val', 'train'], format_func=lambda s: SPLIT_VI[s].capitalize())
    tiles = ids[split]
    tile_id = b.selectbox('Ô ảnh', tiles, index=tiles.index(first) if first in tiles else 0,
                          format_func=lambda t: f'{t} · bay ngày {C.date_vi(t)}')
    c.download_button('Tải 5 file của ô này để thử upload', sample_files(tile_id), f'{tile_id}_5_dai.zip',
                      'application/zip', icon=':material/download:', width='stretch')
    x, mask = ui.tile(tile_id)
    if split == 'train':
        st.warning('Ô này thuộc tập huấn luyện: mô hình đã học trên chính ô này nên kết quả sẽ tốt hơn thực tế. Nên chọn '
                   'ô của tập kiểm tra.', icon=':material/warning:')
else:
    files = st.file_uploader('Năm file PNG 16 bit của một ô ảnh, tên kết thúc bằng _R, _G, _B, _NIR, _RE (như trong '
                             'WeedsGalore), hoặc một file zip chứa 5 file đó', type=['png', 'zip'],
                             accept_multiple_files=True)
    if not files:
        st.info('Chưa có file. Muốn thử nhanh: chọn "Chọn ô ảnh có sẵn", bấm "Tải 5 file của ô này", rồi upload file zip '
                'vừa tải (hoặc 5 file bên trong). Ảnh lớn hơn 600 × 600 điểm ảnh sẽ được cắt lấy góc trên bên trái.',
                icon=':material/upload:')
        ui.footer()
        st.stop()
    try:
        x_full, tile_id = PL.read_band_files([(f.name, f.getvalue()) for f in files])
        x, cropped = PL.fit_tile(x_full)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    split, mask = (None, None) if tile_id is None else PL.match_dataset(x_full, tile_id, ids, read=ui.tile)
    if split:
        st.success(f'Năm dải khớp từng điểm ảnh với ô `{tile_id}` của WeedsGalore ({SPLIT_VI[split]}): trang dùng được '
                   'nhãn thật để chấm kết quả.', icon=':material/check_circle:')
        if split == 'train':
            st.warning('Ô này thuộc tập huấn luyện: mô hình đã học trên chính ô này nên kết quả sẽ tốt hơn thực tế.',
                       icon=':material/warning:')
    else:
        st.info('Ảnh không có trong WeedsGalore nên không có nhãn thật: trang không biết quyết định đúng hay sai. Ảnh chụp '
                'bằng loại máy ảnh khác, ở độ cao khác hay ở ruộng khác thì kết quả tìm cỏ có thể không đáng tin, và '
                'ngưỡng conformal không còn bảo đảm mức sai.', icon=':material/info:')
        if cropped:
            st.caption(f'Ảnh gốc {x_full.shape[2]} × {x_full.shape[1]} điểm ảnh, đã cắt lấy {C.TILE} × {C.TILE} ở góc trên '
                       'bên trái.')
    # an upload that is not exactly a dataset tile keeps its file name, marked so it is not taken for that tile
    tile_id = tile_id if split else (f'{tile_id} (upload)' if tile_id else 'anh_upload')

lo, hi = map(np.asarray, ui.band_limits())
rgb = np.asarray(C.composite(x, (C.R, C.G, C.B), lo, hi))
a, b, c = st.columns([1, 1, 1.3])
a.image(rgb, caption='Ảnh màu thật (R, G, B)', width='stretch')
b.image(np.asarray(C.composite(x, (C.NIR, C.R, C.G), lo, hi)), caption='Ảnh màu giả (NIR, R, G): thực vật màu đỏ',
        width='stretch')
with c:
    st.markdown(f'**Ô ảnh:** `{tile_id}`\n\n'
                f'**Kích thước:** {x.shape[2]} × {x.shape[1]} điểm ảnh, {len(C.BANDS)} dải ({", ".join(C.BANDS)}), 16 bit\n\n'
                f'**Ngoài ruộng:** {ui.num(x.shape[2] * C.GSD_CM / 100, 1)} m × {ui.num(x.shape[1] * C.GSD_CM / 100, 1)} m '
                f'(2,5 mm mỗi điểm ảnh), chia thành {(C.TILE // CELL) ** 2} ô 50 cm × 50 cm\n\n'
                f'**Nhãn thật:** {"có (" + SPLIT_VI[split] + ")" if mask is not None else "không có"}')
true3 = C.to_3class(mask) if mask is not None else None


# ---------------------------------------------------------------- 1. segmentation

st.header('① Tìm cỏ trên ảnh')
st.caption('Mô hình DeepLabv3+ do tác giả bộ dữ liệu công bố (ảnh 5 dải, 3 lớp), chạy trên CPU của máy này. Đầu ra: xác '
           'suất mỗi điểm ảnh là nền đất, ngô hay cỏ.')
try:
    with st.spinner('Mô hình đang tìm cỏ trên CPU (khoảng vài giây đến vài chục giây)…'):
        probs, seconds = segment(x)
except Exception as err:  # a damaged checkpoint, or too little memory
    st.error(f'Không chạy được mô hình ({type(err).__name__}: {err}). Nếu file `{PL.CKPT.name}` bị hỏng, xóa nó đi: '
             'trang sẽ lấy lại từ `ckpts.zip`.')
    st.stop()
p_weed = probs[2]
pred3 = probs.argmax(0).astype(np.uint8)
ui.show(F.segmentation_views(rgb, true3, pred3, p_weed))
m1, m2, m3 = st.columns(3)
m1.metric('Thời gian chạy mô hình', f'{ui.num(seconds, 1)} giây', border=True, help='Lần chạy đầu tiên cho ô ảnh này; '
          'kết quả được giữ lại nên lần sau hiện ngay.')
m2.metric('Diện tích cỏ dự đoán', ui.pct(float(p_weed.mean()), 2), border=True,
          help='Trung bình xác suất cỏ trên cả ô ảnh.')
if true3 is not None:
    iou = C.iou_per_class(pred3, true3)
    m3.metric('IoU cỏ trên ô ảnh này', ui.iou_text(iou[2]), border=True,
              help=f'Nền đất {ui.iou_text(iou[0])}, ngô {ui.iou_text(iou[1])}. Một ô ảnh ít cỏ có thể có IoU thấp dù sai '
                   'rất ít điểm ảnh.')
if split == 'test':
    saved_ids, _, saved_weed = ui.probs(C.MAIN_MODEL, 'test')
    i = list(saved_ids).index(tile_id)
    diff = float(np.abs(saved_weed[i].astype(np.float32) - p_weed).max())
    st.caption(f':material/check_circle: So với bản đồ xác suất đã chạy trên Colab (float16, GPU): chênh lệch lớn nhất '
               f'{ui.num(diff, 4)} (máy này chạy float32 trên CPU).')


# ---------------------------------------------------------------- 2. decisions

st.header('② Quyết định cho từng ô 50 cm × 50 cm')
_, fold_of = ui.locations()
thr = PL.thresholds(tile_id, split, ui.cells(C.MAIN_MODEL, 'test', CELL), fold_of)
cells = PL.decide_tile(p_weed, mask, tile_id, thr)
no_thr, sp_thr = min(thr['lam_lo'], thr['lam_hi']), max(thr['lam_lo'], thr['lam_hi'])
where = ('các ô ảnh kiểm tra ở **phần vị trí còn lại** của ruộng (đúng như cách tính trong báo cáo)'
         if thr['how'] == 'crossfit' else 'toàn bộ ô của tập kiểm tra, 3 đợt bay trước ngày phun thuốc')
st.markdown(
    f'Lượng cỏ dự đoán của mỗi ô = trung bình xác suất cỏ trong ô. **Ngưỡng phun {ui.num(sp_thr * CELL_CM2, 1)} cm²**, '
    f'**ngưỡng không phun {ui.num(no_thr * CELL_CM2, 1)} cm²** (trên ô 2 500 cm²), tính bằng conformal trên {where}: '
    f'{thr["n_need"]} ô cần phun và {thr["n_no_need"]} ô không cần phun, để bỏ sót và phun nhầm không quá '
    f'{ui.pct(C.ALPHA, 0)}. Ô "cần phun" là ô có cỏ từ {ui.pct(TAU, 0)} diện tích trở lên (ngưỡng minh họa).')
if split is not None and C.tile_date(tile_id) not in C.SCOPES[C.PRIMARY['scope']]:
    st.caption(':material/info: Đợt bay này diễn ra sau ngày phun thuốc (12/6), ngoài phạm vi đánh giá trong báo cáo (3 '
               'đợt bay trước ngày phun). Quyết định vẫn tính được, nhưng tỉ lệ bỏ sót và phun nhầm chưa được kiểm tra '
               'cho đợt bay này.')
left, right = st.columns([3, 2])
with left:
    ui.show(F.tile_decisions(rgb, cells, 'decision', CELL, TAU, f'{tile_id}: quyết định của hệ thống'))
with right:
    counts = cells['decision'].value_counts()
    k1, k2, k3 = st.columns(3)
    for col, d in zip((k1, k2, k3), (C.SPRAY, C.DEFER, C.NO_SPRAY)):
        col.metric(C.DECISION_VI[d], f'{int(counts.get(d, 0))} ô', border=True)
    view = {'Ô': cells['cell'] + 1, 'Cỏ dự đoán': [f'{ui.num(s * CELL_CM2, 1)} cm² ({ui.pct(s, 2)})' for s in cells['s']],
            'Quyết định': [C.DECISION_VI[int(d)] for d in cells['decision']]}
    if mask is not None:
        view['Cỏ theo nhãn'] = [ui.pct(w, 2) for w in cells['w']]
        view['Kết quả'] = ['bỏ sót' if d == C.NO_SPRAY and n else 'phun nhầm' if d == C.SPRAY and not n
                           else 'người ra xem' if d == C.DEFER else 'đúng'
                           for d, n in zip(cells['decision'], cells['need'])]
    st.dataframe(pd.DataFrame(view), hide_index=True)
    st.caption('Ô đánh số từ trái sang phải, trên xuống dưới. Ô CHƯA CHẮC: người ra xem, cần thì phun.')
    if thr['how'] == 'crossfit':
        saved = pd.read_csv(C.RESULTS / 'weeds_xai/decisions.csv')
        saved = saved[saved['tile'] == tile_id].set_index('cell')['decision']
        same = int((cells.set_index('cell')['decision'] == saved).sum())
        st.caption(f':material/check_circle: Trùng quyết định đã báo cáo ở {same}/{len(saved)} ô '
                   '(`results/weeds_xai/decisions.csv`).')


# ---------------------------------------------------------------- 3. explanations

st.header('③ Giải thích quyết định của một ô')
est = PL.explain_estimate(p_weed, cells).set_index('cell')


def removal(k, outcome, what='vùng', removed_cm2=None):
    """'bỏ k vùng (… cm²) thì <outcome>', or the sentence for an outcome that removing every patch does not reach."""
    if not np.isfinite(k):
        return f'bỏ hết các vùng của ô vẫn chưa {outcome}'
    area = f' ({ui.num(removed_cm2, 1)} cm² cỏ dự đoán)' if removed_cm2 is not None else ''
    return f'bỏ {int(k)} {what}{area} thì {outcome}'


# SPRAY and DEFER cells first (the ones with a counterfactual to show), the most fragile SPRAY first
order = sorted(cells['cell'].tolist(), key=lambda k: (bool(cells.loc[k, 'decision'] == C.NO_SPRAY),
                                                      bool(cells.loc[k, 'decision'] != C.DEFER),
                                                      float(np.nan_to_num(est.loc[k, 'k_not_spray'], nan=1e9)), k))
k = st.selectbox('Chọn ô', order, format_func=lambda k: f'ô {k + 1} · {C.DECISION_VI[int(cells.loc[k, "decision"])]} · '
                                                       f'cỏ dự đoán {ui.num(cells.loc[k, "s"] * CELL_CM2, 1)} cm²')
row, e = cells.loc[k], est.loc[k]
decision = int(row['decision'])
colour = C.DECISION_COLOR[decision]
weed = row['s'] * CELL_CM2
n_cell_patches = (CELL // C.PATCH) ** 2

# The Colab runs explained the reported decisions: every cell of the test tiles of the pooled dates. For such a cell,
# when this machine gives the same decision, their results are used at once; otherwise the model check runs here on
# request. Whatever is available is fetched before the explanation is written, so the sentences, the figure and the
# check below always state the same numbers: checked by the model when there is a check, else the fast estimate.
reported = thr['how'] == 'crossfit'
cf_cells, cf_steps = ui.explanation_cells()
colab_cf = cf_cells[(cf_cells['tile'] == tile_id) & (cf_cells['cell'] == k)] if reported else cf_cells.iloc[:0]
colab_cf = colab_cf.iloc[0] if len(colab_cf) and int(colab_cf.iloc[0]['decision']) == decision else None
xai = ui.xai_cells()
colab_xai = xai[(xai['tile'] == tile_id) & (xai['cell'] == k)] if reported else xai.iloc[:0]
colab_xai = colab_xai.iloc[0] if len(colab_xai) and int(colab_xai.iloc[0]['decision']) == decision else None
x_key = hashlib.sha1(x.tobytes()).hexdigest()[:12]
check_key = f'check:{x_key}:{k}'

checked, check_steps, check_source = None, None, None
if decision != C.NO_SPRAY:
    if colab_cf is not None:
        checked = colab_cf
        check_steps = cf_steps[(cf_steps['tile'] == tile_id) & (cf_steps['cell'] == k)].sort_values('k')
        check_source = 'trên Colab (GPU T4)'
    elif st.session_state.get(check_key):
        with st.spinner('Mô hình đang chấm lại từng bước…'):
            check_steps, res, n_pass, sec = check_cell(x, row[['cell', 'decision', 'lam_lo', 'lam_hi']].to_dict())
        checked = pd.Series({**res, 'lam_lo': row['lam_lo'], 'lam_hi': row['lam_hi']})
        check_source = f'trên máy này ({n_pass} ảnh, {ui.num(sec, 0)} giây)'

if checked is not None:
    k_not_v = checked['k_not_spray_model'] if decision == C.SPRAY else 0.0
    k_no_v = checked['k_no_spray_model']
    patches = checked['patches']

    def removed_area(kk):
        return CF.removed_cm2(check_steps, int(kk)) if np.isfinite(kk) else None

    area_not, area_no = removed_area(k_not_v), removed_area(k_no_v)
    tag = '(mô hình đã chấm lại ' + ('trên Colab' if colab_cf is not None else 'trên máy này') + ')'
else:
    k_not_v = e['k_not_spray'] if decision == C.SPRAY else 0.0
    k_no_v = e['k_no_spray'] if decision != C.NO_SPRAY else 0.0
    patches = e['patches']
    area_not, area_no = e['removed_cm2_not_spray'], e['removed_cm2_no_spray']
    tag = '(ước lượng nhanh, chưa cho mô hình chấm lại)'

st.markdown(f'#### {ui.decision_badge(decision)} &nbsp; ô {k + 1}', unsafe_allow_html=True)
lines = []
if decision == C.SPRAY:
    lines.append(f'**Vì sao PHUN:** máy thấy khoảng {ui.num(weed, 1)} cm² cỏ trong ô ({ui.pct(row["s"], 2)} diện tích), '
                 f'từ ngưỡng phun {ui.num(sp_thr * CELL_CM2, 1)} cm² trở lên.')
    lines.append(f'**Đổi gì thì quyết định đổi** {tag}: '
                 + removal(k_not_v, 'ô hết PHUN', 'vùng 5 cm × 5 cm nhiều cỏ nhất', area_not)
                 + '; ' + removal(k_no_v, 'KHÔNG PHUN') + '.')
elif decision == C.DEFER:
    lines.append(f'**Vì sao CHƯA CHẮC:** máy thấy khoảng {ui.num(weed, 1)} cm² cỏ, nằm giữa ngưỡng không phun '
                 f'{ui.num(no_thr * CELL_CM2, 1)} cm² và ngưỡng phun {ui.num(sp_thr * CELL_CM2, 1)} cm².')
    lines.append(f'**Đổi gì thì quyết định đổi** {tag}: '
                 + removal(k_no_v, 'KHÔNG PHUN', 'vùng nhiều cỏ nhất', area_no)
                 + f'; thêm khoảng {ui.num(e["add_to_spray_cm2"], 1)} cm² cỏ thì PHUN (theo quy tắc ngưỡng). Người '
                   'kiểm tra nên xem các vùng này trước.')
else:
    lines.append(f'**Vì sao KHÔNG PHUN:** máy thấy khoảng {ui.num(weed, 1)} cm² cỏ, dưới ngưỡng không phun '
                 f'{ui.num(no_thr * CELL_CM2, 1)} cm².')
    lines.append(f'**Đổi gì thì quyết định đổi:** thêm khoảng {ui.num(e["add_to_spray_cm2"], 1)} cm² cỏ dự đoán thì ô '
                 'thành PHUN (theo quy tắc ngưỡng).')
if mask is not None:
    lines.append(f'**Thực tế (theo nhãn):** cỏ chiếm {ui.pct(row["w"], 2)} diện tích ô → '
                 + ('ô thật sự cần phun.' if row['need'] else 'ô thực sự không cần phun.'))
st.markdown('\n'.join(f'- {line}' for line in lines))

k_not = int(np.nan_to_num(k_not_v, nan=n_cell_patches))
k_no = int(np.nan_to_num(k_no_v, nan=n_cell_patches))
panel_row = pd.Series({'tile': tile_id, 'cell': k, 'patches': patches if isinstance(patches, str) else ''})
ui.show(F.cell_panels(rgb, true3, p_weed, panel_row, colour, k_not, k_no))
if decision != C.NO_SPRAY:
    st.caption('Viền liền: các vùng cần bỏ để ô hết PHUN; viền đứt: các vùng bỏ thêm để thành KHÔNG PHUN. Số trên hình là '
               'thứ tự bỏ (vùng nhiều cỏ dự đoán nhất trước).')


def check_message(res, how):
    # against the estimate of the same run (the dashed line of its curve): the Colab run's estimate came from its own map
    same = (res['k_no_spray_model'] == res['k_no_spray_estimate']
            and (decision != C.SPRAY or res['k_not_spray_model'] == res['k_not_spray_estimate']))
    return (f'Mô hình chấm lại {how}: '
            + (removal(res['k_not_spray_model'], 'hết PHUN') + ', ' if decision == C.SPRAY else '')
            + removal(res['k_no_spray_model'], 'KHÔNG PHUN') + '; '
            + ('trùng ước lượng nhanh.' if same else 'khác ước lượng nhanh, nên lời giải thích ở trên dùng số của mô '
               'hình.'))


if decision != C.NO_SPRAY:
    st.subheader('Kiểm chứng lời giải thích bằng mô hình')
    st.markdown('Ước lượng nhanh chỉ trừ phần xác suất cỏ của các vùng bị bỏ. Muốn chắc, **xóa thật** các vùng đó khỏi '
                'ảnh (thay bằng đất trống) rồi cho mô hình chấm lại từng bước.')
    if checked is not None:
        if np.isfinite([k_not_v, k_no_v]).all():
            ui.show(F.removal_curve(checked, check_steps))
        st.success(check_message(checked, check_source), icon=':material/verified:')
        if colab_cf is not None:
            st.caption('Nguồn: `results/weeds_cfcheck/` (cùng mô hình, cùng quyết định với ô này).')
    else:
        passes = PL.planned_passes(e)
        wait = passes * seconds
        if wait > 60:
            st.caption(f':material/schedule: Ô này cần bỏ nhiều vùng nên kiểm chứng lâu (từ khoảng '
                       f'{ui.num(wait / 60, 0 if wait >= 120 else 1)} phút trở lên). Ô ít cỏ hoặc CHƯA CHẮC chỉ mất vài '
                       'giây đến vài chục giây.')
        # on_click runs before the rerun, so on that rerun the explanation above already uses the check
        st.button(f'Cho mô hình chấm lại trên máy này (ít nhất {passes} ảnh, ~{ui.num(wait, 0)} giây trở lên)',
                  icon=':material/play_arrow:', key='btn_' + check_key,
                  on_click=lambda: st.session_state.update({check_key: True}))
        # a few steps past NO_SPRAY, so the curve ends well below the zone label at the right edge
        steps_est = PL.estimate_steps(p_weed, k, row['lam_lo'], row['lam_hi']).iloc[:max(k_no + 8, 2 * k_no + 2)]
        c_est = {'decision_rescored': decision, 'k_not_spray_model': k_not, 'k_no_spray_model': k_no,
                 'lam_lo': row['lam_lo'], 'lam_hi': row['lam_hi']}
        ui.show(F.removal_curve(pd.Series(c_est), steps_est, estimate_only=True))


def show_attributions(ig_map, cam_map, band_score, total_score, gap_score, source):
    """IG and Grad-CAM over the tile, and IG per band in cm^2 of predicted weed (scores times the cell area)."""
    ui.show(F.attribution_panels(rgb, ig_map, cam_map, k, colour))
    bands = pd.DataFrame({'band': list(C.BANDS), 'cm2': np.asarray(band_score) * CELL_CM2})
    bands['text'] = [('+' if v >= 0 else '−') + ui.num(abs(v), 1) for v in bands['cm2']]
    base = alt.Chart(bands).encode(y=alt.Y('band:N', sort=list(C.BANDS), title=None),
                                   x=alt.X('cm2:Q', title='đóng góp vào lượng cỏ dự đoán của ô (cm²)'))
    a, b = st.columns([3, 2])
    with a:
        st.altair_chart((base.mark_bar(color='#2a78d6', cornerRadiusEnd=4, size=22).encode(
            tooltip=[alt.Tooltip('band:N', title='dải'), alt.Tooltip('cm2:Q', title='cm²', format='.1f')])
            + base.mark_text(align='left', dx=4, color='#0b0b0b').encode(text='text:N')).properties(height=220),
            width='stretch')
    with b:
        top = bands.sort_values('cm2', ascending=False).iloc[0]
        st.markdown(f'Dải **{top["band"]}** đẩy lượng cỏ dự đoán lên nhiều nhất ({ui.num(top["cm2"], 1)} cm²).')
        st.caption(f'Tổng các dải {ui.num(total_score * CELL_CM2, 1)} cm²; lượng cỏ dự đoán của ô so với ảnh đất trống '
                   f'{ui.num(gap_score * CELL_CM2, 1)} cm² (hai số phải gần bằng nhau). {source} IG trả lời "vì sao là '
                   'cỏ mà không phải đất", không trả lời "vì sao là cỏ mà không phải ngô".')
    st.caption('Trang "Kiểm chứng lời giải thích" cho thấy IG trung thực, còn Grad-CAM gần như giống nhau cho mọi ô của '
               'cùng ô ảnh nên không giải thích được từng ô.')


st.subheader('Dải phổ và vùng ảnh nào quan trọng')
st.markdown('**Integrated Gradients (IG)** chia lượng cỏ dự đoán của ô cho từng điểm ảnh và từng dải phổ, so với một ảnh '
            '"đất trống". **Grad-CAM** là phương pháp phổ biến khác, đưa vào để so sánh.')
maps = ui.xai_maps()
i_map = np.flatnonzero((maps['tile'] == tile_id) & (maps['cell'] == k)) if colab_xai is not None else []
if len(i_map):
    show_attributions(maps['ig'][i_map[0]], maps['grad_cam'][i_map[0]], [colab_xai[f'ig_{b}'] for b in C.BANDS],
                      colab_xai['ig_sum'], colab_xai['ig_gap'], 'Đã tính trên Colab (GPU T4), nguồn `results/weeds_xai/`.')
else:
    key_ig = f'ig:{x_key}:{k}'
    ig_minutes = PL.IG_STEPS * IG_STEP_FACTOR * seconds / 60
    if st.button(f'Tính IG và Grad-CAM cho ô {k + 1} trên máy này (khoảng {ui.num(max(ig_minutes, 1), 0)} phút, cần '
                 f'khoảng {ui.num(IG_RAM_GB, 1)} GB RAM trống)', icon=':material/play_arrow:', key='btn_' + key_ig,
                 help='IG chạy xuôi và ngược mô hình 32 lần trên ảnh 600 × 600. Máy thiếu RAM sẽ chạy rất chậm: nên '
                      'đóng bớt ứng dụng khác trước khi bấm.'):
        st.session_state[key_ig] = True
    if st.session_state.get(key_ig):
        with st.spinner(f'Đang tính IG ({PL.IG_STEPS} bước, mỗi bước chạy xuôi và ngược mô hình)…'):
            ig, gap, cam, sec = attribute_cell(x, k)
        show_attributions(ig.sum(0), cam, ig.sum((1, 2)), ig.sum(), gap,
                          f'Tính trên máy này trong {ui.num(sec, 0)} giây.')
ui.footer()
