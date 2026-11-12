"""Explanation of one cell's decision: why (predicted weed against the thresholds), what would change it (the
counterfactual as the model rescored it on Colab), which bands drive it (integrated gradients) and the attribution
maps. Exists for the main model at the primary setting, the decisions the Colab runs explained."""
import numpy as np
import streamlit as st

import core as C
import ui

CF = C.CF
CELL = C.PRIMARY['cell_size']
CELL_CM2 = C.cell_cm2(CELL)

ui.need_data()
if C.MAIN_MODEL not in C.available_models() or not (C.RESULTS / 'weeds_xai/xai_maps.npz').exists():
    st.error('Thiếu bản đồ xác suất của mô hình chính hoặc `results/weeds_xai/xai_maps.npz` (không có trên GitHub).')
    st.stop()
st.title('4. Giải thích từng quyết định')
st.markdown(
    'Mỗi quyết định có ba lời giải thích:\n'
    '1. **Vì sao**: lượng cỏ máy thấy trong ô so với hai ngưỡng.\n'
    '2. **Đổi gì thì quyết định đổi** (giải thích phản thực): bỏ những vùng cỏ 5 cm × 5 cm nào thì ô hết PHUN. Số vùng '
    'ghi ở đây **đã được kiểm chứng**: xóa đúng các vùng đó khỏi ảnh (thay bằng đất trống) rồi cho mô hình chấm lại '
    '(chạy trên Colab).\n'
    '3. **Dải phổ nào quan trọng**: Integrated Gradients (IG; Sundararajan và cộng sự, ICML 2017) so với ảnh đất trống.\n\n'
    'Có cho mô hình chính và thiết lập chính của báo cáo (ô 50 cm, τ = 1%, α = β = 10%).')

cells, steps = ui.explanation_cells()
ns = ui.no_spray_cells()
maps = ui.xai_maps()
ids, _, p_weed = ui.probs(C.MAIN_MODEL, 'test')
ids = list(ids)
masks = ui.masks('test')

spray = cells[cells['decision_rescored'] == C.SPRAY]
c1, c2, c3, c4 = st.columns(4)
c1.metric('Ô PHUN', len(spray), border=True)
c2.metric('Ô CHƯA CHẮC', int((cells['decision_rescored'] == C.DEFER).sum()), border=True)
c3.metric('Ô KHÔNG PHUN', len(ns), border=True)
c4.metric('Ô PHUN mong manh', int((spray['k_not_spray_model'] == 1).sum()), border=True,
          help='Ô PHUN mà chỉ cần bỏ một vùng cỏ 5 cm là hệ thống không còn quyết định PHUN (mô hình đã chấm lại). '
               'Người dùng nên được báo về những ô này.')

mode = st.segmented_control('Chọn ô', ['examples', 'pick'], default='examples', required=True,
                            format_func={'examples': 'Ba ví dụ tiêu biểu', 'pick': 'Tự chọn ô'}.get)
DEC_ORDER = (C.SPRAY, C.DEFER, C.NO_SPRAY)
if mode == 'examples':
    picks = CF.pick_examples(cells)
    i = st.radio('Ví dụ', range(len(picks)), horizontal=True,
                 format_func=lambda i: CF.SIMPLE_TITLES[picks[i][0]].format(i=i + 1))
    row, decision = picks[i][1], int(picks[i][1]['decision_rescored'])
    st.caption('Ba ví dụ được chọn theo quy tắc cố định (`scripts/make_counterfactual_figure.py`, hàm `pick_examples`), '
               'không chọn bằng mắt: ô PHUN sát ngưỡng nhiều cỏ nhất; ô CHƯA CHẮC nhiều cỏ nhất; ô nhiều cỏ ngày 6/6 có '
               'ước lượng nhanh thiếu từ 2 vùng trở lên so với mô hình (lấy trung vị).')
else:
    a, b, c = st.columns([2, 2, 5])
    decision = a.selectbox('Quyết định', DEC_ORDER, format_func=C.DECISION_VI.get)
    pool = ns if decision == C.NO_SPRAY else cells[cells['decision_rescored'] == decision]
    date = b.selectbox('Đợt bay', ['all', *sorted(pool['date'].unique())],
                       format_func=lambda d: 'tất cả' if d == 'all' else C.date_vi(d + '_'))
    pool = (pool if date == 'all' else pool[pool['date'] == date]).sort_values(['date', 'tile', 'cell'])

    def label(r):
        s = r['s'] if decision == C.NO_SPRAY else r['s_model']
        text = f'{r["tile"]} · ô {int(r["cell"]) + 1} · cỏ dự đoán {ui.pct(s, 2)}'
        if decision == C.SPRAY:
            text += f' · bỏ {int(r["k_not_spray_model"])} vùng là hết PHUN'
        return text

    j = c.selectbox(f'Ô ({len(pool)} ô)', range(len(pool)), format_func=lambda j: label(pool.iloc[j]))
    row = pool.iloc[j]

tile_id, cell_idx = row['tile'], int(row['cell'])
t = ids.index(tile_id)
r0, col0 = divmod(cell_idx, C.TILE // CELL)
p_cell = p_weed[t][r0 * CELL:(r0 + 1) * CELL, col0 * CELL:(col0 + 1) * CELL]
true3 = C.to_3class(masks[t])
colour = C.DECISION_COLOR[decision]
st.markdown(f'#### {ui.decision_badge(decision)} &nbsp; ô {cell_idx + 1} của ô ảnh {tile_id}, bay ngày '
            f'{C.date_vi(tile_id)}', unsafe_allow_html=True)

if decision == C.NO_SPRAY:
    no_thr, sp_thr = CF.thresholds_cm2(row)
    need = row['w'] >= C.PRIMARY['tau']
    st.markdown(
        f'- **Vì sao KHÔNG PHUN:** máy thấy khoảng {ui.num(row["s"] * CELL_CM2, 1)} cm² cỏ trong ô 50 cm × 50 cm '
        f'({ui.pct(row["s"], 2)} diện tích), dưới ngưỡng không phun {ui.num(no_thr, 1)} cm².\n'
        f'- **Đổi gì thì quyết định đổi:** thêm khoảng {ui.num(row["add_to_spray_cm2"], 1)} cm² cỏ dự đoán thì ô thành '
        'PHUN. Con số này tính theo quy tắc ngưỡng; không kiểm chứng bằng mô hình được vì không thể "thêm cỏ" vào ảnh.\n'
        f'- **Thực tế (theo nhãn):** cỏ chiếm {ui.pct(row["w"], 2)} diện tích ô → '
        + ('ô thật sự **cần phun**: đây là một ô bị bỏ sót.' if need else 'ô thực sự không cần phun.'))
    ui.figure('cell_panels', ui.rgb(tile_id), true3, p_weed[t], row, colour, 0, 0)
else:
    st_steps = steps[(steps['tile'] == tile_id) & (steps['cell'] == cell_idx)].sort_values('k')
    lines = CF.text_lines(None, row, st_steps, row['add_to_spray_cm2'])
    st.markdown('\n'.join(f'- {line}' for line in lines))
    k_no = int(row['k_no_spray_model'])
    k_solid = int(row['k_not_spray_model']) if decision == C.SPRAY else 0
    ui.figure('cell_panels', ui.rgb(tile_id), true3, p_weed[t], row, colour, k_solid, k_no)
    st.caption('Viền liền: các vùng cần bỏ để ô hết PHUN; viền đứt: các vùng bỏ thêm để ô thành KHÔNG PHUN. Các vùng '
               'được bỏ lần lượt, vùng nhiều cỏ dự đoán nhất trước (số đánh trên hình là thứ tự này). Số vùng là số mô '
               'hình đã kiểm chứng theo thứ tự đó, không phải số vùng ít nhất có thể.')
    try:
        CF.check_example(row, p_cell)
        st.caption(':material/check_circle: Bản đồ xác suất đã lưu cho đúng điểm số của lần chạy trên Colab, và thứ tự '
                   'vùng trên Colab trùng với thứ tự tính trên máy này.')
    except ValueError as e:
        st.error(f'Dữ liệu không nhất quán: {e}')
    ui.figure('removal_curve', row, st_steps, width='content')
    st.markdown('**Dải phổ nào đẩy lượng cỏ dự đoán lên** (IG so với ảnh đất trống)')
    left, right = st.columns([3, 2])
    with left:
        ui.band_chart([row[f'ig_{b}'] * CELL_CM2 for b in C.BANDS], 'đóng góp vào lượng cỏ dự đoán của ô (cm²)')
    with right:
        st.caption(f'Tổng các dải ≈ lượng cỏ dự đoán của ô ({ui.num(row["ig_sum"] * CELL_CM2, 1)} cm²), vì ảnh đất '
                   'trống có lượng cỏ dự đoán gần 0. IG trả lời "vì sao là cỏ mà không phải đất", không trả lời "vì sao '
                   'là cỏ mà không phải ngô".')

st.markdown('##### Bản đồ giải thích trên cả ô ảnh')
k = np.flatnonzero((maps['tile'] == tile_id) & (maps['cell'] == cell_idx))
if len(k):
    ui.figure('attribution_panels', ui.rgb(tile_id), maps['ig'][k[0]], maps['grad_cam'][k[0]], cell_idx, colour)
    st.caption('Cả hai bản đồ giải thích quyết định của ô có viền màu. IG tập trung trong ô đó; Grad-CAM gần như giống '
               'nhau cho mọi ô của cùng ô ảnh (thử chọn ô khác cùng ô ảnh để thấy), nên không chỉ ra được vùng riêng '
               'của từng ô. Trang 5 kiểm chứng điều này bằng số.')
ui.footer()
