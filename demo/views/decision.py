"""Spraying decisions per cell. The setting can be changed and every decision is recomputed with src.weeds.decision
from the saved probability maps; a setting that a run's decision_metrics.csv also holds is checked against it."""
import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import core as C
import figures as F
import ui

ui.need_data()
P = C.PRIMARY_SETTING
DEFAULTS = {'dec_model': C.MAIN_MODEL, 'dec_cell': P['cell'], 'dec_tau': P['tau'], 'dec_alpha': P['alpha'],
            'dec_beta': P['beta'], 'dec_scope': P['scope'], 'dec_cal': P['calibration']}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def reset():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


def side_cm(cell):
    return ui.num(cell * C.GSD_CM, 0) + ' cm'


def tau_text(v):
    return ui.pct(v, 1 if round(v * 1000) % 10 else 0)


TAUS = [0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
RATES = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
with st.sidebar:
    st.subheader('Thiết lập')
    key = ui.model_picker(key='dec_model')
    cell = st.segmented_control('Kích thước ô', [100, 150, 200, 300], key='dec_cell', required=True,
                                format_func=lambda c: f'{side_cm(c)} × {side_cm(c)}')
    tau = st.select_slider('Ngưỡng "cần phun" τ (cỏ chiếm từ … diện tích ô)', TAUS, key='dec_tau',
                           format_func=tau_text)
    alpha = st.select_slider('Giới hạn bỏ sót α', RATES, key='dec_alpha', format_func=lambda v: ui.pct(v, 0))
    beta = st.select_slider('Giới hạn phun nhầm β', RATES, key='dec_beta', format_func=lambda v: ui.pct(v, 0))
    scope = st.selectbox('Phạm vi (đợt bay)', list(C.SCOPES), key='dec_scope', format_func=C.SCOPE_VI.get)
    calibration = st.radio('Hiệu chỉnh ngưỡng trên', [C.CAL_TEST_2FOLD, C.CAL_VAL], key='dec_cal',
                           format_func=C.CALIBRATION_VI.get)
    st.button('Về thiết lập chính của báo cáo', on_click=reset, icon=':material/restart_alt:', width='stretch')

st.title('3. Quyết định phun cho từng ô')
st.markdown(
    'Ruộng được chia thành ô vuông. **Lượng cỏ dự đoán** của ô là trung bình xác suất cỏ của các điểm ảnh trong ô '
    '(ước lượng % diện tích ô là cỏ). Lượng này được so với **hai ngưỡng conformal**, tính trên *các vị trí khác* của '
    'ruộng sao cho bỏ sót không quá α và phun nhầm không quá β: '
    f'trên ngưỡng phun thì {ui.decision_badge(C.SPRAY)}, dưới ngưỡng không phun thì {ui.decision_badge(C.NO_SPRAY)}, '
    f'ở giữa thì {ui.decision_badge(C.DEFER)} (người ra xem, cần thì phun). Đổi thiết lập ở thanh bên trái: hệ thống '
    'tính lại ngay.', unsafe_allow_html=True)

res = ui.decide(key, cell, tau, alpha, beta, scope, calibration)
diff = None if res is None else C.saved_difference(res[1], ui.saved_metrics(key), cell, tau, alpha, beta, scope,
                                                     calibration)
primary = all(st.session_state[k] == v for k, v in DEFAULTS.items())
saved_file = f'results/{C.MODELS[key]["dir"]}/decision_metrics.csv'
if res is None:
    st.warning('Không hiệu chỉnh được ngưỡng với thiết lập này: phạm vi đã chọn chỉ có ô ảnh ở một phần vị trí (hoặc '
               'tập val không có ô nào của đợt bay này). Hãy chọn phạm vi khác.', icon=':material/warning:')
    ui.footer()
    st.stop()
if diff is not None and not diff < 1e-9:
    st.error(f'Số tính lại KHÔNG khớp `{saved_file}` (chênh {diff:.3g}): cần kiểm tra trước khi trình bày.')
elif diff is not None and primary:
    st.success(f'**Thiết lập chính của báo cáo.** Số dưới đây được tính lại và khớp `{saved_file}` (chênh lệch tối đa '
               f'{diff:.0e}).', icon=':material/verified:')
elif diff is not None:
    st.info(f'Thiết lập này có trong `{saved_file}`: số tính lại khớp file (chênh lệch tối đa {diff:.0e}). Đây không '
            'phải thiết lập chính của báo cáo.', icon=':material/check_circle:')
else:
    st.warning('**Đang khám phá:** thiết lập này không có trong file kết quả đã lưu. Số dưới đây tính trực tiếp từ bản '
               'đồ xác suất, không phải số trong báo cáo.', icon=':material/explore:')

cells, metrics, parts = res
m = metrics.set_index('policy')
cf = m.loc['conformal_3way']
n, n_need = int(cf['n_cells']), int(cf['n_need'])
st.caption(f'{C.MODELS[key]["label"]} · {n} ô {side_cm(cell)} × {side_cm(cell)} của tập kiểm tra · {n_need} ô cần phun '
           f'theo nhãn ({ui.pct(n_need / n)}) · {C.SCOPE_VI[scope]}')
c1, c2, c3, c4 = st.columns(4)
c1.metric('Giảm diện tích phun', ui.pct(cf['spray_reduction']), border=True,
          help=f'Mức tốt nhất có thể (biết trước đáp án): {ui.pct(m.loc["oracle", "spray_reduction"])}.')
c2.metric('Ô CHƯA CHẮC', f'{ui.pct(cf["deferred"])} · {int(cf["n_deferred"])} ô', border=True)
c3.metric(f'Bỏ sót (giới hạn {ui.pct(alpha, 0)})', f'{ui.pct(cf["missed_need_rate"])} · {int(cf["n_missed"])} ô',
          border=True)
c4.metric(f'Phun nhầm (giới hạn {ui.pct(beta, 0)})',
          f'{ui.pct(cf["false_spray_rate"])} · {int(cf["n_false_spray"])} ô', border=True)

st.subheader('So sánh các cách quyết định')


def flag(rate, limit):
    return ' ⚠️' if isinstance(rate, float) and math.isfinite(rate) and rate > limit + 1e-12 else ''


rows = []
for name in C.POLICIES:
    r = m.loc[name]
    rows.append({'Cách quyết định': C.POLICY_VI[name].replace('τ', tau_text(tau)),
                 'Giảm diện tích phun': ui.pct(r['spray_reduction']),
                 'Ô CHƯA CHẮC': f'{ui.pct(r["deferred"])} ({int(r["n_deferred"])} ô)',
                 'Bỏ sót': f'{ui.pct(r["missed_need_rate"])} ({int(r["n_missed"])}/{n_need} ô)'
                           + flag(r['missed_need_rate'], alpha),
                 'Phun nhầm': f'{ui.pct(r["false_spray_rate"])} ({int(r["n_false_spray"])}/{n - n_need} ô)'
                              + flag(r['false_spray_rate'], beta)})
st.dataframe(pd.DataFrame(rows), hide_index=True)
st.caption('⚠️: vượt giới hạn α hoặc β. "Giảm diện tích phun" tính cả ô CHƯA CHẮC là đã có người ra xem, cần thì phun '
           '(theo Melki và cộng sự, 2025). Ngưỡng cố định chỉ so lượng cỏ dự đoán với τ, không có bảo đảm nào về mức '
           'sai; phun cả ruộng và "biết trước đáp án" là hai mốc so sánh.')

with st.expander('Hai ngưỡng được tính ra sao, và giá trị hiện tại', expanded=False):
    area = C.cell_cm2(cell)
    thr = []
    for i, p in enumerate(parts):
        lo, hi = p['lam_lo'], p['lam_hi']
        no_thr, sp_thr = min(lo, hi), max(lo, hi)
        thr.append({'Phần': (f'Phần {i}: hiệu chỉnh trên các vị trí của phần {1 - i}' if len(parts) == 2
                             else 'Hiệu chỉnh trên tập val'),
                    'Ngưỡng không phun': '–' if not math.isfinite(no_thr) else
                    f'{ui.pct(no_thr, 2)} ({ui.num(no_thr * area, 1)} cm²)',
                    'Ngưỡng phun': '–' if not math.isfinite(sp_thr) else
                    f'{ui.pct(sp_thr, 2)} ({ui.num(sp_thr * area, 1)} cm²)',
                    'Ô hiệu chỉnh cần phun / không cần': f'{p["n_cal_need"]} / {p["n_cal_no_need"]}'})
    st.dataframe(pd.DataFrame(thr), hide_index=True)
    st.markdown(
        '- **Ngưỡng chặn bỏ sót** λ_lo: trong các ô *cần phun* dùng để hiệu chỉnh, chỉ khoảng α ô có lượng cỏ dự đoán '
        'thấp hơn λ_lo. Vì vậy ô mới dưới λ_lo mới được KHÔNG PHUN.\n'
        '- **Ngưỡng chặn phun nhầm** λ_hi: trong các ô *không cần phun*, chỉ khoảng β ô vượt λ_hi. Vì vậy ô mới trên '
        'λ_hi mới được PHUN.\n'
        '- Ô nằm giữa hai ngưỡng là CHƯA CHẮC. Khi λ_hi < λ_lo, ô nằm giữa bị loại cả "cần phun" lẫn "không cần phun", '
        'nên cũng là CHƯA CHẮC; bảng trên ghi ngưỡng nhỏ là "không phun", ngưỡng lớn là "phun".\n'
        '- **Hiệu chỉnh chéo theo vị trí:** 8 vị trí trên ruộng chia thành 2 phần; ngưỡng tính từ phần này dùng để quyết '
        'định phần kia, nên ô được quyết định không bao giờ nằm trong dữ liệu hiệu chỉnh. Không dùng tập val vì val '
        'đã dùng để chọn mô hình.\n'
        '- Bảo đảm chỉ gần đúng, vì các ô cạnh nhau thường giống nhau. Phương pháp: conformal theo lớp (Vovk và cộng '
        'sự), như Melki và cộng sự (2025) dùng cho trừ cỏ. "–": quá ít ô để tính ngưỡng đó, nên không loại trừ được '
        'khả năng nào.')

st.subheader('Bản đồ quyết định trên ruộng')
policy = st.segmented_control('Hiển thị quyết định của', ['conformal_3way', 'plug_in', 'oracle'],
                              default='conformal_3way', required=True, key='dec_policy',
                              format_func=lambda p: {'conformal_3way': 'Đề tài (conformal)',
                                                     'plug_in': 'Ngưỡng cố định τ',
                                                     'oracle': 'Đáp án (theo nhãn)'}[p])
loc, _ = ui.locations()
dates = sorted(cells['date'].unique())
with st.spinner('Đang đọc ảnh…'):
    panels = {}
    for tile_id, g in cells.groupby('tile', sort=False):
        panels[(C.tile_date(tile_id), loc[tile_id])] = (ui.thumbnail(tile_id), g[policy].to_numpy(),
                                                        g['need'].to_numpy())
fold_of_loc = {loc[t]: f for t, f in zip(cells['tile'], cells['fold'])}
ui.show(F.field_overview(panels, dates, max(loc.values()), cell,
                         lambda j: f'Vị trí {j}' + (f' · phần {fold_of_loc[j]}' if j in fold_of_loc else '')))
st.caption('Mỗi cột là một vị trí trên ruộng (1,5 m × 1,5 m), mỗi hàng là một đợt bay, nên đọc theo cột thấy cỏ ở '
           'cùng một chỗ thay đổi qua thời gian. Dấu X: quyết định sai so với nhãn (ô KHÔNG PHUN mà cần phun là bỏ sót; '
           'ô PHUN mà không cần là phun nhầm). Ô CHƯA CHẮC không tính là sai vì người sẽ ra xem.')

st.subheader('Xem chi tiết một ô ảnh')
tiles = list(dict.fromkeys(cells['tile']))
tile_id = st.selectbox('Ô ảnh', tiles, format_func=lambda t: f'{t} · vị trí {loc[t]} · bay ngày {C.date_vi(t)}')
g = cells[cells['tile'] == tile_id]
left, right = st.columns([3, 2])
with left:
    ui.show(F.tile_decisions(ui.rgb(tile_id), g, policy, cell, tau,
                             f'{tile_id}: "máy thấy" = lượng cỏ dự đoán, "nhãn thật" = cỏ theo nhãn'))
with right:
    view = pd.DataFrame({'Ô': g['cell'] + 1, 'Quyết định': [C.DECISION_VI[int(d)] for d in g[policy]],
                         'Cỏ dự đoán': [ui.pct(v, 2) for v in g['s']], 'Cỏ theo nhãn': [ui.pct(v, 2) for v in g['w']],
                         'Cần phun?': np.where(g['need'], 'có', 'không')})
    st.dataframe(view, hide_index=True)
    st.caption('Ô được đánh số từ trái sang phải, trên xuống dưới. Muốn biết vì sao một ô PHUN hay CHƯA CHẮC, xem '
               'trang 4 (giải thích có cho thiết lập chính và mô hình chính).')

st.subheader('Lượng cỏ dự đoán so với lượng cỏ thật')
plot = cells.assign(s_pct=100 * cells['s'], w_pct=100 * cells['w'],
                    decision=[C.DECISION_VI[int(d)] for d in cells['conformal_3way']], cell_no=cells['cell'] + 1,
                    outcome=np.select([(cells['conformal_3way'] == C.NO_SPRAY) & cells['need'],
                                       (cells['conformal_3way'] == C.SPRAY) & ~cells['need'],
                                       cells['conformal_3way'] == C.DEFER], ['bỏ sót', 'phun nhầm', 'người ra xem'],
                                      'đúng'))
order = [C.DECISION_VI[d] for d in (C.SPRAY, C.DEFER, C.NO_SPRAY)]
ticks = [0, 0.1, 0.3, 1, 3, 10, 30, 100]
scale = alt.Scale(type='symlog', constant=0.1, domain=[0, 100])
points = alt.Chart(plot).mark_circle(size=60, opacity=0.85, stroke='white', strokeWidth=0.8).encode(
    x=alt.X('s_pct:Q', title='lượng cỏ dự đoán của ô (% diện tích)', scale=scale, axis=alt.Axis(values=ticks)),
    y=alt.Y('w_pct:Q', title='cỏ theo nhãn (% diện tích)', scale=scale, axis=alt.Axis(values=ticks)),
    color=alt.Color('decision:N', title='Quyết định (đề tài)',
                    scale=alt.Scale(domain=order, range=[C.DECISION_COLOR[d] for d in (C.SPRAY, C.DEFER, C.NO_SPRAY)]),
                    legend=alt.Legend(orient='top')),
    tooltip=[alt.Tooltip('tile:N', title='ô ảnh'), alt.Tooltip('cell_no:Q', title='ô'),
             alt.Tooltip('decision:N', title='quyết định'), alt.Tooltip('outcome:N', title='kết quả'),
             alt.Tooltip('s_pct:Q', title='cỏ dự đoán (%)', format='.2f'),
             alt.Tooltip('w_pct:Q', title='cỏ theo nhãn (%)', format='.2f')])
tau_rule = alt.Chart(pd.DataFrame({'y': [100 * tau], 'label': ['τ: từ đây trở lên là cần phun']})).mark_rule(
    color='#52514e', strokeDash=[5, 3]).encode(y='y:Q', tooltip=alt.Tooltip('label:N', title=''))
lam = pd.DataFrame([{'x': 100 * v, 'label': f'{name} ({"phần " + str(i) if len(parts) == 2 else "val"}): '
                                            f'{ui.pct(v, 2)}'}
                    for i, p in enumerate(parts) for name, v in (('ngưỡng không phun', min(p['lam_lo'], p['lam_hi'])),
                                                                 ('ngưỡng phun', max(p['lam_lo'], p['lam_hi'])))
                    if math.isfinite(v)])
layers = [points, tau_rule]
if not lam.empty:
    layers.insert(0, alt.Chart(lam).mark_rule(color='#898781', strokeWidth=1).encode(
        x='x:Q', tooltip=alt.Tooltip('label:N', title='')))
st.altair_chart(alt.layer(*layers).properties(height=420), width='stretch')
st.caption('Mỗi chấm là một ô. Đường ngang đứt: ngưỡng "cần phun" τ (theo nhãn). Đường dọc mảnh: hai ngưỡng conformal '
           'của từng phần hiệu chỉnh. Chấm xanh (KHÔNG PHUN) nằm trên đường ngang là bỏ sót; chấm đỏ (PHUN) nằm dưới là '
           'phun nhầm. Trục chia giãn ở gần 0 để thấy rõ các ô ít cỏ. Đưa chuột lên chấm để xem chi tiết.')

st.subheader('Cùng thiết lập, khác mô hình phân vùng')
rows = []
for k in C.available_models():
    r = ui.decide(k, cell, tau, alpha, beta, scope, calibration)
    if r is None:
        continue
    mm = r[1].set_index('policy')
    a, b = mm.loc['conformal_3way'], mm.loc['plug_in']
    seg = C.read_json(C.model_dir(k) / 'segmentation_metrics.json')['test']['miou']
    rows.append({'Mô hình phân vùng': C.MODELS[k]['label'], 'mIoU test (%)': ui.num(seg, 2),
                 'Đề tài: giảm phun': ui.pct(a['spray_reduction']), 'Đề tài: chưa chắc': ui.pct(a['deferred']),
                 'Đề tài: bỏ sót': ui.pct(a['missed_need_rate']) + flag(a['missed_need_rate'], alpha),
                 'Đề tài: phun nhầm': ui.pct(a['false_spray_rate']) + flag(a['false_spray_rate'], beta),
                 'Ngưỡng cố định: bỏ sót': ui.pct(b['missed_need_rate']) + flag(b['missed_need_rate'], alpha),
                 'Ngưỡng cố định: phun nhầm': ui.pct(b['false_spray_rate']) + flag(b['false_spray_rate'], beta)})
st.dataframe(pd.DataFrame(rows), hide_index=True)
st.caption('Ý chính của tầng quyết định: ngưỡng cố định phụ thuộc vào mô hình. Mô hình nào đoán thiếu cỏ (ví dụ mô '
           'hình chỉ dùng ảnh RGB) thì ngưỡng cố định bỏ sót nhiều; ngưỡng conformal tự điều chỉnh theo mô hình để giữ '
           'mức sai trong giới hạn, đổi lại phải ra xem một số ô. mIoU cao hơn không nhất thiết cho quyết định tốt '
           'hơn.')
ui.footer()
