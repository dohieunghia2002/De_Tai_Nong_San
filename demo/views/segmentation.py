"""Segmentation: the authors' models reproduce the paper, per-class IoU, any test tile, and the project's own
training runs."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import core as C
import figures as F
import ui

ui.need_data()
st.title('2. Tìm cỏ trên ảnh (phân vùng)')
st.markdown('Mạng học sâu **DeepLabv3+** gán cho mỗi điểm ảnh một trong ba lớp: nền đất, ngô, cỏ dại. Đầu ra dùng cho '
            'bước sau là **xác suất cỏ** của từng điểm ảnh. Mô hình chính của đề tài là mô hình **tác giả bộ dữ liệu '
            'đã công bố**; các lần đề tài tự huấn luyện dùng để kiểm tra khả năng tái hiện. mIoU là trung bình IoU '
            '(độ trùng giữa vùng dự đoán và vùng thật) của các lớp, tính bằng %.')

st.subheader('Đối sánh với bài báo gốc')
labels = {'dlv3p_msi_3': '5 dải phổ, 3 lớp (đất / ngô / cỏ)', 'dlv3p_rgb_3': 'Ảnh màu RGB, 3 lớp',
          'dlv3p_msi_6': '5 dải phổ, 6 lớp (tách 4 loài cỏ)', 'dlv3p_rgb_6': 'Ảnh màu RGB, 6 lớp'}
p = C.authors_vs_paper()
st.dataframe(pd.DataFrame({
    'Mô hình của tác giả': [labels.get(k, k) for k in p['model']],
    'mIoU đề tài đánh giá (%)': [ui.num(v, 2) for v in p['ours']],
    'mIoU bài báo (%)': [ui.num(v, 2) for v in p['paper']],
    'Chênh lệch (điểm)': [ui.num(v, 3) for v in p['diff']]}), hide_index=True)
st.caption('Chạy các mô hình tác giả công bố bằng code đánh giá của đề tài (fp32, như code của tác giả) ra đúng số của '
           'bài báo, nên code đánh giá của đề tài là đúng. Nguồn: `results/weeds_authors/authors_vs_paper.json`.')

st.subheader('IoU từng lớp trên tập kiểm tra')
seg = C.segmentation_table()
st.dataframe(pd.DataFrame({
    'Mô hình': seg['model'], 'mIoU (%)': [ui.num(v, 2) for v in seg['miou']],
    'Nền đất': [ui.num(v, 2) for v in seg['background']], 'Ngô': [ui.num(v, 2) for v in seg['crop']],
    'Cỏ dại': [ui.num(v, 2) for v in seg['weed']]}), hide_index=True)
st.caption('Nguồn: `segmentation_metrics.json` của từng mô hình (khóa "test"; mô hình tác giả đánh giá ở fp16, chênh '
           'fp32 dưới 0,002 điểm). Lớp ngô khó nhất vì ngô non nhỏ và giống cỏ.')

st.subheader('Xem kết quả trên một ô ảnh kiểm tra')
c1, c2 = st.columns(2)
with c1:
    key = ui.model_picker(key='seg_model')
ids, p_crop, p_weed = ui.probs(key, 'test')
with c2:
    tile_id = st.selectbox('Ô ảnh kiểm tra', list(ids), format_func=lambda t: f'{t} · bay ngày {C.date_vi(t)}')
i = list(ids).index(tile_id)
true3 = C.to_3class(ui.masks('test')[i])
pred3 = C.predicted_classes(p_crop[i], p_weed[i])
ui.show(F.segmentation_views(ui.rgb(tile_id), true3, pred3, p_weed[i]))
iou = C.iou_per_class(pred3, true3)
st.caption(f'IoU trên riêng ô ảnh này, tính lại từ bản đồ xác suất đã lưu: nền đất {ui.iou_text(iou[0])}, ngô '
           f'{ui.iou_text(iou[1])}, cỏ dại {ui.iou_text(iou[2])}. Một ô ảnh ít ngô hoặc ít cỏ có thể có IoU rất thấp '
           'dù sai rất ít điểm ảnh; số đáng tin là số trên cả tập kiểm tra ở bảng trên.')

st.subheader('Các lần đề tài tự huấn luyện (kiểm tra khả năng tái hiện)')
runs = C.training_runs()
schedule_vi = {'100 epochs fixed': '100 epoch cố định', '100 epochs, cosine lr': '100 epoch, lr giảm dần (cosine)',
               'early stopping (30)': 'dừng sớm (30 epoch không cải thiện)'}
selection_vi = {'best val': 'val cao nhất', 'last epoch': 'epoch cuối'}
st.dataframe(pd.DataFrame({
    'Lần chạy': runs['run'], 'Lịch huấn luyện': runs['schedule'].map(schedule_vi).fillna(runs['schedule']),
    'Batch': runs['batch'].astype(int), 'Seed': runs['seed'].astype(int),
    'Chọn mô hình theo': runs['selection'].map(selection_vi), 'Epoch': runs['epoch'].astype(int),
    'mIoU val (%)': [ui.num(v, 2) for v in runs['val_miou']], 'mIoU test (%)': [ui.num(v, 2) for v in runs['test_miou']],
    'IoU ngô': [ui.num(v, 2) for v in runs['test_crop']], 'IoU cỏ': [ui.num(v, 2) for v in runs['test_weed']],
    'Giảm phun': [ui.pct(v) for v in runs['spray_reduction']], 'Chưa chắc': [ui.pct(v) for v in runs['deferred']],
    'Bỏ sót': [ui.pct(v) for v in runs['missed_need_rate']], 'Phun nhầm': [ui.pct(v) for v in runs['false_spray_rate']],
}), hide_index=True)
own = runs[(runs['batch'] == 8) & (runs['selection'] == 'best val')]['test_miou']
st.caption(f'{len(own)} lần batch 8, chọn theo val: mIoU test {ui.num(own.min(), 2)}–{ui.num(own.max(), 2)}%, trung '
           f'bình {ui.num(own.mean(), 2)}%; bài báo đạt {ui.num(p.set_index("model").loc["dlv3p_msi_3", "paper"], 2)}%. '
           'Khác biệt giữa các lần chủ yếu là dao động ngẫu nhiên của quá trình huấn luyện. Bốn cột cuối là tầng quyết '
           'định (thiết lập chính) chạy trên mô hình của từng lần. Nguồn: `results/weeds_runs/runs_summary.csv`, '
           '`results/weeds_pilot*/`.')

names = {'weeds_pilot': 'lần 1 (batch 2)', 'weeds_pilot_b8': 'lần 2 (batch 8, dừng sớm, seed 0)'}
options = [r for r in ['weeds_pilot', 'weeds_pilot_b8', *dict.fromkeys(runs['run'])] if C.training_history(r) is not None]
chosen = st.multiselect('Đường huấn luyện: mIoU trên tập val theo epoch (chọn tối đa 4 lần chạy)', options,
                        default=[r for r in ('weeds_pilot', 'weeds_pilot_b8') if r in options], max_selections=4,
                        format_func=lambda r: names.get(r, r))
if chosen:
    hist = pd.concat([C.training_history(r)[['epoch', 'val_miou']].assign(run=names.get(r, r)) for r in chosen])
    domain = [names.get(r, r) for r in chosen]
    palette = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'][:len(domain)]  # reference categorical slots 1-4
    base = alt.Chart(hist).encode(
        x=alt.X('epoch:Q', title='epoch'),
        y=alt.Y('val_miou:Q', title='mIoU tập val (%)', scale=alt.Scale(zero=False)),
        color=alt.Color('run:N', title=None, scale=alt.Scale(domain=domain, range=palette),
                        legend=alt.Legend(orient='top')))
    hover = alt.selection_point(fields=['epoch'], nearest=True, on='pointerover', empty=False)
    chart = (base.mark_line(strokeWidth=2)
             + base.mark_point(size=70, filled=True).encode(
                 opacity=alt.condition(hover, alt.value(1), alt.value(0)),
                 tooltip=[alt.Tooltip('run:N', title='lần chạy'), alt.Tooltip('epoch:Q'),
                          alt.Tooltip('val_miou:Q', title='mIoU val (%)', format='.2f')]).add_params(hover))
    st.altair_chart(chart.properties(height=320), width='stretch')
    st.caption('Lần 1 (batch 2): mIoU val dao động mạnh giữa các epoch và mô hình chọn được có mIoU test thấp; lần 2 '
               '(batch 8, mọi thiết lập khác giữ nguyên) ổn định hơn nhiều. Nguồn: `history.json` của từng lần chạy.')
ui.footer()
