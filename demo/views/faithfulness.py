"""Are the explanations faithful? The deletion test against random controls (xai_summary.json), why Grad-CAM fails
at cell level (recomputed from xai_maps.npz), the spectral bands, and the model check of the counterfactuals."""
import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import core as C
import ui

ui.need_data()
st.title('5. Kiểm chứng lời giải thích có trung thực không')
st.markdown(
    'Một lời giải thích **trung thực** phải phản ánh đúng lý do mô hình quyết định, chứ không chỉ "trông hợp lý". '
    'Cách kiểm tra: **xóa** (thay bằng đất trống) những vùng 5 cm × 5 cm mà lời giải thích chỉ ra là quan trọng nhất, '
    'cho mô hình chấm lại, rồi so với xóa **cùng số vùng chọn ngẫu nhiên**. Lời giải thích trung thực thì xóa vùng của '
    'nó phải làm lượng cỏ dự đoán giảm nhiều hơn hẳn. Phép thử chạy trên Colab cho các ô PHUN và CHƯA CHẮC của mô hình '
    'chính. *"Trung thực" không có nghĩa là mô hình nhận đúng cỏ.*')

summary = ui.read_json('weeds_xai/xai_summary.json')
faith = summary['faithfulness']
METHODS = {'integrated_gradients': ('Theo IG', '#2a78d6', 'lời giải thích'),
           'predicted_weed': ('Theo xác suất cỏ của mô hình (mốc)', '#eb6834', 'lời giải thích'),
           'grad_cam': ('Theo Grad-CAM', '#1baf7a', 'lời giải thích'),
           'random_cell': ('Ngẫu nhiên trong ô', '#898781', 'đối chứng ngẫu nhiên'),
           'random_tile': ('Ngẫu nhiên trên cả ô ảnh', '#c3c2b7', 'đối chứng ngẫu nhiên')}
PATCHES_PER_TILE = (C.TILE // C.PATCH) ** 2
PATCHES_PER_CELL = (C.PRIMARY['cell_size'] // C.PATCH) ** 2
rows = []
for method, by_fraction in faith.items():
    for f, v in by_fraction.items():
        n = math.ceil(float(f) * PATCHES_PER_TILE)  # as src.weeds.explain.deletion_tile
        rows.append({'method': METHODS[method][0], 'kind': METHODS[method][2], 'fraction': float(f), 'patches': n,
                     'drop': 100 * v['mean_rel_drop'], 'flip': 100 * v['spray_flip_rate'],
                     'beats_cell': v.get('beats_random_cell'), 'cells': v['cells']})
df = pd.DataFrame(rows)
n_cells = int(df['cells'].max())

st.subheader('Phép xóa vùng')
domain = [v[0] for v in METHODS.values()]
colour = alt.Color('method:N', title=None, scale=alt.Scale(domain=domain, range=[v[1] for v in METHODS.values()]),
                   legend=alt.Legend(orient='top', columns=3))
dash = alt.StrokeDash('kind:N', title=None, scale=alt.Scale(domain=['lời giải thích', 'đối chứng ngẫu nhiên'],
                                                            range=[[1, 0], [5, 3]]), legend=None)


def lines(y, title):
    base = alt.Chart(df).encode(x=alt.X('patches:Q', title='số vùng 5 cm bị xóa', scale=alt.Scale(type='log'),
                                        axis=alt.Axis(values=sorted(df['patches'].unique()))),
                                y=alt.Y(f'{y}:Q', title=title, scale=alt.Scale(domain=[0, 100])), color=colour)
    hover = alt.selection_point(fields=['patches'], nearest=True, on='pointerover', empty=False)
    return (base.mark_line(strokeWidth=2).encode(strokeDash=dash)
            + base.mark_point(size=70, filled=True).encode(
                opacity=alt.condition(hover, alt.value(1), alt.value(0.35)),
                tooltip=[alt.Tooltip('method:N', title='chọn vùng'), alt.Tooltip('patches:Q', title='số vùng'),
                         alt.Tooltip(f'{y}:Q', title=title, format='.1f')]).add_params(hover)).properties(height=330)


a, b = st.columns(2)
with a:
    st.markdown('**Lượng cỏ dự đoán của ô giảm bao nhiêu (%)**')
    st.altair_chart(lines('drop', 'mức giảm trung bình (%)'), width='stretch')
with b:
    st.markdown('**Bao nhiêu ô PHUN bị đổi quyết định (%)**')
    st.altair_chart(lines('flip', 'ô PHUN đổi quyết định (%)'), width='stretch')

one = df[np.isclose(df['fraction'], 0.01)]
if not one.empty:
    n9 = int(one['patches'].iloc[0])
    st.markdown(f'**Khi xóa {n9} vùng** ({ui.num(100 * n9 / PATCHES_PER_CELL, 0)}% diện tích một ô):')
    st.dataframe(pd.DataFrame({
        'Cách chọn vùng để xóa': one['method'], 'Lượng cỏ dự đoán giảm': [ui.num(v, 1) + '%' for v in one['drop']],
        'Ô PHUN bị đổi quyết định': [ui.num(v, 0) + '%' for v in one['flip']],
        f'Thắng "ngẫu nhiên trong ô" (trong {n_cells} ô)': [
            '–' if b is None or (isinstance(b, float) and math.isnan(b)) else ui.pct(b, 0) for b in one['beats_cell']]}),
        hide_index=True)
ig = df[df['method'] == METHODS['integrated_gradients'][0]].set_index('fraction')
gc = df[df['method'] == METHODS['grad_cam'][0]].set_index('fraction')
small = [f for f in ig.index if f <= 0.05]
st.markdown(
    f'- **IG trung thực:** ở mọi mức xóa từ {int(ig.loc[min(small), "patches"])} đến '
    f'{int(ig.loc[max(small), "patches"])} vùng, xóa theo IG làm lượng cỏ dự đoán giảm nhiều hơn xóa ngẫu nhiên ở '
    f'{ui.pct(min(ig.loc[f, "beats_cell"] for f in small), 0)} số ô.\n'
    f'- **Grad-CAM không dùng được để giải thích từng ô:** so từng ô, nó chỉ hơn xóa ngẫu nhiên trong ô ở '
    f'{ui.pct(gc.loc[0.01, "beats_cell"], 0) if 0.01 in gc.index else "–"} số ô (xóa {n9 if not one.empty else "–"} '
    'vùng), gần như tung đồng xu.\n'
    '- **IG không hơn mốc "xác suất cỏ của mô hình"**: điều này dễ hiểu, vì lượng cỏ dự đoán chính là trung bình xác '
    'suất cỏ trong ô. IG có thêm: chia đóng góp theo dải phổ, và tổng đóng góp bằng đúng lượng cỏ dự đoán so với đất '
    'trống.\n'
    f'- Ở mức xóa lớn nhất ({int(df["patches"].max())} vùng), "ngẫu nhiên trong ô" gần như xóa cả ô nên so sánh không '
    'còn ý nghĩa.')
st.caption('Nguồn: `results/weeds_xai/xai_summary.json` (Colab, mô hình tác giả 5 dải, IG 32 bước, 10 lần chọn ngẫu '
           'nhiên). Tỉ lệ đổi quyết định tính trên các ô PHUN.')

st.subheader('Vì sao Grad-CAM thất bại ở mức ô')
maps = ui.xai_maps()
xc = ui.xai_cells()
dec = xc.set_index(['tile', 'cell']).loc[pd.MultiIndex.from_arrays([maps['tile'], maps['cell']]), 'decision'].to_numpy()
share_ig = C.in_cell_share(maps['ig'], maps['cell'])
share_gc = C.in_cell_share(maps['grad_cam'], maps['cell'])
corr_ig = C.within_tile_correlation(maps['ig'], maps['tile'])
corr_gc = C.within_tile_correlation(maps['grad_cam'], maps['tile'])
c1, c2, c3, c4 = st.columns(4)
c1.metric('IG: phần nằm trong ô (ô PHUN)', ui.pct(np.nanmedian(share_ig[dec == C.SPRAY]), 0), border=True)
c2.metric('IG: phần nằm trong ô (ô CHƯA CHẮC)', ui.pct(np.nanmedian(share_ig[dec == C.DEFER]), 0), border=True)
c3.metric('Grad-CAM: phần nằm trong ô', ui.pct(np.nanmedian(share_gc[dec != C.NO_SPRAY]), 0), border=True,
          help='Một ô chiếm 1/9 ô ảnh (11%): bản đồ rải đều khắp ô ảnh cũng cho khoảng chừng ấy.')
c4.metric('Tương quan bản đồ giữa các ô cùng ô ảnh', f'IG {ui.num(corr_ig.median(), 3)} · Grad-CAM '
          f'{ui.num(corr_gc.median(), 3)}', border=True,
          help='1 nghĩa là mọi ô của cùng ô ảnh có bản đồ giống hệt nhau.')
st.caption(f'Tính lại ngay từ `results/weeds_xai/xai_maps.npz` (trung vị; phần dương của bản đồ; tương quan trung bình '
           f'theo từng ô ảnh, {len(corr_gc)} ô ảnh). Grad-CAM gộp gradient thành một trọng số cho mỗi kênh trên cả bản '
           'đồ đặc trưng, nên nó chỉ ra mọi chỗ "giống cỏ" trên ô ảnh, không riêng ô đang quyết định. Đây là bản chất '
           'của phương pháp, không phải lỗi code.')

st.subheader('Dải phổ nào quan trọng (theo IG)')
shares, top = C.band_shares(xc)
band = pd.DataFrame({'band': list(C.BANDS), 'median': [100 * shares[b].median() for b in C.BANDS],
                     'top': [int((top == b).sum()) for b in C.BANDS]})
band['text'] = [('+' if v >= 0 else '−') + ui.num(abs(v), 0) + '%' for v in band['median']]
a, b = st.columns([3, 2])
with a:
    base = alt.Chart(band).encode(y=alt.Y('band:N', sort=list(C.BANDS), title=None),
                                  x=alt.X('median:Q', title='tỉ lệ đóng góp, trung vị trên các ô (%)'))
    st.altair_chart((base.mark_bar(color='#2a78d6', cornerRadiusEnd=4, size=22).encode(
        tooltip=[alt.Tooltip('band:N', title='dải'), alt.Tooltip('median:Q', title='trung vị (%)', format='.1f'),
                 alt.Tooltip('top:Q', title='số ô đứng đầu')])
        + base.mark_text(align='left', dx=4, color='#0b0b0b').encode(text='text:N')).properties(height=230),
        width='stretch')
with b:
    st.dataframe(pd.DataFrame({'Dải phổ': band['band'], 'Số ô dải này đóng góp nhiều nhất': band['top']}),
                 hide_index=True)
lead = band.sort_values('top', ascending=False).iloc[0]
negative = [b for b, v in zip(band['band'], band['median']) if v < 0]
st.caption(f'{len(top)} ô PHUN và CHƯA CHẮC; nguồn `results/weeds_xai/xai_cells.csv`. Dải **{lead["band"]}** đóng góp '
           f'nhiều nhất ở {lead["top"]}/{len(top)} ô (RE: rìa đỏ, NIR: cận hồng ngoại).'
           + (f' Dải {", ".join(negative)} có đóng góp âm (trung vị): chúng kéo lượng cỏ dự đoán xuống.' if negative
              else ''))

st.subheader('Kiểm chứng giải thích phản thực bằng mô hình')
cfs = ui.read_json('weeds_cfcheck/cfcheck_summary.json')
cc = pd.read_csv(C.RESULTS / 'weeds_cfcheck/cfcheck_cells.csv')
k = cfs['k_not_spray']
hist = pd.DataFrame({'gap': [int(g) for g in k['model_minus_estimate']],
                     'cells': list(k['model_minus_estimate'].values())}).sort_values('gap')
within1 = int(hist.loc[hist['gap'].abs() <= 1, 'cells'].sum())
sp = cc[cc['decision'] == C.SPRAY]
de = cc[cc['decision'] == C.DEFER]
st.markdown(
    f'"Bỏ k vùng này thì ô hết PHUN" có hai cách tính: **ước lượng nhanh** chỉ trừ phần xác suất cỏ của các vùng bị bỏ '
    f'(không chạy lại mô hình), và **mô hình chấm lại** sau khi xóa thật các vùng đó khỏi ảnh (Colab, '
    f'{ui.num(cfs["seconds_per_cell"], 2)} giây mỗi ô trên GPU T4). Trên {k["compared"]} ô PHUN:')
a, b = st.columns([3, 2])
with a:
    hist['label'] = hist['gap'].map(lambda g: f'{g:+d}' if g else '0')
    st.altair_chart(alt.Chart(hist).mark_bar(color='#2a78d6', cornerRadiusEnd=4).encode(
        x=alt.X('label:N', sort=[f'{g:+d}' if g else '0' for g in hist['gap']],
                title='k mô hình − k ước lượng (số vùng)', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('cells:Q', title='số ô PHUN'),
        tooltip=[alt.Tooltip('label:N', title='chênh (vùng)'), alt.Tooltip('cells:Q', title='số ô')]
    ).properties(height=260), width='stretch')
with b:
    st.metric('Ước lượng đúng chính xác', f'{k["exact"]}/{k["compared"]} ô', border=True)
    st.metric('Lệch không quá 1 vùng', f'{within1}/{k["compared"]} ô ({ui.pct(within1 / k["compared"], 0)})', border=True)
st.markdown(
    f'- Ước lượng nhanh **thường hơi lạc quan** (cột dương: mô hình cần bỏ nhiều vùng hơn), vì xóa một vùng cũng làm '
    'mô hình nhìn khác các vùng xung quanh. Vì vậy lời giải thích đưa vào báo cáo dùng **số đã được mô hình kiểm chứng**.\n'
    f'- **Quyết định mong manh là có thật:** mô hình xác nhận {int((sp["k_not_spray_model"] <= 1).sum())} ô PHUN hết '
    f'phun chỉ khi bỏ 1 vùng (ước lượng: {int((sp["k_not_spray_estimate"] <= 1).sum())} ô); '
    f'{int((sp["k_not_spray_model"] <= 2).sum())} ô với tối đa 2 vùng (ước lượng '
    f'{int((sp["k_not_spray_estimate"] <= 2).sum())}).\n'
    f'- **{len(de)} ô CHƯA CHẮC:** bỏ ' + ', '.join(
        f'{kk} vùng ({n} ô)' for kk, n in de['k_no_spray_model'].astype(int).value_counts().sort_index().items())
    + ' là thành KHÔNG PHUN, nên người kiểm tra biết ngay nên xem chỗ nào trước. Đó là chỗ xem trước, không thay cho '
      'việc xem cả ô, vì cỏ mà mô hình bỏ sót có thể nằm ở chỗ khác.')
st.caption('Nguồn: `results/weeds_cfcheck/cfcheck_summary.json`, `cfcheck_cells.csv` (Colab ngày 10/11/2026).')
ui.footer()
