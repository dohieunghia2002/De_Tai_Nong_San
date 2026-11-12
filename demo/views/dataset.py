"""The WeedsGalore dataset: what it is, how it is split, and any tile in true colour, false colour and labels."""
import numpy as np
import pandas as pd
import streamlit as st

import core as C
import figures as F
import ui

ui.need_data()
st.title('1. Dữ liệu ảnh drone: WeedsGalore')
st.markdown('Bộ dữ liệu công khai, công bố tại hội nghị **IEEE/CVF WACV 2025** (Celikkan và cộng sự), giấy phép '
            'CC BY 4.0. Tải tại <https://doidata.gfz.de/weedsgalore_e_celikkan_2024/>.')

left, right = st.columns([3, 2])
with left:
    st.dataframe(pd.DataFrame([
        ('Nơi chụp', 'Ruộng ngô ở Potsdam (Đức), năm 2023'),
        ('Cách chụp', 'Drone DJI P4 Multispectral bay ở độ cao 5 m; mỗi điểm ảnh ứng với 2,5 mm ngoài ruộng'),
        ('Ảnh', 'Ô ảnh 600 × 600 điểm ảnh (1,5 m × 1,5 m), 5 dải phổ: đỏ (R), lục (G), lam (B), cận hồng ngoại '
                '(NIR), rìa đỏ (RE); ảnh 16 bit'),
        ('Thời gian', '4 đợt bay: 25/5, 30/5, 6/6, 15/6. Ruộng được phun thuốc trừ cỏ ngày 12/6'),
        ('Nhãn', 'Từng điểm ảnh: nền đất, ngô, và 4 lớp cỏ (dền, cỏ lồng vực, cúc dại, cỏ khác). Đề tài gộp các '
                 'loài cỏ thành một lớp "cỏ dại", như bài báo gốc'),
        ('Nhãn "cần phun"', 'Không có sẵn: suy ra từ nhãn, một ô cần phun nếu cỏ chiếm từ τ diện tích ô trở lên '
                            '(τ = 1%, chỉ minh họa)'),
    ], columns=['', 'Mô tả']), hide_index=True)
with right:
    ids = ui.dataset_ids()
    split_vi = {'train': 'Huấn luyện (train)', 'val': 'Chọn mô hình (val)', 'test': 'Kiểm tra (test)'}
    counts = pd.DataFrame({split_vi[s]: pd.Series([C.tile_date(t) for t in ids[s]]).value_counts()
                           for s in C.SPLITS}).reindex(list(C.DATES)).fillna(0).astype(int)
    counts.index = [C.date_vi(d + '_') for d in counts.index]
    counts.loc['Tổng'] = counts.sum()
    st.dataframe(counts)
    st.caption('Số ô ảnh theo đợt bay và tập, đếm từ file chia tập của tác giả (giữ nguyên, không chia lại). Tập '
               'được chia theo vị trí trên ruộng.')

st.subheader('Xem từng ô ảnh')
date = st.segmented_control('Đợt bay', list(C.DATES), default=C.DATES[0], format_func=lambda d: C.date_vi(d + '_'),
                            required=True)
split_of = {t: s for s in C.SPLITS for t in ids[s]}
tiles = sorted(t for t in split_of if C.tile_date(t) == date)
tile_id = st.selectbox('Ô ảnh', tiles, format_func=lambda t: f'{t} · {split_vi[split_of[t]]}')
x, mask = ui.tile(tile_id)
lo, hi = map(np.asarray, ui.band_limits())
ui.show(F.tile_views(x, mask, lo, hi))

shares = pd.DataFrame({'Lớp': [name for name, _ in C.CLASSES],
                       'Diện tích trong ô ảnh': [ui.pct((mask == k).mean(), 2) for k in range(len(C.CLASSES))]})
c1, c2 = st.columns([2, 3])
c1.dataframe(shares, hide_index=True)
c2.caption('Ảnh gốc là 5 dải phổ riêng, 16 bit. Ở đây các dải được ghép thành ảnh màu và kéo giãn độ sáng (giữa '
           'phân vị 1% và 99% của toàn bộ dữ liệu, từng dải) chỉ để xem. Ảnh màu giả ghép NIR, R, G: thực vật phản '
           'xạ mạnh cận hồng ngoại nên hiện màu đỏ.')

sheet = C.IMAGES / 'anh_xem_truoc' / f'0_tong_quan_{date}.jpg'
if sheet.exists():
    with st.expander(f'Toàn bộ ô ảnh của đợt bay {C.date_vi(date + "_")} (ảnh thu nhỏ)'):
        st.image(str(sheet), width='stretch')
ui.footer()
