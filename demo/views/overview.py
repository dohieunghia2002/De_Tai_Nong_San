"""Overview: what the system does, the headline numbers of the primary setting (recomputed and checked against
go_no_go.json), the pilot criteria, progress and limits."""
import math

import streamlit as st

import core as C
import ui

ui.need_data()
st.title('Hệ thống hỗ trợ ra quyết định phun thuốc trừ cỏ từ ảnh drone')
st.markdown('Drone chụp ruộng ngô. Máy **tìm cỏ trên ảnh**, rồi **quyết định cho từng ô 50 cm × 50 cm**: '
            f'{ui.decision_badge(C.SPRAY)} &nbsp; {ui.decision_badge(C.NO_SPRAY)} &nbsp; hay '
            f'{ui.decision_badge(C.DEFER)} (cần người ra xem), và **giải thích vì sao** quyết định như vậy.',
            unsafe_allow_html=True)

main = C.MAIN_MODEL
if main not in C.available_models():
    st.error('Thiếu bản đồ xác suất của mô hình chính (`results/weeds_authors/dlv3p_msi_3/probs_*.npz`).')
    st.stop()
s = C.PRIMARY_SETTING
cells, metrics, parts = ui.decide(main, s['cell'], s['tau'], s['alpha'], s['beta'], s['scope'], s['calibration'])
m = metrics.set_index('policy')
cf, oracle = m.loc['conformal_3way'], m.loc['oracle']
n, n_need = int(cf['n_cells']), int(cf['n_need'])
go = ui.read_json(f'{C.MODELS[main]["dir"]}/go_no_go.json')

st.subheader('Kết quả chính hiện tại')
st.caption(f'Mô hình phân vùng công bố của tác giả (ảnh 5 dải phổ); {n} ô 50 cm × 50 cm của tập kiểm tra, thuộc 3 đợt '
           f'bay trước ngày phun thuốc; {n_need} ô thật sự cần phun theo nhãn (cỏ chiếm từ '
           f'{ui.pct(s["tau"], 0)} diện tích ô). Giới hạn đặt ra: bỏ sót và phun nhầm không quá '
           f'{ui.pct(s["alpha"], 0)}.')
c1, c2, c3, c4 = st.columns(4)
c1.metric('Giảm diện tích phun', ui.pct(cf['spray_reduction']), border=True,
          help=f'So với phun cả ruộng. Ô CHƯA CHẮC được tính là đã có người ra xem, cần thì phun. Mức tốt nhất có '
               f'thể (biết trước đáp án) là {ui.pct(oracle["spray_reduction"])}.')
c2.metric('Ô CHƯA CHẮC (người ra xem)', f'{ui.pct(cf["deferred"])} · {int(cf["n_deferred"])} ô', border=True)
c3.metric('Bỏ sót', f'{ui.pct(cf["missed_need_rate"])} · {int(cf["n_missed"])} ô', border=True,
          help=f'Trong {n_need} ô cần phun, số ô bị quyết định KHÔNG PHUN.')
c4.metric('Phun nhầm', f'{ui.pct(cf["false_spray_rate"])} · {int(cf["n_false_spray"])} ô', border=True,
          help=f'Trong {n - n_need} ô không cần phun, số ô bị quyết định PHUN.')
same = all(math.isclose(a, b, abs_tol=1e-12) for a, b in (
    (cf['spray_reduction'], go['checks']['C2_spray_reduction']['value']),
    (cf['deferred'], go['checks']['C3_deferred']['value']),
    (cf['missed_need_rate'], go['info']['missed_need_rate']),
    (cf['false_spray_rate'], go['info']['false_spray_rate'])))
if same:
    st.success(f'Bốn con số trên được tính lại ngay khi mở trang, từ bản đồ xác suất đã lưu, và **khớp** file '
               f'`results/{C.MODELS[main]["dir"]}/go_no_go.json`.', icon=':material/check_circle:')
else:
    st.error('Số tính lại KHÔNG khớp `go_no_go.json`: cần kiểm tra trước khi trình bày.')

st.subheader('Hệ thống gồm 3 bước')
a, b, c = st.columns(3, border=True)
with a:
    st.markdown('**Bước 1. Tìm cỏ trên ảnh**\n\nMạng học sâu DeepLabv3+ (mô hình tác giả bộ dữ liệu công bố) cho biết '
                'xác suất mỗi điểm ảnh là cỏ, từ ảnh 5 dải phổ: đỏ, lục, lam, cận hồng ngoại (NIR), rìa đỏ (RE).')
    st.page_link('views/segmentation.py', label='Xem trang 2. Tìm cỏ trên ảnh', icon=':material/arrow_forward:')
with b:
    st.markdown('**Bước 2. Quyết định cho từng ô**\n\nLượng cỏ dự đoán của ô được so với **hai ngưỡng** tính bằng '
                'phương pháp *conformal* trên các vị trí khác của ruộng: trên ngưỡng phun thì PHUN, dưới ngưỡng không '
                'phun thì KHÔNG PHUN, ở giữa thì CHƯA CHẮC.')
    st.page_link('views/decision.py', label='Xem trang 3. Quyết định phun', icon=':material/arrow_forward:')
with c:
    st.markdown('**Bước 3. Giải thích và kiểm chứng**\n\n"Bỏ những vùng cỏ nào thì ô hết PHUN" (giải thích phản '
                'thực), dải phổ nào quan trọng (Integrated Gradients), và kiểm tra lời giải thích có trung thực '
                'không bằng cách xóa vùng rồi cho mô hình chấm lại.')
    st.page_link('views/explanation.py', label='Xem trang 4. Giải thích từng ô', icon=':material/arrow_forward:')

figure = C.IMAGES / 'so_do_pipeline_tac_gia.png'
if figure.exists():
    with st.expander('Sơ đồ xử lý trên một ô ảnh thật của tập kiểm tra'):
        st.image(str(figure), width='stretch')

st.subheader('Tiêu chí đề ra trước khi chạy thử')
names = {'C1_test_miou': 'C1. Tìm cỏ: mIoU trên tập kiểm tra (%)', 'C2_spray_reduction': 'C2. Giảm diện tích phun',
         'C3_deferred': 'C3. Tỉ lệ ô CHƯA CHẮC'}
rows = []
for k, ch in go['checks'].items():
    v = ch['value']
    rows.append({'Tiêu chí': names.get(k, k), 'Kết quả': ui.num(v, 2) if k == 'C1_test_miou' else ui.pct(v),
                 'Yêu cầu': ch['required'].replace('.', ','), 'Đạt': '✅ đạt' if ch['pass'] else '❌ chưa đạt'})
st.dataframe(rows, hide_index=True)
st.caption(f'Kết luận: **{go["verdict"]}** (nguồn: `results/{C.MODELS[main]["dir"]}/go_no_go.json`). C1 yêu cầu '
           'mIoU không thấp hơn số của bài báo quá 5 điểm.')

st.subheader('Tiến độ')
done, todo = st.columns(2, border=True)
paper = C.authors_vs_paper().set_index('model')
runs = C.training_runs()
own = runs[(runs['batch'] == 8) & (runs['selection'] == 'best val')]['test_miou']
with done:
    st.markdown('**Đã làm được**')
    st.markdown(
        f'- Chạy mô hình của tác giả bằng code của đề tài ra **đúng số của bài báo**: mIoU '
        f'{ui.num(paper.loc["dlv3p_msi_3", "ours"], 2)}% (bài báo {ui.num(paper.loc["dlv3p_msi_3", "paper"], 2)}%) '
        f'với ảnh 5 dải, {ui.num(paper.loc["dlv3p_rgb_3", "ours"], 2)}% '
        f'({ui.num(paper.loc["dlv3p_rgb_3", "paper"], 2)}%) với ảnh RGB.\n'
        f'- Tự huấn luyện {len(own)} lần (batch 8): mIoU {ui.num(own.min(), 2)}–{ui.num(own.max(), 2)}%, trung bình '
        f'{ui.num(own.mean(), 2)}%. Số của bài báo nằm trong khoảng này.\n'
        '- Tầng quyết định PHUN / KHÔNG PHUN / CHƯA CHẮC với ngưỡng conformal, thử trên mô hình của tác giả và mọi '
        'mô hình tự huấn luyện.\n'
        '- Giải thích từng quyết định: Integrated Gradients, Grad-CAM, giải thích phản thực.\n'
        '- Kiểm chứng độ trung thực của lời giải thích bằng cách xóa vùng rồi cho mô hình chấm lại (Colab).')
with todo:
    st.markdown('**Chưa làm / cần thầy góp ý**')
    st.markdown(
        '- Ngưỡng "cần phun" (hiện cỏ chiếm 1% diện tích ô) **chỉ để minh họa**: cần ngưỡng có căn cứ từ tài liệu về '
        'cỏ dại.\n'
        '- Kích thước ô 50 cm chưa có căn cứ: nên khớp với vùng mà máy phun điều khiển riêng được.\n'
        '- Chờ thầy xác nhận: cỏ dại có được tính là dịch hại không; dữ liệu một ruộng, một mùa vụ có đủ không; dữ '
        'liệu công bố ở hội nghị (WACV 2025) có được chấp nhận không.\n'
        '- Chưa kiểm tra trên ruộng khác.')

with st.expander('Hạn chế cần lưu ý khi đọc kết quả'):
    st.markdown(
        f'- **Tập kiểm tra nhỏ**: {n} ô, 8 vị trí trên ruộng. Mỗi tỉ lệ bỏ sót hay phun nhầm chỉ dựa trên vài ô, '
        'nên có thể dao động.\n'
        '- **Một ruộng, một mùa vụ** (Potsdam, Đức, 2023).\n'
        '- **Bảo đảm conformal chỉ gần đúng**, vì các ô nằm cạnh nhau thường giống nhau.\n'
        '- **Hiệu chỉnh riêng cho từng đợt bay còn yếu** vì quá ít ô để tính ngưỡng (thử ở trang 3, mục "Phạm vi").\n'
        '- Khi mô hình tìm cỏ đã tốt, ngưỡng cố định cũng gần đạt mức tốt nhất; giá trị của conformal là giữ mức sai '
        'trong giới hạn khi mô hình yếu hơn (xem trang 3 với mô hình RGB), đổi lại phải ra xem một số ô.')

with st.expander('Cách dùng bản demo'):
    st.markdown(
        '- Thanh bên trái chuyển giữa các trang, theo đúng thứ tự xử lý: dữ liệu → tìm cỏ → quyết định → giải thích '
        '→ kiểm chứng.\n'
        '- Trang 3 cho phép đổi ngưỡng, giới hạn sai và mô hình: hệ thống tính lại ngay. Khi thiết lập trùng một '
        'thiết lập đã lưu trong `results/`, trang báo số tính lại có khớp file không; khi khác, trang ghi rõ là '
        '"đang khám phá", không phải số trong báo cáo.\n'
        '- Không có mạng nơ-ron nào chạy trên máy này: các bước cần GPU đã chạy trên Colab, kết quả nằm trong '
        '`results/`.')
ui.footer()
