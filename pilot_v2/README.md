# Pilot XAI-DSS trên IndPenSim (bản v2)

Thí nghiệm ban đầu cho đề tài *"Mô hình hỗ trợ ra quyết định điều hành sản xuất theo thời gian thực tích hợp AI giải thích được (XAI-driven DSS) trong chuỗi chế biến nông sản"*.

## Pilot trả lời câu hỏi gì

1. Bộ mô phỏng IndPenSim (bản Python) có chạy đúng không? Sản lượng mẻ theo công thức có gần số liệu đã công bố không?
2. DSS có tăng được sản lượng so với chạy theo công thức không? **Và có hơn một quy tắc cố định không** (bản v2)?
3. **Điểm mới chính:** lời giải thích cho từng khuyến nghị có đúng với "sự thật" không? Sự thật ở đây lấy bằng cách chạy lại bộ mô phỏng với từng tổ hợp thay đổi. Pilot so sánh ba cách giải thích:
   - M1, TreeSHAP chuẩn: cách các bài báo hiện nay hay dùng.
   - M2, hiệu TreeSHAP: so với hành động theo công thức.
   - M3, Shapley đối chiếu: phương pháp đề xuất.
4. Trên dữ liệu thịt bò thật (e-nose), mô hình có bị rò rỉ dữ liệu không? Mô hình dựa vào cảm biến khí hay vào nhiệt độ/độ ẩm phòng?

## Bản v2 khác bản v1 thế nào

| | v1 (đã chạy) | v2 |
|---|---|---|
| Dải hành động khi sinh dữ liệu | ×0,7 – ×1,3 | **×0,6 – ×1,6** |
| Các mức DSS được chọn | 0,8 / 0,9 / 1,0 / 1,1 / 1,2 | **0,6 / 0,8 / 1,0 / 1,2 / 1,4 / 1,6** |
| Ràng buộc thể tích bình | không có | **DSS bỏ các hành động được dự báo làm thể tích vượt 100.000 L** (thêm một mô hình dự báo thể tích) |
| Mốc so sánh | chỉ công thức | công thức; **quy tắc cố định "đường và dầu ×1,2"**; **hành động cố định tốt nhất** do mô hình chọn trên dữ liệu huấn luyện |
| Ghi lại thêm | — | thể tích lớn nhất, lượng đường và dầu đã dùng của mọi mẻ |
| Quy mô mặc định | 50 mẻ huấn luyện, 10 mẻ kiểm thử | **120 mẻ huấn luyện, 24 mẻ kiểm thử** |

Kết quả v2 nằm trong `runs/v2/`, tách riêng khỏi kết quả v1.

## Cách chạy trên Google Colab

1. Upload **`pilot_colab_v2.zip`** và **`chay_tren_colab_v2.ipynb`** vào cùng thư mục trên Google Drive với bản cũ, ví dụ `MyDrive/De_Tai_Nong_San/`. Không cần giải nén. Notebook giải nén vào thư mục mới `pilot_v2/` và không đụng đến `pilot/` cũ.
2. Mở `chay_tren_colab_v2.ipynb` bằng Google Colab.
3. Chọn *Runtime → Change runtime type → CPU*. Không cần GPU: phần chậm nhất là giải phương trình vi phân, chỉ chạy trên CPU.
4. Chạy lần lượt từng ô. Mọi kết quả được lưu thẳng vào Drive.
5. Nếu Colab bị ngắt kết nối: chạy lại các ô 1–3, rồi chạy lại ô đang dở. Những mẻ đã xong sẽ được bỏ qua, không chạy lại.

Thời gian ước tính trên Colab miễn phí (2 CPU):

| Chế độ | Mẻ huấn luyện | Mẻ kiểm thử | Thời gian |
|---|---|---|---|
| `NHANH` | 50 | 10 | khoảng 1–1,5 giờ |
| `DAY_DU` (mặc định) | 120 | 24 | khoảng 2–3 giờ |

Toàn bộ tính toán chạy trên máy chủ của Google. Máy tính của bạn chỉ mở trang web Colab, nên cần giữ tab Colab mở trong lúc chạy.

## Các file

| File | Chức năng |
|---|---|
| `pensimpy/` | Bộ mô phỏng IndPenSim bản Python (PenSimPy 1.2.5, giấy phép MIT, xem `pensimpy/LICENSE_PenSimPy_MIT.txt`) đã sửa lỗi |
| `sim.py` | Lớp bọc bộ mô phỏng: chạy từng bước, tách nhánh (fork) mẻ tại thời điểm quyết định |
| `dss_common.py` | Cấu hình thí nghiệm v1/v2; thời điểm quyết định (96, 120, …, 216 h); 4 biến điều khiển (đường, dầu, sục khí, PAA); các biến quan sát được; giới hạn thể tích |
| `gen_data.py` | Sinh dữ liệu huấn luyện. Tại mỗi thời điểm quyết định, thử 6 hành động khác nhau từ cùng một trạng thái; ghi sản lượng, thể tích lớn nhất, lượng đường và dầu |
| `train_dss.py` | Huấn luyện mô hình dự báo sản lượng 24 h tới và mô hình dự báo thể tích; chọn hành động cố định tốt nhất. Kiểm định chéo chia theo mẻ (không rò rỉ) |
| `eval_dss.py` | Chạy 4 chính sách trên các mẻ kiểm thử (công thức, 2 quy tắc cố định, DSS); giải thích từng khuyến nghị bằng M1/M2/M3; tính sự thật bằng mô phỏng |
| `analyze.py` | Tổng hợp số liệu vào `runs/v2/results/summary.json` |
| `beef_pilot.py` | Thí nghiệm phụ trên dữ liệu e-nose thịt bò thật (không đổi so với v1, không cần chạy lại) |
| `data/beef/` | Dữ liệu e-nose (Wijaya và cộng sự, 2018) |

## Kết quả sinh ra (trong `runs/v2/`)

- `results/surrogate_cv.json`: độ chính xác của hai mô hình dự báo (kiểm định chéo theo mẻ) và hành động cố định tốt nhất.
- `results/batches.csv`: mỗi mẻ kiểm thử, theo 4 chính sách: sản lượng cuối, thể tích lớn nhất, lượng đường và dầu đã dùng.
- `results/decisions.csv`: từng khuyến nghị của DSS, kèm lợi ích dự báo, lợi ích thật, thể tích, và ba lời giải thích so với sự thật.
- `results/summary.json`: các con số tổng hợp để đưa vào báo cáo.
- `models/`: hai mô hình đã huấn luyện và hành động cố định tốt nhất.
- Ô cuối notebook gom kết quả thành `ket_qua_pilot_v2.zip` trên Drive. Bạn tải file này về, đặt vào `D:\De_Tai_Nong_San\`, rồi báo Claude để phân tích và viết báo cáo cho giáo sư.

## Những chỗ đã sửa so với PenSimPy gốc

Bộ PenSimPy 1.2.5 trên PyPI không chạy được nguyên trạng. Các chỗ đã sửa:

1. **Bộ giải ODE.** Bản gốc cần module C++ `fastodeint` (boost::odeint, phương pháp Dormand–Prince RK45, sai số 1e-6), phải biên dịch mới dùng được. Ở đây dùng `scipy.integrate.solve_ivp(method="RK45", rtol=1e-6, atol=1e-6)` với cùng phương pháp và sai số (`sim.py`).
2. **Lỗi trong phương trình pH** (`pensimpy/ode/indpensim_ode_py.py`). Bản Python ghi đè biến trạng thái `y[6]` ([H+]) ngay trong hàm đạo hàm, làm hỏng phép tích phân. Đã sửa giống bản C++, dùng biến cục bộ `x6`.
3. **[H+] âm tạm thời.** Khi châm bazơ mạnh, các bước thử của RK45 có thể cho [H+] hơi âm. Giá trị này được chặn dưới 1e-12, chỉ khi tính đạo hàm. Lời giải được chấp nhận vẫn do bộ kiểm soát sai số RK45 quyết định (`sim.py`).
4. **Cách đọc công thức (recipe).** Điểm đặt `{time: t, value: v}` nghĩa là "giá trị v áp dụng **đến** thời điểm t", như bản MATLAB gốc. PenSimPy lại hiểu là "từ t trở đi", khiến lần xả dịch 2 giờ (4000 L/h) kéo dài thành 28 giờ và rút cạn bình (`sim.py`).
5. **Chế độ chạy theo công thức** (`Raman_spec = 0`). PAA đi theo công thức thay vì bị vòng PID dựa trên Raman ghi đè, giống mẻ 1–30 trong bộ dữ liệu của Goldrick 2019 (`sim.py`).
6. **Số liệu phân tích phòng thí nghiệm (offline).** Bản gốc ghi sai mốc thời gian nên không bao giờ ghi lại số liệu offline. Ở đây tự tính lại theo quy tắc của IndPenSim: lấy mẫu mỗi 12 giờ, có kết quả sau 4 giờ (`dss_common.py`).

Kiểm tra nhanh trên máy cục bộ ở chế độ công thức: sản lượng mẻ seed 1, 2, 3 lần lượt là 3640,4 / 3258,4 / 2704,8 kg, và thể tích bình luôn dưới 100.000 L. Để so sánh, Goldrick 2019 (Bảng 2) báo cáo trung bình 2912 ± 786 kg cho mẻ theo công thức, độ dài cố định.

## Nguồn dữ liệu và giấy phép

- **IndPenSim:** Goldrick, S. và cộng sự (2015), *J. Biotechnol.* 193:70–82; Goldrick, S. và cộng sự (2019), *Comput. Chem. Eng.* 130:106471. Bộ mô phỏng được công bố làm chuẩn so sánh (benchmark).
- **PenSimPy:** Quartic.ai, giấy phép MIT.
- **E-nose thịt bò:** Wijaya, D.R., Sarno, R., Zulaika, E. (2018), "Electronic nose dataset for beef quality monitoring in uncontrolled ambient conditions", *Data in Brief*. Dữ liệu tại Mendeley Data, doi:10.17632/mwmhh766fc.3, giấy phép CC BY 4.0. Dữ liệu được dùng nguyên bản, không chỉnh sửa.
