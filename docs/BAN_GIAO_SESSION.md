# Bàn giao giữa các phiên làm việc với Claude (cập nhật 05/11/2026)

Đọc file này cùng `CLAUDE.md` trước khi làm tiếp. Mọi số liệu dưới đây lấy từ file kết quả thật, có ghi nguồn.

## 1. Đề tài đang ở đâu

**Đề tài giáo sư giao:** *"Nghiên cứu mô hình hỗ trợ ra quyết định điều hành sản xuất theo thời gian thực tích hợp AI giải thích được (XAI-driven DSS) trong chuỗi chế biến nông sản."* Người làm việc với Claude là người nhận bàn giao dự án từ ngày 12/10/2026.

**Các hướng đã đi qua:**
1. **Lên men penicillin (IndPenSim):** đã bỏ. Code còn trong lịch sử git, từ commit `14a3857` trở về trước.
2. **Xoài, phổ NIR** (Anderson 2020, file CSV):
   - Bước 1 (tái hiện baseline PLS) và Bước 2 (tầng quyết định) đã chạy xong. Bước 3 (ANN, CNN trên Colab) đã viết code nhưng chưa chạy.
   - Ngày 18/10, thầy nhận xét dữ liệu chỉ có CSV thì "đề tài yếu"; luận văn cần dữ liệu **ảnh hoặc video**, tốt nhất là **UAV**. Phần NIR có thể dùng để **viết bài báo**.
   - **Ngày 05/11/2026 đã xóa toàn bộ phần xoài khỏi nhánh main.** Code, kết quả và đặc tả nằm ở tag git `xoai-nir` (commit `812cb14`), lấy lại bằng `git checkout xoai-nir`. File CSV và PDF của phần xoài cũng đã xóa khỏi máy; nếu cần, tải lại từ Mendeley Data (doi:10.17632/46htwnp833).
3. **Tìm dữ liệu UAV** (18–30/10). Các bộ đã loại:
   - lúa: quá nhiều người làm;
   - cam UAV đa phổ CampanetaOrangeFruit: 21 GB, nặng so với Colab miễn phí;
   - cà phê chụp bằng drone (Makerere): quả chín chỉ khoảng 1%, không có chuỗi thời gian;
   - CADI-AI (điều): bài gốc chỉ là workshop; bản Hugging Face bị trộn ảnh điện thoại, mất GPS, khoảng 10% ảnh test trùng với train;
   - ngô bệnh đốm lá lớn (NLB): 55 GB;
   - đậu tương MH-SoyaHealthVision: chỉ khoảng 28 video, nhãn lệch theo ngày quay.
4. **Hướng hiện tại: WeedsGalore.** Phân vùng cỏ dại trên ảnh UAV đa phổ, rồi **quyết định phun thuốc trừ cỏ theo ô lưới**.
   - **Chưa có xác nhận của thầy** về ba điểm: (a) cỏ dại có được tính là dịch hại không; (b) dữ liệu 1 ruộng, 1 mùa có đủ không; (c) có chấp nhận dữ liệu công bố ở hội nghị, hoặc bài báo dữ liệu kiểu *Data in Brief*, không.
   - Ngày 04–05/11, Claude đã soạn giúp người dùng các tin nhắn gửi thầy: phần trình bày đề tài, ba câu hỏi ở trên, và câu trả lời về input/output của dữ liệu (thầy hỏi). Claude cũng tạo gói ảnh xem trước của toàn bộ dữ liệu, vì thầy muốn xem ảnh trông như thế nào.
   - **Ngày 05/11, thầy góp ý về input/output:** "output không phải là model quyết định; output phải xác định: kết quả phân lớp / vùng...". Vì vậy output được trình bày thành hai kết quả:
     - output 1: kết quả phân vùng (segmentation) cho từng điểm ảnh;
     - output 2: kết quả phân lớp (classification) cho từng ô 0,5 m: phun / không phun / chưa chắc.
     Người dùng đã gửi thầy sơ đồ xử lý vẽ trên một ô ảnh test thật (`scripts/make_pipeline_figure.py`, ảnh lưu tại `data/weedsgalore/so_do_pipeline.png`) và bài báo gốc.
   - **Yêu cầu của thầy (05/11):** "chạy thử rồi đối sánh với bài báo có dataset đó, xem thế nào; nếu ổn thì viết rồi hẹn vô gặp mình họp chốt".

## 2. Dữ liệu WeedsGalore (đã kiểm tra trực tiếp)

- **Bài báo gốc:** Celikkan và cộng sự, WACV 2025 (IEEE/CVF).
- **Nơi lưu:** https://doidata.gfz.de/weedsgalore_e_celikkan_2024/
  - `weedsgalore-dataset.zip`: 336.989.804 byte, giấy phép CC BY 4.0. Bản trên máy: `data/weedsgalore/weedsgalore-dataset.zip`, đã bị gitignore.
  - Ảnh ghép toàn ruộng: 11,4 GB, là thứ duy nhất có tọa độ. Trong mỗi file TIFF, phần thông tin tọa độ nằm ở khoảng 1 GB tính từ đầu file.
- **Bối cảnh thu thập:**
  - Một ruộng ngô ở Potsdam (Đức), năm 2023.
  - Drone DJI P4 Multispectral bay ở độ cao 5 m, mỗi điểm ảnh 2,5 mm.
  - 4 đợt bay: 25/5, 30/5, 6/6, 15/6. **Ruộng được phun thuốc trừ cỏ ngày 12/6.**
- **Cấu trúc:**
  - 156 ô ảnh 600×600, mỗi ô tương đương 1,5 m × 1,5 m ngoài ruộng. Ba đợt đầu mỗi đợt có 48 ô, đợt 15/6 có 12 ô.
  - 5 dải phổ, kiểu số nguyên 16 bit, thứ tự R, G, B, NIR, RE. Giá trị điểm ảnh khoảng 5.000–65.000.
  - Mặt nạ phân vùng ngữ nghĩa (thư mục `semantics/`) có 6 lớp: nền, ngô, dền, cỏ lồng vực, cúc dại, cỏ khác. Bài toán 3 lớp gộp mọi loại cỏ thành một.
  - Ngoài ra có mặt nạ tách từng cây (`instances/`); đề tài chưa dùng.
  - Tập train/val/test là 104/26/26 ô ảnh, chia theo vị trí trên ruộng.
- **Ghép vị trí qua các ngày:** trong file chia tập val và test, mỗi khối các ô có ngày giảm dần là cùng một vị trí. Đã kiểm tra bằng mắt với tập test: có 8 vị trí. Tập train không ghép được theo cách này.
- **Độ phủ cỏ (trung vị theo ô ảnh):** 0,2% / 0,9% / 4,9% / 34% theo 4 đợt bay.
- **Ảnh xem trước:** `scripts/export_weeds_images.py` tạo ra `data/weedsgalore/anh_xem_truoc/`, gồm 4 ảnh tổng quan theo ngày, 156 ảnh chi tiết, README và LICENSE. Gói zip của thư mục này là `data/weedsgalore/anh_WeedsGalore_156_o_anh.zip` (76 MB).

## 3. Tính mới (đã đọc kỹ ba bài gần nhất; PDF nằm trong `docs/`)

PDF bài WACV 2025 tải ngày 05/11/2026 từ CVF Open Access: `docs/Celikkan_2025_WeedsGalore_WACV.pdf`. Đã đối chiếu: Bảng 3 ghi DeepLabv3+ ảnh 5 dải, 3 lớp, mIoU 82,90; phần cài đặt ghi batch 8, Adam lr 0,001.

- **Celikkan và cộng sự 2025 (bài gốc WeedsGalore):** độ bất định bằng MC dropout, đo ECE. Chỉ **gợi ý** hệ thống "không phun khi độ bất định cao", không xây dựng thành quy tắc quyết định.
- **Celikkan và cộng sự 2023 (ICCVW):** bản đồ độ bất định cho phân vùng cỏ và cây trồng. Không có quy tắc quyết định, không dùng conformal.
- **Melki và cộng sự 2025 (bản thảo arXiv):**
  - Dữ liệu riêng của công ty: ảnh RGB chụp từ máy phun mặt đất, không phải UAV.
  - Phân loại ảnh, conformal theo lớp, chỉ có quyết định **phun hoặc không phun**.
  - Không hiệu chỉnh lại khi gặp trang trại mới, không có trạng thái "chưa quyết", không có XAI.
  - Bài có định nghĩa sẵn các chỉ số "sprayed ratio", "spray reduction" và "spray surplus"; đề tài có thể dùng lại và trích dẫn.
- **Khoảng trống của đề tài:**
  - quyết định **phun / không phun / chưa quyết** cho từng ô lưới, dựa trên kết quả phân vùng ảnh UAV đa phổ công khai;
  - bảo đảm conformal theo từng lớp;
  - hiệu chỉnh lại khi sang đợt bay mới;
  - XAI có kiểm chứng độ trung thực, bằng phép che vùng có đối chứng ngẫu nhiên.
- **Ngưỡng phun có nguồn, nhưng chưa đọc toàn văn để lấy số:** Gutjahr 2012 và Ali 2015 (*Weed Research*), MoDiCoVi (*Sensors* 2016), Longchamps (*Weed Science*). **Ngưỡng τ hiện tại (1%) chỉ là minh họa.**

## 4. Code

**Các module:**
- `src/weeds/data.py`: đọc file zip và tạo npz; `location_blocks` ghép vị trí qua các ngày.
- `src/weeds/model.py`: DeepLabv3+ ResNet50, output stride 8, theo kiến trúc VainF mà tác giả dùng; lớp tích chập đầu 5 kênh khởi tạo ngẫu nhiên.
- `src/weeds/train.py`:
  - Adam lr 1e-3, cross-entropy, tăng cường bằng xoay 90°/lật/nhiễu Gauss (độ lệch chuẩn 1/25), dùng fp16.
  - Chọn mô hình theo mIoU trên val, dừng sớm khi 30 epoch không cải thiện, tự dừng trước 3,5 giờ.
  - Chạy tiếp được nhờ `last.pt` cộng `best.json`. Lưu xác suất dự đoán vào `probs_*.npz`.
  - `train_done.json` ghi batch, lr và seed của lần chạy.
  - Tùy chọn `--no-pretrained` chỉ dùng cho test trên máy.
- `src/weeds/decision.py`:
  - Chia ô ảnh thành ô lưới. Điểm số của ô là xác suất cỏ trung bình, tính bằng float64.
  - Ngưỡng conformal theo lớp: `lower_threshold` (bỏ sót ≤ α) và `upper_threshold` (phun nhầm ≤ β).
  - Các chính sách: phun toàn ruộng, oracle, plug-in, conformal chỉ chặn bỏ sót, conformal 3 lựa chọn.
  - `evaluate_crossfit`: hiệu chỉnh chéo 2 phần theo vị trí.
- `src/weeds/analyze.py`:
  - **Phương án chính:** hiệu chỉnh chéo 2 phần trên tập test theo vị trí, vì val đã dùng để chọn mô hình.
  - Phương án phụ: hiệu chỉnh bằng val.
  - Tiêu chí đề xuất: mIoU test ≥ 77,90; giảm thuốc ≥ 30%; "chưa quyết" ≤ 30%.
  - Thiết lập chính: ô lưới 200 điểm ảnh (0,5 m), τ = 1%, α = β = 0,1, gộp 3 ngày trước khi phun.

- `src/weeds/authors.py`: đánh giá mô hình công bố sẵn của tác giả bằng code của đề tài.
  - Nguồn: `ckpts.zip` (887 MB) trên trang dữ liệu, chứa 6 mô hình. Module dùng 4 mô hình thường: ảnh 5 dải và RGB, bài toán 3 lớp và 6 lớp. Hai mô hình MC dropout (`dlv3p_do_*`) chưa dùng.
  - Cách nạp: mạng của tác giả là DeepLabV3Plus theo VainF, chỉ khác tên tham số. Module đổi tên rồi nạp với `strict=True` và `weights_only=True`.
  - Cách đánh giá: dùng **chính hàm `train.evaluate`** đã cho ra kết quả lần chạy 2 (hàm này đã được mở rộng cho 6 lớp, cho RGB và cho tùy chọn fp32/fp16). Mỗi mô hình được đánh giá hai lần:
    - fp32, như `evaluate.py` của tác giả, để so với Bảng 3 và Bảng 4;
    - fp16, như lần chạy 2, để tách riêng phần chênh lệch do fp16.
  - Với mô hình 3 lớp, module lưu xác suất (từ lần đánh giá fp16) theo đúng định dạng của `train.py`, để chạy tiếp được `src.weeds.analyze`.
  - Tiêu chí C1 trong `analyze.go_no_go` so mIoU với số của bài báo cho đúng loại đầu vào (khóa `paper_test`, ví dụ RGB là 79,33 − 5). Các lần chạy của đề tài vẫn so với 82,90 − 5 = 77,90.
- Ngày 06/11 đã review lại code và sửa các lỗi review tìm ra; kết quả lần 2 tính lại bằng code mới khớp hoàn toàn với `decision_metrics.csv` và `go_no_go.json` đã lưu. Các thay đổi:
  - `decision.crossfit_decisions` trả về quyết định cho từng ô; `evaluate_crossfit` và script vẽ sơ đồ cùng dùng hàm này.
  - Lời giải thích trên sơ đồ đúng cho cả trường hợp ngưỡng không "bắt chéo", và cả khi ngưỡng là vô cực.
  - Bước giải nén checkpoint kiểm tra kích thước file.
- Đã đọc code của tác giả (github.com/GFZ/weedsgalore):
  - lệnh mẫu huấn luyện 100 epoch cố định, batch 8, lưu checkpoint mỗi 100 vòng lặp, rồi chọn theo val;
  - tính ở fp32, không dùng fp16;
  - ảnh đầu vào chia 65535; RGB là 3 dải R, G, B đầu tiên.

**Notebook và test:**
- `notebooks/colab_weeds_pilot.ipynb` được sinh bởi `scripts/make_weeds_notebook.py`, nhúng sẵn các module. **Sửa `src/weeds` thì phải chạy lại script sinh notebook.**
  - Biến `BATCH` ở mục 1 quyết định batch, thư mục kết quả trên Drive (`run_dlv3p_msi_seed0_b<BATCH>`) và tên gói kết quả (`ket_qua_weeds_pilot_b<BATCH>.zip`, giải nén ra `results/weeds_pilot_b<BATCH>/`).
  - Nếu chạy lần mới với cùng batch (ví dụ seed khác), phải đổi tên thư mục và tên gói trong script, nếu không sẽ bị bỏ qua hoặc ghi đè.
- `notebooks/colab_weeds_authors.ipynb` (sinh bởi cùng script): tải `ckpts.zip` lên Drive, đánh giá 4 mô hình của tác giả, chạy tầng quyết định trên 2 mô hình 3 lớp, rồi đóng gói thành `ket_qua_weeds_authors.zip`. Giải nén gói này vào `results/weeds_authors/`.
- Trên runtime T4 chỉ cần bấm Run all; lỗi ở bước nào thì dừng ở bước đó.
- `src/weeds/runs.py` định nghĩa các lần huấn luyện thêm và hàm tổng hợp; `train.py` có thêm `--lr-schedule cosine`, lưu trạng thái lịch learning rate trong checkpoint để chạy tiếp đúng; `train_done.json` ghi thêm epochs, patience, lr_schedule. `notebooks/colab_weeds_runs.ipynb` sinh bởi cùng script.
- Có 36 test trong `tests/` (các file `test_weeds_*.py` và `test_pipeline_figure.py`). Chạy test cần dữ liệu bằng `WEEDSGALORE_ZIP=data/weedsgalore/weedsgalore-dataset.zip .venv/Scripts/python -m pytest`. Venv đã có torch 2.6.0 bản CPU.

## 5. Kết quả thí nghiệm thử

**Lần 1** (`results/weeds_pilot/`, batch 2, chạy ngày 03/11/2026): **NO-GO, chỉ vì C1.**
- Dừng sớm sau 45 epoch, tốt nhất ở epoch 14 (mIoU val 71,69); 17,8 giây mỗi epoch.
- mIoU val dao động 31–72 giữa các epoch.
- mIoU test 58,20. Trên 2 ô ảnh test ngày 15/6, 69% điểm ảnh cỏ bị đoán thành ngô.

**Lần 2** (`results/weeds_pilot_b8/`, batch 8, mọi thiết lập khác giữ nguyên, chạy ngày 04/11/2026): **GO, đạt cả 3 tiêu chí.**
- **Huấn luyện:** dừng sớm sau 180 epoch, tốt nhất ở epoch 149 (mIoU val 83,94); 17,5 giây mỗi epoch, tổng 52,8 phút.
- **Val ổn định hơn nhiều:** 156/180 epoch có mIoU ≥ 70. Vẫn còn dao động: trong 30 epoch cuối, trung vị 82,9 nhưng có epoch tụt xuống 50,7.

| Phân vùng, IoU % (`segmentation_metrics.json`) | Lần 1 | Lần 2 | Bài báo |
|---|---|---|---|
| mIoU test, cả 4 ngày | 58,20 | **79,25** | 82,90 |
| IoU nền / ngô / cỏ | 97,93 / 36,98 / 39,70 | 98,12 / 65,16 / 74,46 | 98,45 / 72,93 / 77,31 |
| mIoU test, 3 ngày trước khi phun (tính lại từ `probs_test.npz`) | 73,1 | 84,2 | – |
| mIoU test theo ngày 25/5, 30/5, 6/6, 15/6 (tính lại) | 49,2 / 64,5 / 77,2 / 40,7 | 71,9 / 78,5 / 86,2 / 61,4 | – |

- Chênh lệch của lần 2 so với bài báo nằm chủ yếu ở lớp ngô: 15,5% điểm ảnh ngô bị đoán là nền, 12,7% bị đoán là cỏ.
- Ngày 15/6: tỉ lệ điểm ảnh cỏ bị đoán thành ngô giảm từ 69% xuống 0,4%, nhưng IoU ngô chỉ còn 26,4 (2 ô ảnh).

**Tầng quyết định, lần 2** (`decision_metrics.csv`, thiết lập chính, 216 ô lưới, 108 ô cần phun):

| Chính sách | Giảm thuốc | Chưa quyết | Bỏ sót | Phun nhầm |
|---|---|---|---|---|
| Conformal 3 lựa chọn | 49,5% | 6,5% (14 ô) | 0,9% (1 ô) | 1,9% (2 ô) |
| Plug-in | 51,4% | 0% | 5,6% (6 ô) | 2,8% (3 ô) |
| Oracle | 50,0% | 0% | 0% | 0% |
| *Conformal, lần 1* | *50,9%* | *13,9%* | *10,2%* | *8,3%* |

- **Hiệu chỉnh bằng val (lần 2):** bỏ sót 2,8%, phun nhầm 0%, chưa quyết 9,7%, đều trong giới hạn α, β. Lần 1 phun nhầm 14,8%, vượt β.
- **Hiệu chỉnh riêng từng ngày vẫn yếu do ít mẫu:** ngày 25/5 có 86% số ô chưa quyết (phần hiệu chỉnh chỉ có 2 ô cần phun), ngày 6/6 có 93% (97% số ô cần phun). Ngày 30/5 chỉ có 4,2%.
- **Ngưỡng bị "bắt chéo":** ở cả 2 phần hiệu chỉnh, λ_hi (khoảng 0,0076–0,0078) nhỏ hơn λ_lo (khoảng 0,0097–0,0109). Vì vậy 14 ô chưa quyết là các ô có điểm số sát ngưỡng 1%, nơi conformal loại được cả "cần phun" lẫn "không cần phun" (tập dự đoán rỗng). Khi viết phải diễn giải đúng điểm này.
- **Plug-in đã gần bằng oracle.** Conformal không giảm thuốc nhiều hơn plug-in; nó đổi 14 ô phải kiểm tra lấy việc giảm bỏ sót từ 6 xuống 1 ô.

**Đối sánh bằng mô hình công bố của tác giả** (`results/weeds_authors/`, chạy trên Colab ngày 06/11/2026):
- **Code đánh giá của đề tài được kiểm chứng.** Cả 4 mô hình ra đúng số của bài báo, lệch dưới 0,005 điểm mIoU, và mọi IoU từng lớp khớp đến 2 chữ số thập phân. Số liệu (fp32, `authors_vs_paper.json`):
  - ảnh 5 dải, 3 lớp: 82,90 (bài báo 82,90);
  - RGB, 3 lớp: 79,33 (79,33);
  - ảnh 5 dải, 6 lớp: 55,52 (55,52);
  - RGB, 6 lớp: 50,81 (50,81).
- Đánh giá bằng fp16 chỉ lệch so với fp32 dưới 0,002 điểm. Vì vậy chênh lệch 3,65 điểm của lần chạy 2 **hoàn toàn nằm ở khâu huấn luyện**. Mô hình 5 dải do đề tài tự huấn luyện chỉ ngang mô hình RGB của tác giả.
- Ma trận nhầm lẫn của mô hình tác giả (ảnh 5 dải) khớp Hình 4 của bài: ngô 0,079 / 0,792 / 0,130; cỏ 0,070 / 0,006 / 0,924. Mô hình đề tài bỏ sót thành nền nhiều hơn: ngô 15,5%, cỏ 12,0%.
- **Tầng quyết định**, thiết lập chính (216 ô, 108 ô cần phun), `decision_metrics.csv`:

| Mô hình | Chính sách | Giảm thuốc | Chưa quyết | Bỏ sót | Phun nhầm |
|---|---|---|---|---|---|
| Đề tài, 5 dải (79,25) | conformal / plug-in | 49,5% / 51,4% | 6,5% / 0% | 0,9% / 5,6% | 1,9% / 2,8% |
| Tác giả, 5 dải (82,90) | conformal / plug-in | 49,5% / 48,1% | 4,2% / 0% | 3,7% / 3,7% | 4,6% / 7,4% |
| Tác giả, RGB (79,33) | conformal / plug-in | 51,4% / 58,3% | 3,2% / 0% | 6,5% / **16,7%** | 3,7% / 0% |

- **Nhận xét:**
  - Với cả 3 mô hình, conformal giữ tỉ lệ bỏ sót và phun nhầm dưới α = β = 0,1, và tỉ lệ chưa quyết chỉ 3–7%.
  - Plug-in phụ thuộc vào mô hình: với mô hình RGB, plug-in bỏ sót 16,7% (18 ô). Đó là vì mô hình này đoán thiếu cỏ, nên ngưỡng 1% cố định là sai; conformal tự hạ ngưỡng xuống khoảng 0,6–0,8% và kéo tỉ lệ bỏ sót về 6,5%. Đây là luận điểm chính cho tầng quyết định.
  - mIoU cao hơn không có nghĩa là quyết định tốt hơn. Mô hình tác giả hơn 3,65 điểm mIoU nhưng bỏ sót 4 ô so với 1 ô của mô hình đề tài. Tuy nhiên chênh lệch chỉ vài ô nên chưa kết luận được mô hình nào tốt hơn ở mức quyết định.
  - Hiệu chỉnh riêng từng ngày yếu như nhau với cả 3 mô hình (25/5 và 6/6 có 85–93% ô chưa quyết), nên đây là giới hạn của dữ liệu, không phải của mô hình.

**Huấn luyện thêm** (`results/weeds_runs/`, đủ 6 lần chạy, Colab ngày 08/11/2026; batch 8, ảnh 5 dải; khoảng 17,3 giây mỗi epoch; tổng 3,26 giờ huấn luyện):

| Lần chạy | mIoU test, mô hình val cao nhất (epoch) | mIoU test, mô hình epoch cuối (epoch) |
|---|---|---|
| 100 epoch cố định, seed 0 | 79,60 (99) | 79,60 (99) |
| 100 epoch cố định, seed 1 | 81,74 (89) | 80,77 (99) |
| 100 epoch cố định, seed 2 | 81,53 (86) | 83,26 (99) |
| 100 epoch, lr cosine, seed 0 | 81,49 (99) | 81,49 (99) |
| Dừng sớm, seed 0 (lần chạy 2, `results/weeds_pilot_b8/`) | 79,25 (149) | chưa đánh giá |
| Dừng sớm, seed 1 | 81,20 (71) | **66,93** (101) |
| Dừng sớm, seed 2 | **83,57** (133) | 83,74 (163) |

- **Kết luận chính: khoảng cách 3,65 điểm là dao động ngẫu nhiên của quá trình huấn luyện, không phải code sai.**
  - 7 mô hình chọn theo val có mIoU test 79,25–83,57, trung bình 81,20 (độ lệch chuẩn 1,44).
  - `es_seed2` vượt bài báo (83,57 so với 82,90).
- **Cùng seed vẫn ra kết quả khác:** GPU dùng thuật toán không tất định (`cudnn.benchmark`).
  - Phiên 2 đã huấn luyện lại từ đầu cả 4 lần chạy 100 epoch đã xong ở phiên 1, không chạy tiếp; có thể do thư mục `runs` trên Drive không còn, ví dụ khi dùng tài khoản Google khác.
  - Số của phiên 1 (chỉ còn ghi lại, file đã bị ghi đè khi giải nén zip mới) lệch so với phiên 2 từ −2,90 đến +3,89 điểm cho cùng seed:
    - seed 0: 81,76 (phiên 1) so với 79,60 (phiên 2);
    - seed 1: 77,85 so với 81,74;
    - seed 2: 84,42 so với 81,53;
    - cosine: 81,74 so với 81,49.
  - Gộp cả 11 mô hình chọn theo val của hai phiên: trung bình 81,29, độ lệch chuẩn 1,86, khoảng 77,85–84,42; 2 mô hình đạt từ 82,90 trở lên.
  - **Khi báo cáo chỉ dùng số của phiên 2**, vì phiên 1 không còn file.
- **Chọn theo val an toàn hơn lấy epoch cuối.** Mô hình epoch cuối dao động mạnh hơn nhiều: độ lệch chuẩn 6,25, và `es_seed1` ở epoch cuối chỉ đạt 66,93 (IoU ngô 48,4). Giả thuyết "chọn val may mắn" bị bác bỏ. Learning rate cosine làm hai cách chọn trùng nhau.
- **Dao động tập trung ở ngày 15/6** (2 ô ảnh sau khi phun): mIoU 55,3–80,5 giữa 12 mô hình. Các ngày khác: 25/5 63,3–72,6; 30/5 71,3–78,5; 6/6 75,6–85,9 (`test_by_date`).
- **Tầng quyết định vững với cả 13 mô hình** (mIoU test 66,93–83,74, thiết lập chính):
  - giảm thuốc 49,5–51,4%;
  - chưa quyết 0–7,9%;
  - bỏ sót tối đa 8,3%, phun nhầm tối đa 7,4%, cả hai đều dưới α = β = 10%.
- Lần chạy 2 không có trên Drive của phiên Colab này, nên mô hình epoch cuối của nó chưa được đánh giá.

**Hạn chế:** lịch 100 epoch và lịch dừng sớm mỗi lịch có 3 seed, lịch cosine chỉ có 1; kết quả huấn luyện không tất định, nên cần báo cáo theo phân bố, không theo một lần chạy; tập test nhỏ (26 ô ảnh, 8 vị trí), nên các tỉ lệ bỏ sót và phun nhầm chỉ dựa trên 1–3 ô lưới; mIoU chỉ vượt ngưỡng C1 1,35 điểm.

## 6. Việc tiếp theo

1. **Đối sánh với bài báo** (thầy yêu cầu ngày 05/11), làm theo từng bước:
   - **Bước 1: đã xong ngày 06/11.** Code đánh giá khớp bài báo; kết quả ở mục 5.
   - **Bước 2 (người dùng yêu cầu ngày 06/11, notebook đã sẵn sàng, chưa chạy):** `notebooks/colab_weeds_runs.ipynb` chạy 6 lần huấn luyện định nghĩa trong `src/weeds/runs.py`, theo thứ tự ưu tiên:
     - 3 seed với lịch của tác giả: 100 epoch cố định, learning rate không đổi;
     - 1 lần 100 epoch với learning rate giảm dần (cosine);
     - 2 seed với lịch của lần chạy 2 (dừng sớm).
     Tổng khoảng 4 giờ GPU, 2 phiên Colab; notebook tự chạy tiếp phiên sau. Kết quả: `ket_qua_weeds_runs.zip`, giải nén vào `results/weeds_runs/`, chỉ gồm json và csv.
     - Đã đối chiếu kỹ code huấn luyện của tác giả: loss, tăng cường ảnh, DataLoader, khởi tạo mô hình đều giống. Chỉ khác lịch huấn luyện, cách chọn checkpoint và fp32/A100. fp32 không thử vì batch 8 ở fp32 nhiều khả năng không vừa bộ nhớ T4.
     - Lần chạy 2: mIoU val tốt nhất trong 100 epoch đầu là 83,19 (epoch 94), cả quá trình là 83,94 (epoch 149). Như vậy chỉ rút về 100 epoch chưa chắc giúp ích.
   - Mô hình phân vùng chính trong báo cáo gửi thầy là mô hình công bố của tác giả; các lần tự huấn luyện là phép kiểm tra khả năng tái hiện.
   - **Phát hiện khi review (06/11): có thể đang chọn mô hình "may mắn" trên val.**
     - Mô hình của tác giả: val 79,53, test 82,90 (RGB: val 74,26, test 79,32).
     - Lần chạy 2 thì ngược lại: val 83,94, test 79,25.
     - Chọn epoch có val cao nhất trong 180 epoch dao động mạnh có thể chọn trúng một epoch may mắn trên val. Vì vậy mỗi lần chạy giờ được đánh giá thêm **mô hình ở epoch cuối** (`--stage test --weights last`, kết quả trong `<run>/last/`). Mô hình epoch cuối của lần chạy 2 cũng được đánh giá ở đầu notebook.
     - `segmentation_metrics.json` có thêm `test_by_date` (mIoU test theo từng ngày bay) và `model` (mô hình nào, epoch nào).
   - **Sửa sau review (06/11):**
     - Đồng hồ phiên tính từ lúc máy Colab khởi động (`/proc/uptime`).
     - Cuối notebook gọi `drive.flush_and_unmount()`.
     - Chạy tiếp với thiết lập khác checkpoint thì báo lỗi.
     - Trạng thái bộ sinh ngẫu nhiên được lưu trong `last.pt`.
     - "patience" đếm lại khi `best.pt` mới hơn `last.pt`.
     - Epoch cuối không bị tính là "hết giờ".
     - `runs.py` không còn phụ thuộc giá trị mặc định của `train.py` (dùng `LEGACY`).
     - Bảng tổng hợp có cột batch và cột cách chọn mô hình, và xử lý được trường hợp chưa có lần chạy nào xong.
   - **Báo cáo gửi thầy (đã soạn ngày 06/11, dùng mô hình của tác giả làm chính theo lựa chọn của người dùng):** https://claude.ai/artifact/J1XPMkEg2d4pwGXoFKVeCs. Gồm tóm tắt, input/output kèm sơ đồ, dữ liệu, đối sánh, tầng quyết định kèm biểu đồ bỏ sót/phun nhầm, tính mới, hạn chế và kế hoạch, 6 điểm xin thầy chốt. Việc tiếp theo: người dùng gửi thầy và hẹn họp chốt.
2. **Chờ thầy trả lời** ba điểm ở mục 1.4.
3. **Đọc toàn văn các bài về ngưỡng phun** để thay τ minh họa bằng ngưỡng có nguồn.
4. **Thiết kế cách hiệu chỉnh lại khi sang đợt bay mới:** hiệu chỉnh riêng từng ngày hiện có quá ít mẫu.
5. **Làm mô-đun giải thích (XAI)** và phép kiểm chứng độ trung thực (che vùng, có đối chứng ngẫu nhiên).

## 7. Lưu ý về cách làm việc

Các lưu ý này đã có trong CLAUDE.md hoặc bộ nhớ, nhắc lại ở đây cho đủ:
- Mọi tin nhắn viết bằng tiếng Việt; code và comment viết bằng tiếng Anh.
- Chỉ cài thư viện vào `.venv`, và thêm vào `requirements.txt` đúng phiên bản đã dùng.
- Không chạy tính toán nặng trên máy người dùng. Notebook Colab phải tính đến giới hạn khoảng 5 giờ GPU mỗi phiên.
- Không bịa số liệu.
- Dữ liệu phải công khai và có bài báo uy tín.
- **Không gửi email hay thông tin cá nhân của người dùng tới dịch vụ bên ngoài.** Đã từng lỡ gửi email cho Unpaywall một lần.
- Văn bản gửi thầy dùng cụm "Hệ thống hỗ trợ ra quyết định", viết đầy đủ.
- Chỉ commit hoặc push khi người dùng đồng ý.
