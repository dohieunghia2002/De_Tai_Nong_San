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

## 3. Tính mới (đã đọc kỹ ba bài gần nhất; PDF nằm trong `docs/`, riêng bài WACV 2025 chưa có PDF)

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

**Notebook và test:**
- `notebooks/colab_weeds_pilot.ipynb` được sinh bởi `scripts/make_weeds_notebook.py`, nhúng sẵn các module. **Sửa `src/weeds` thì phải chạy lại script sinh notebook.**
  - Biến `BATCH` ở mục 1 quyết định batch, thư mục kết quả trên Drive (`run_dlv3p_msi_seed0_b<BATCH>`) và tên gói kết quả (`ket_qua_weeds_pilot_b<BATCH>.zip`, giải nén ra `results/weeds_pilot_b<BATCH>/`).
  - Nếu chạy lần mới với cùng batch (ví dụ seed khác), phải đổi tên thư mục và tên gói trong script, nếu không sẽ bị bỏ qua hoặc ghi đè.
- Trên runtime T4 chỉ cần bấm Run all; lỗi ở bước nào thì dừng ở bước đó.
- Có 20 test trong `tests/test_weeds_*.py`. Chạy test cần dữ liệu bằng `WEEDSGALORE_ZIP=data/weedsgalore/weedsgalore-dataset.zip .venv/Scripts/python -m pytest`. Venv đã có torch 2.6.0 bản CPU.

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

**Hạn chế:** mỗi cấu hình chỉ chạy một seed; tập test nhỏ (26 ô ảnh, 8 vị trí), nên các tỉ lệ bỏ sót và phun nhầm chỉ dựa trên 1–3 ô lưới; mIoU chỉ vượt ngưỡng C1 1,35 điểm.

## 6. Việc tiếp theo

1. **Chờ thầy trả lời** ba điểm ở mục 1.4, cùng phản hồi về phần trình bày đề tài và gói ảnh.
2. **Chạy thêm 2 seed với batch 8** (khoảng 1 giờ GPU mỗi lần) để biết mIoU 79,25 có ổn định không. Phải đổi tên thư mục và tên gói kết quả trong `scripts/make_weeds_notebook.py` (xem mục 4).
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
