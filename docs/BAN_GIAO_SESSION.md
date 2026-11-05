# Bàn giao giữa các phiên làm việc với Claude (cập nhật 03/11/2026)

Đọc file này cùng `CLAUDE.md` trước khi làm tiếp. Mọi số liệu dưới đây lấy từ file kết quả thật, có ghi nguồn.

## 1. Đề tài đang ở đâu

**Đề tài giáo sư giao:** *"Nghiên cứu mô hình hỗ trợ ra quyết định điều hành sản xuất theo thời gian thực tích hợp AI giải thích được (XAI-driven DSS) trong chuỗi chế biến nông sản."* Người làm việc với Claude là người nhận bàn giao dự án từ ngày 12/10/2026.

**Các hướng đã đi qua:**
1. **Lên men penicillin (IndPenSim):** đã bỏ. Code còn trong lịch sử git, từ commit `14a3857` trở về trước.
2. **Xoài, phổ NIR** (Anderson 2020, file CSV; đặc tả trong `docs/PROMPT_ClaudeCode_DSS_XAI_xoai.md`):
   - Bước 1 (tái hiện baseline PLS) và Bước 2 (tầng quyết định) đã xong và đã push.
   - Bước 3 (ANN, CNN trên Colab: `src/step3_*.py`, `src/models_nn.py`, `notebooks/colab_cnn.ipynb`) đã viết code nhưng **chưa commit, chưa chạy**.
   - Ngày 18/10, thầy nhận xét: dữ liệu chỉ có CSV thì "đề tài yếu". Luận văn cần dữ liệu **ảnh hoặc video**, tốt nhất là **UAV**, hướng tới thu hoạch hoặc phun thuốc tự động. Phần NIR có thể dùng để **viết bài báo**.
3. **Tìm dữ liệu UAV** (18–30/10). Đã loại:
   - lúa: quá nhiều người làm;
   - cam UAV đa phổ CampanetaOrangeFruit: 21 GB, nặng so với Colab miễn phí;
   - cà phê chụp bằng drone (Makerere): quả chín chỉ khoảng 1%, không có chuỗi thời gian;
   - CADI-AI (điều): bài gốc chỉ là workshop; bản Hugging Face bị trộn ảnh điện thoại, mất GPS, khoảng 10% ảnh test trùng với train;
   - ngô bệnh đốm lá lớn (NLB): 55 GB;
   - đậu tương MH-SoyaHealthVision: chỉ khoảng 28 video, nhãn lệch theo ngày quay.
4. **Hướng hiện tại: WeedsGalore.** Phân vùng cỏ dại trên ảnh UAV đa phổ, rồi **quyết định phun thuốc trừ cỏ theo ô lưới**. Cỏ dại được coi là "dịch hại".
   - **Chưa có xác nhận của thầy** về ba điểm: (a) cỏ dại có được tính là dịch hại không; (b) dữ liệu 1 ruộng, 1 mùa có đủ không; (c) có chấp nhận bài báo dữ liệu kiểu *Data in Brief* hoặc workshop không.

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
  - 156 ô ảnh 600×600, mỗi ô tương đương 1,5 m × 1,5 m ngoài ruộng.
  - 5 dải phổ, kiểu số nguyên 16 bit, thứ tự R, G, B, NIR, RE.
  - Mặt nạ có 6 lớp: nền, ngô, dền, cỏ lồng vực, cúc dại, cỏ khác. Bài toán 3 lớp gộp mọi loại cỏ thành một.
  - Tập train/val/test là 104/26/26 ô ảnh, chia theo vị trí trên ruộng.
- **Ghép vị trí qua các ngày:** trong file chia tập val và test, mỗi khối các ô có ngày giảm dần là cùng một vị trí. Đã kiểm tra bằng mắt với tập test: có 8 vị trí. Tập train không ghép được theo cách này.
- **Độ phủ cỏ (trung vị theo ô ảnh):** 0,2% / 0,9% / 4,9% / 34% theo 4 đợt bay.

## 3. Tính mới (đã đọc kỹ ba bài gần nhất; PDF nằm trong `docs/`)

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

## 4. Code của thí nghiệm thử (chưa commit)

**Các module:**
- `src/weeds/data.py`: đọc file zip và tạo npz; `location_blocks` ghép vị trí qua các ngày.
- `src/weeds/model.py`: DeepLabv3+ ResNet50, output stride 8, theo kiến trúc VainF mà tác giả dùng; lớp tích chập đầu 5 kênh khởi tạo ngẫu nhiên.
- `src/weeds/train.py`:
  - Adam lr 1e-3, cross-entropy, tăng cường bằng xoay 90°/lật/nhiễu Gauss (độ lệch chuẩn 1/25), dùng fp16.
  - Chọn mô hình theo mIoU trên val, dừng sớm khi 30 epoch không cải thiện, tự dừng trước 3,5 giờ.
  - Chạy tiếp được nhờ `last.pt` cộng `best.json`. Lưu xác suất dự đoán vào `probs_*.npz`.
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
- `notebooks/colab_weeds_pilot.ipynb` được sinh bởi `scripts/make_weeds_notebook.py`, nhúng sẵn các module. **Sửa `src/weeds` thì phải chạy lại script sinh notebook.** Trên runtime T4 chỉ cần bấm Run all; lỗi ở bước nào thì dừng ở bước đó.
- Có 43 test trong `tests/test_weeds_*.py` và các test cũ. Chạy test cần dữ liệu bằng `WEEDSGALORE_ZIP=data/weedsgalore/weedsgalore-dataset.zip .venv/Scripts/python -m pytest`. Venv đã có torch 2.6.0 bản CPU.

## 5. Kết quả lần chạy 1 (`results/weeds_pilot/`, chạy trên Colab ngày 03/11/2026)

**Huấn luyện** (`train_done.json`, `history.json`):
- Dừng sớm sau 45 epoch; tốt nhất ở epoch 14 với mIoU val 71,69.
- Khoảng 18 giây mỗi epoch (trung vị), tổng 13,5 phút huấn luyện.
- mIoU trên val **dao động 30–72** giữa các epoch.

**Phân vùng** (`segmentation_metrics.json`):

| | mIoU | IoU ngô | IoU cỏ |
|---|---|---|---|
| Test, cả 4 ngày | **58,20** (bài báo: 82,90) | 36,98 | 39,70 |
| Test, 3 ngày trước khi phun | 73,1 (tính lại từ `probs_test.npz`) | – | – |

Trên 2 ô ảnh test của ngày 15/6, **69% điểm ảnh cỏ bị đoán thành ngô**.

**Kết luận đạt/không đạt** (`go_no_go.json`): **NO-GO, chỉ vì C1** (mIoU). C2 = 0,509 và C3 = 0,139 đều đạt.

**Tầng quyết định** (`decision_metrics.csv`, thiết lập chính):

| Chính sách | Giảm thuốc | Chưa quyết | Bỏ sót | Phun nhầm |
|---|---|---|---|---|
| Conformal 3 lựa chọn | 50,9% | 13,9% | 10,2% | 8,3% |
| Plug-in | 39,8% | 0% | 4,6% | 25,0% |
| Oracle | 50,0% | 0% | 0% | 0% |

- Hiệu chỉnh bằng val phun nhầm 14,8%, vượt β.
- Hiệu chỉnh riêng từng ngày có quá ít mẫu, nên xếp 64–92% số ô vào "chưa quyết".

**Giả thuyết chính (chưa kiểm chứng):** BatchNorm với batch 2 cộng learning rate cố định làm huấn luyện không ổn định. Bài báo dùng batch 8; code của tác giả mặc định batch 2.

## 6. Việc tiếp theo

1. **Chạy lần 2 với `--batch 8`, giữ nguyên mọi thứ khác.**
   - Phải dùng **thư mục kết quả mới** (ví dụ `run_dlv3p_msi_seed0_b8`). Nếu dùng lại thư mục cũ, notebook sẽ thấy `train_done.json` và bỏ qua huấn luyện.
   - Bộ nhớ ước tính khoảng 9–10 GB: kích hoạt khoảng 2,2 GB fp16 cho mỗi batch 2, đo bằng meta device. Phải xem dòng `peak GPU` ở bước chạy thử.
   - Nếu vẫn dao động, lần 3 thêm learning rate giảm dần.
   - **Đã chuẩn bị (03/11/2026):** notebook đã sinh lại với `BATCH = 8`, dùng chung batch này cho cả bước chạy thử. Kết quả nằm trong thư mục Drive `run_dlv3p_msi_seed0_b8`; file `ket_qua_weeds_pilot_b8.zip` giải nén vào `results/weeds_pilot_b8/`, không ghi đè lần 1. `train_done.json` giờ ghi thêm batch, lr và seed. Chưa chạy trên Colab.
2. Đọc toàn văn các bài về ngưỡng phun để thay τ minh họa bằng ngưỡng có nguồn.
3. Hỏi thầy ba điểm ở mục 1.4.
4. Commit: hiện còn nhiều file chưa commit, gồm `src/weeds/`, các test, notebook, `scripts/make_weeds_notebook.py`, `requirements.txt`, `.gitignore`, CLAUDE.md, README.md, code Bước 3 của xoài và `results/weeds_pilot/` (trừ file npz). **Chỉ commit hoặc push khi người dùng đồng ý.**

## 7. Lưu ý về cách làm việc

Các lưu ý này đã có trong CLAUDE.md hoặc bộ nhớ, nhắc lại ở đây cho đủ:
- Mọi tin nhắn viết bằng tiếng Việt; code và comment viết bằng tiếng Anh.
- Chỉ cài thư viện vào `.venv`, và thêm vào `requirements.txt` đúng phiên bản đã dùng.
- Không chạy tính toán nặng trên máy người dùng. Notebook Colab phải tính đến giới hạn khoảng 5 giờ GPU mỗi phiên.
- Không bịa số liệu.
- Dữ liệu phải công khai và có bài báo uy tín.
- **Không gửi email hay thông tin cá nhân của người dùng tới dịch vụ bên ngoài.** Đã từng lỡ gửi email cho Unpaywall một lần.
- Văn bản gửi thầy dùng cụm "Hệ thống hỗ trợ ra quyết định", viết đầy đủ.
