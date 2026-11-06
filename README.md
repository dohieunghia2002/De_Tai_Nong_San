# Hệ thống hỗ trợ ra quyết định phun thuốc trừ cỏ theo vùng từ ảnh UAV đa phổ (XAI-driven DSS)

**Tên đề tài do giáo sư giao:** *"Nghiên cứu mô hình hỗ trợ ra quyết định điều hành sản xuất theo thời gian thực tích hợp AI giải thích được (XAI-driven DSS) trong chuỗi chế biến nông sản."*

## Bài toán

Phun thuốc trừ cỏ đều khắp ruộng thì tốn thuốc và ảnh hưởng môi trường, trong khi cỏ thường chỉ mọc thành từng đám. Ảnh drone đa phổ cho biết chỗ nào có cỏ, nhưng mô hình AI vẫn có lúc nhận nhầm cỏ với cây trồng. Vì vậy phun hoàn toàn theo kết quả AI có thể bỏ sót cỏ hoặc phun thừa.

Hệ thống gồm ba phần:
1. Mô hình học sâu (DeepLabv3+) phân vùng ảnh drone 5 dải phổ thành nền đất / ngô / cỏ dại.
2. Mô-đun quyết định chia ruộng thành ô 0,5 m × 0,5 m. Với mỗi ô, nó đưa ra một trong ba quyết định: **phun**, **không phun** hoặc **chưa quyết** (cho người ra kiểm tra). Ngưỡng conformal giới hạn tỉ lệ bỏ sót và tỉ lệ phun nhầm.
3. Giải thích từng quyết định và kiểm chứng độ trung thực của lời giải thích (chưa làm).

## Dữ liệu

- **WeedsGalore** (Celikkan và cộng sự, IEEE/CVF WACV 2025). Tải tại https://doidata.gfz.de/weedsgalore_e_celikkan_2024/, giấy phép CC BY 4.0.
- Dùng file `weedsgalore-dataset.zip` (337 MB), đặt vào `data/weedsgalore/` (không commit).
- **Bối cảnh thu thập:** một ruộng ngô ở Potsdam (Đức), năm 2023. Drone DJI P4 Multispectral bay ở độ cao 5 m, mỗi điểm ảnh 2,5 mm.
- **Các đợt bay:** 25/5, 30/5, 6/6 (mỗi đợt 48 ô ảnh) và 15/6 (12 ô ảnh). Ruộng được phun thuốc trừ cỏ ngày 12/6.
- **Input:** ô ảnh 600 × 600 điểm ảnh (1,5 m × 1,5 m ngoài ruộng), 5 dải phổ R, G, B, NIR, RE, ảnh 16 bit.
- **Output (nhãn):** mặt nạ phân vùng ngữ nghĩa 6 lớp: nền đất, ngô, dền, cỏ lồng vực, cúc dại, cỏ khác. Bài toán 3 lớp gộp các loài cỏ thành một lớp.
- **Chia tập:** 104 / 26 / 26 ô ảnh train / val / test, chia theo vị trí trên ruộng. Giữ nguyên, không chia lại.
- **Nhãn phun / không phun** không có sẵn trong dữ liệu. Nhãn này được suy ra từ mặt nạ: một ô cần phun nếu cỏ dại chiếm từ τ diện tích ô trở lên. Hiện dùng τ = 1%, chỉ là ngưỡng minh họa.

## Tình trạng hiện tại (05/11/2026)

Đã chạy thí nghiệm thử 2 lần trên Colab (GPU T4). Lần 2 chỉ đổi batch từ 2 lên 8, mọi thiết lập khác giữ nguyên. Mỗi lần chạy một seed.

**Phân vùng** (nguồn: `results/weeds_pilot*/segmentation_metrics.json`, IoU tính bằng %):

| | Lần 1 (batch 2) | Lần 2 (batch 8) | Bài báo gốc |
|---|---|---|---|
| mIoU test | 58,20 | **79,25** | 82,90 |
| IoU nền / ngô / cỏ | 97,93 / 36,98 / 39,70 | 98,12 / 65,16 / 74,46 | 98,45 / 72,93 / 77,31 |

**Tầng quyết định** (nguồn: `results/weeds_pilot_b8/decision_metrics.csv`, lần 2):
- Thiết lập chính: ô 0,5 m, τ = 1%, α = β = 0,1, gộp 3 ngày trước khi phun, hiệu chỉnh chéo 2 phần theo vị trí trên tập test.
- Có 216 ô lưới, trong đó 108 ô cần phun.

| Chính sách | Giảm thuốc so với phun cả ruộng | Chưa quyết | Bỏ sót | Phun nhầm |
|---|---|---|---|---|
| Conformal 3 lựa chọn | 49,5% | 6,5% | 0,9% | 1,9% |
| Plug-in (phun nếu cỏ dự đoán ≥ τ) | 51,4% | 0% | 5,6% | 2,8% |
| Oracle (biết nhãn thật) | 50,0% | 0% | 0% | 0% |

Kết luận đạt/không đạt (`go_no_go.json`):
- Lần 1 **không đạt** (NO-GO) vì mIoU thấp.
- Lần 2 **đạt** (GO): mIoU test ≥ 77,90, giảm thuốc ≥ 30%, chưa quyết ≤ 30%.
- Tập test nhỏ (26 ô ảnh, 8 vị trí), nên các tỉ lệ trên chỉ dựa trên vài ô lưới.

Việc tiếp theo: xem mục 6 của [`docs/BAN_GIAO_SESSION.md`](docs/BAN_GIAO_SESSION.md).

## Cách chạy

Tạo môi trường ảo một lần (Windows, Python 3.10), từ thư mục gốc repo:

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Chạy test. Test cần đến dữ liệu sẽ bị bỏ qua nếu không có file zip:

```
set WEEDSGALORE_ZIP=data/weedsgalore/weedsgalore-dataset.zip
.venv\Scripts\python -m pytest
```

**Huấn luyện chạy trên Google Colab** (GPU T4), không chạy trên laptop:

1. Nếu đã sửa `src/weeds/`, sinh lại notebook: `.venv\Scripts\python scripts/make_weeds_notebook.py`.
2. Mở `notebooks/colab_weeds_pilot.ipynb` trên Colab (*File → Upload notebook*), chọn *Runtime → Change runtime type → T4 GPU*, rồi *Runtime → Run all*. Notebook tự tải dữ liệu về Drive.
3. Tải file `ket_qua_weeds_pilot_*.zip` trên Drive về và giải nén vào thư mục gốc repo. File sẽ tạo thư mục `results/weeds_pilot_*/`.

Chạy lại tầng quyết định trên máy, từ xác suất dự đoán đã tải về (dữ liệu cần ở dạng npz, do notebook tạo):

```
.venv\Scripts\python -m src.weeds.analyze --data <thư mục chứa weedsgalore_*.npz> --out results/weeds_pilot_b8
```

Xuất ảnh JPG xem được của toàn bộ 156 ô ảnh: `.venv\Scripts\python scripts/export_weeds_images.py`. Mỗi ô ảnh gồm ảnh màu thật, ảnh màu giả NIR và nhãn.

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| `data/weedsgalore/` | File zip dữ liệu và ảnh xem trước (**không commit**) |
| `src/weeds/data.py` | Đọc file zip, kiểm tra kích thước và nhãn, tạo file npz gọn, ghép vị trí qua các ngày |
| `src/weeds/model.py` | DeepLabv3+ ResNet-50, output stride 8, lớp tích chập đầu 5 kênh (theo code của tác giả) |
| `src/weeds/train.py` | Huấn luyện, chọn mô hình theo mIoU val, dừng sớm, chạy tiếp từ checkpoint, lưu xác suất dự đoán |
| `src/weeds/decision.py` | Chia ô lưới, ngưỡng conformal theo lớp, các chính sách quyết định, chỉ số ở mức quyết định |
| `src/weeds/analyze.py` | Chạy tầng quyết định cho mọi thiết lập và kết luận đạt/không đạt |
| `src/weeds/authors.py` | Đánh giá các mô hình công bố sẵn của tác giả bằng code của đề tài, so với Bảng 3–4 của bài báo |
| `src/weeds/runs.py` | Các lần huấn luyện thêm (nhiều seed, lịch huấn luyện của tác giả, learning rate giảm dần) và bảng tổng hợp |
| `scripts/make_weeds_notebook.py` | Sinh 2 notebook Colab có nhúng sẵn mã nguồn `src/weeds/` |
| `scripts/export_weeds_images.py` | Xuất ảnh xem trước của toàn bộ dữ liệu |
| `scripts/make_pipeline_figure.py` | Vẽ sơ đồ xử lý của hệ thống trên một ô ảnh test thật |
| `notebooks/colab_weeds_pilot.ipynb` | Notebook Colab huấn luyện và chạy tầng quyết định (sinh tự động, không sửa tay) |
| `notebooks/colab_weeds_authors.ipynb` | Notebook Colab đối sánh với bài báo bằng mô hình của tác giả (sinh tự động) |
| `notebooks/colab_weeds_runs.ipynb` | Notebook Colab chạy các lần huấn luyện thêm, tự chạy tiếp qua nhiều phiên (sinh tự động) |
| `tests/` | Test đọc dữ liệu, huấn luyện trên ảnh nhỏ (CPU), tầng quyết định |
| `results/weeds_pilot/`, `results/weeds_pilot_b8/` | Kết quả lần 1 (batch 2) và lần 2 (batch 8); file `probs_*.npz` không commit |
| `docs/BAN_GIAO_SESSION.md` | Bàn giao giữa các phiên làm việc: tình trạng, kết quả, việc tiếp theo |
| `docs/*.pdf` | Bài báo liên quan (chỉ có trên máy, không commit) |

Các hướng đã bỏ vẫn còn trong lịch sử git: lên men penicillin (commit `14a3857` trở về trước) và thu hoạch xoài từ phổ NIR (tag `xoai-nir`).

## Tài liệu tham khảo

- Celikkan E., Kunzmann T., Yeskaliyev Y., Itzerott S., Klein N., Herold M. (2025). WeedsGalore: A Multispectral and Multitemporal UAV-based Dataset for Crop and Weed Segmentation in Agricultural Maize Fields. *IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025*, tr. 4767–4777. Bản công khai: [CVF Open Access](https://openaccess.thecvf.com/content/WACV2025/html/Celikkan_WeedsGalore_A_Multispectral_and_Multitemporal_UAV-Based_Dataset_for_Crop_and_WACV_2025_paper.html). Bài gốc của bộ dữ liệu; kết quả phân vùng dùng để so sánh nằm ở Bảng 3.
- Celikkan E., Saberioon M., Herold M., Klein N. (2023). Semantic Segmentation of Crops and Weeds with Probabilistic Modeling and Uncertainty Quantification. *ICCV 2023 Workshops.*
- Melki P., Bombrun L., Diallo B., Dias J., Da Costa J.-P. (2025). Uncertainty Guarantees on Automated Precision Weeding using Conformal Prediction. Bản thảo arXiv:2501.07185. Nguồn của các chỉ số "sprayed ratio", "spray reduction", "spray surplus".

## Yêu cầu và quy ước

- **Giáo sư chỉ chấp nhận bộ dữ liệu công khai đã được các bài báo uy tín công bố hoặc sử dụng.** Không tự thu thập hay tự sinh dữ liệu. Luận văn cần dữ liệu ảnh hoặc video, tốt nhất là UAV.
- Huấn luyện mạng nơ-ron chạy trên **Google Colab (GPU T4)**, không chạy trên laptop. Test và phân tích tầng quyết định chạy cục bộ được.
- Mọi số liệu trong báo cáo lấy từ file kết quả trong `results/`. Chưa chạy thì ghi "chưa có kết quả".
- Văn bản gửi giáo sư viết bằng tiếng Việt, dùng **"Hệ thống hỗ trợ ra quyết định"** (viết đầy đủ, không rút gọn thành "Hệ hỗ trợ").
