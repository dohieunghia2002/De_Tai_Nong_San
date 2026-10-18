# Hệ thống hỗ trợ ra quyết định thu hoạch xoài từ phổ NIR (XAI-driven DSS)

**Tên đề tài do giáo sư giao:** *"Nghiên cứu mô hình hỗ trợ ra quyết định điều hành sản xuất theo thời gian thực tích hợp AI giải thích được (XAI-driven DSS) trong chuỗi chế biến nông sản."*

**Hướng cụ thể (từ 18/10/2026):** *Hệ thống hỗ trợ ra quyết định thu hoạch xoài theo thời gian thực từ phổ NIR, tích hợp AI giải thích được.* Đặc tả đầy đủ: [`docs/PROMPT_ClaudeCode_DSS_XAI_xoai.md`](docs/PROMPT_ClaudeCode_DSS_XAI_xoai.md).

## Bài toán

Hợp tác xã và nhà đóng gói xoài cần quyết định quả nào đủ độ chín để thu hoạch hoặc phân hạng. Độ chín đo bằng hàm lượng chất khô (DMC, %FW). Cắt quả để đo thì chậm và phá mẫu; máy NIR cầm tay cho kết quả tức thì nhưng có sai số, nhất là khi sang mùa vụ mới.

Hệ thống nhận phổ NIR của một quả, dự đoán DMC, rồi đưa ra một trong ba quyết định: **đủ chín**, **chưa chín** hoặc **chưa quyết** (chuyển sang đo phá mẫu). Mỗi quyết định kèm lời giải thích và chỉ số tin cậy.

## Tình trạng hiện tại

- Ngày 18/10/2026: chuyển từ đề tài lên men penicillin sang đề tài này. Mã và kết quả cũ đã xóa; nếu cần, xem lịch sử git (commit `14a3857` trở về trước).
- Đã có pilot `scripts/pilot_dss.py` (PLS 11 thành phần, khoảng conformal, hiệu chỉnh lô, cờ bất thường phổ). Các mốc mà pilot đạt được ghi trong mục 5 của đặc tả.
- **Bước 1 (tái hiện baseline PLS) đã xong ngày 18/10/2026**, khớp các mốc. Số liệu trong `results/summary.json`, mục `step1`:

| Tập kiểm thử | Số phổ | RMSEP (%FW) | Bias, dự đoán − thật (%FW) | Mốc |
|---|---|---|---|---|
| Mùa vụ 4 (`Val Ext`) | 1.448 | 1,014 | −0,065 | Anderson 2020: 1,01 và −0,07 |
| Mùa vụ 5 (`Val Ext 2`) | 320 | 1,831 | −1,475 | Pilot: 1,83 và −1,48 |

- Bước tiếp theo: Bước 2 của đặc tả, tầng quyết định với PLS.

## Cách chạy

Tạo môi trường ảo một lần (Windows, Python 3.10), từ thư mục gốc repo:

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Sau đó chạy bằng Python của venv:

```
.venv\Scripts\python -m pytest                 # kiểm tra số phổ, số quả và RMSEP mùa vụ 4 (vài giây)
.venv\Scripts\python -m src.step1_baseline     # Bước 1: in so sánh với các mốc, ghi kết quả vào results/
```

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| `data/` | `MangoDMC_NIR_Data_v3.csv` (35 MB, **không commit**, tải theo hướng dẫn ở mục Dữ liệu) |
| `docs/PROMPT_ClaudeCode_DSS_XAI_xoai.md` | Đặc tả đề tài: dữ liệu, kiến trúc M1–M5, các bước thực nghiệm, mốc kiểm tra |
| `docs/*.pdf` | Bài báo gốc Anderson 2020 (`1-s2.0-S0925521420301629-main.pdf`), bài Walsh 2024 và luận văn của Walsh (chỉ có trên máy, không commit) |
| `scripts/pilot_dss.py` | Pilot ban đầu, dùng làm tham chiếu |
| `src/data.py` | Đọc dữ liệu, tạo mã quả, kiểm tra số phổ và số quả của từng tập |
| `src/preprocess.py` | M1: tiền xử lý phổ (Savitzky–Golay trên toàn phổ, cắt 684–990 nm; tùy chọn SNV) |
| `src/models_pls.py` | M2: mô hình PLS và các chỉ số RMSEP, bias, SEP, R² |
| `src/step1_baseline.py` | Chạy Bước 1 và ghi kết quả |
| `tests/` | Test cho dữ liệu, tiền xử lý và mốc RMSEP |
| `results/` | Kết quả: `summary.json` (số liệu tổng hợp), `step1_*.json/csv` |
| `notebooks/` | Notebook Colab cho ANN và 1D-CNN (sẽ tạo ở Bước 3) |

## Dữ liệu

- **Mango DMC and NIR spectra**, Anderson, Walsh, Subedi, Walsh. Mendeley Data, doi:[10.17632/46htwnp833](https://doi.org/10.17632/46htwnp833) (bản v5), giấy phép CC BY 4.0.
- Bài báo gốc: Anderson et al. (2020), *Postharvest Biology and Technology* 168:111202 (xem mục Tài liệu tham khảo).
- File dùng: `MangoDMC_NIR_Data_v3.csv`, gồm 12.011 phổ và 315 cột (9 cột thông tin, nhãn `DM`, 306 cột phổ hấp thụ 285–1200 nm, bước 3 nm). Tải bộ dữ liệu từ Mendeley Data và đặt file này vào `data/`.
- Phân chia có sẵn trong cột `Set`: `Cal`, `Tuning` (huấn luyện); `Val Ext` (mùa vụ 4); `Val Ext 2` (mùa vụ 5). Không chia ngẫu nhiên.

## Tài liệu tham khảo

- Anderson N.T., Walsh K.B., Subedi P.P., Hayes C.H. (2020). Achieving robustness across season, location and cultivar for a NIRS model for intact mango fruit dry matter content. *Postharvest Biology and Technology* 168:111202, doi:10.1016/j.postharvbio.2020.111202. Bài gốc của bộ dữ liệu và baseline PLS.
- Walsh J., Neupane A., Li M. (2024). Evaluation of 1D convolutional neural network in estimation of mango dry matter content. *Spectrochimica Acta Part A* 311:124003, doi:10.1016/j.saa.2024.124003. Code: https://github.com/jemjemwalsh/research-deep-learning-nirs
- Walsh J.P. (2024). *Deep Learning in Estimation of Fruit Attributes Using Near Infrared Spectroscopy.* Luận văn Master of Research, Central Queensland University.

## Yêu cầu và quy ước

- **Giáo sư chỉ chấp nhận bộ dữ liệu công khai đã được các bài báo uy tín công bố hoặc sử dụng.** Không tự thu thập hay tự sinh dữ liệu.
- Huấn luyện mạng nơ-ron (ANN, 1D-CNN) chạy trên **Google Colab (GPU T4)**, không chạy trên laptop. PLS và các kiểm tra nhanh chạy cục bộ được.
- Mọi số liệu trong báo cáo lấy từ `results/summary.json`; chưa chạy thì ghi "chưa có kết quả".
- Văn bản gửi giáo sư viết bằng tiếng Việt, dùng **"Hệ thống hỗ trợ ra quyết định"** (viết đầy đủ, không rút gọn thành "Hệ hỗ trợ").
