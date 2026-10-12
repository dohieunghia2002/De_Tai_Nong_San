# Đề tài XAI-driven DSS trong chế biến nông sản

**Tên đề tài chính thức:** *"Nghiên cứu mô hình hỗ trợ ra quyết định điều hành sản xuất theo thời gian thực tích hợp AI giải thích được (XAI-driven DSS) trong chuỗi chế biến nông sản."*

**Tên đề xuất** (chưa trình giáo sư duyệt): *"Hệ thống hỗ trợ ra quyết định thời gian thực tích hợp trí tuệ nhân tạo giải thích được, có kiểm chứng lời giải thích bằng mô phỏng, cho điều hành quá trình lên men theo mẻ trong chế biến nông sản."*

## Bàn giao (12/10/2026)

Dự án được chuyển cho đồng nghiệp tiếp tục. Toàn bộ dự án gồm hai phần:

1. **Repo GitHub này:** mã nguồn, notebook Colab, dữ liệu đã sinh, mô hình, kết quả các lần thực nghiệm, sơ đồ quy trình.
2. **Thư mục `De_Tai_Nong_San_FILE_RIENG_TU`** (gửi riêng, không có trên GitHub): 6 bài báo PDF tham khảo và lịch sử làm việc với Claude. Xem file `DOC_TRUOC.md` trong thư mục đó.

Đọc theo thứ tự: file này → `pilot/README.md` (chi tiết phương pháp, các lỗi đã sửa trong PenSimPy) → `lich_su_trao_doi.md` trong thư mục riêng (toàn bộ quá trình trao đổi, yêu cầu của giáo sư, cách diễn giải kết quả).

## Tình trạng hiện tại

- **Bản mới nhất là v4.** Mã nguồn trong `pilot/`, kết quả trong `Ket_qua_v4/runs/v4/`.
- Số liệu tổng hợp đầy đủ: `Ket_qua_v4/runs/v4/results/summary.json`.

Kết quả chính của v4 (48 mẻ kiểm thử, 288 khuyến nghị, không mẻ nào tràn bình):

| Chính sách vận hành | Sản lượng trung bình |
|---|---|
| Giữ nguyên công thức | 2.943 kg |
| Quy tắc cố định: tăng đường và dầu 20% | 3.748 kg |
| Hành động cố định tốt nhất do mô hình chọn | 3.987 kg |
| **Hệ thống hỗ trợ ra quyết định** | **4.537 kg** |

| Tiêu chí kiểm chứng lời giải thích | SHAP tiêu chuẩn | Hiệu SHAP | **Shapley đối chiếu (đề xuất)** |
|---|---|---|---|
| Chọn đúng yếu tố chính | 82% | 90% | **93%** |
| Tương quan thứ hạng với sự thật | 0,56 | 0,61 | **0,82** |
| Sai số trung bình mỗi thành phần | 52,3 kg | 42,1 kg | **22,4 kg** |

**Hạn chế và việc còn mở:**
1. Kết quả chính đều trên mô phỏng lên men penicillin, chỉ *tương tự* chế biến nông sản.
2. Hệ thống hay chọn mức trần của dải hành động; cần đưa giới hạn thật của thiết bị vào.
3. Chưa đánh giá kinh tế (cần giá điện, giá nguyên liệu có nguồn trích dẫn).
4. Các khuyến nghị mà PAA là yếu tố chính đều bị cả ba phương pháp giải thích bỏ sót.
5. Mô hình trên dữ liệu thịt bò chưa đủ tin cậy (tỷ lệ cho xuất nhầm thịt hỏng 31–33%).
6. Trong thư mục dự án chưa có báo cáo chính thức gửi giáo sư. Phần trả lời 5 câu hỏi về đề tài (tên, dữ liệu, quy trình, kết quả, đóng góp) đã soạn sẵn trong `lich_su_trao_doi.md`, mục ngày 06/10/2026.

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| `pilot/` | **Mã nguồn bản v4 (bản đang dùng).** Bộ mô phỏng GPU (`jaxsim.py`), sinh dữ liệu, huấn luyện, đánh giá, bổ sung dữ liệu có chủ đích, đường cong học, thí nghiệm thịt bò. Có `README.md` riêng. |
| `pilot/pensimpy/` | Bộ mô phỏng IndPenSim bản Python (PenSimPy 1.2.5, MIT) đã sửa lỗi |
| `pilot/data/beef/` | Dữ liệu e-nose thịt bò (Wijaya 2018, CC BY 4.0) |
| `pilot/_debug.py`, `_dbg2.py`, `_smoke.py`, `_test_solver.py` | Script kiểm tra nhanh lúc phát triển |
| `pilot_v2/` | Mã nguồn và kết quả bản v2 (chạy CPU). **Bước kiểm chứng GPU của v3/v4 cần `pilot_v2/runs/v2/`.** |
| `Ket_qua_v3/runs/v3/` | Kết quả v3 (mốc "trước khi sửa điểm mù lời giải thích") |
| `Ket_qua_v4/runs/v4/` | Kết quả v4: `data/train.csv` (54.720 dòng), `models/`, `results/` |
| `data/train.csv`, `models/surrogate.txt` | Dữ liệu và mô hình của bản v1 |
| `chay_tren_colab*.ipynb` | Notebook chạy trên Colab, từ v1 đến v4 |
| `pilot_colab_v3.zip`, `pilot_colab_v4.zip` | Gói mã nguồn để upload lên Google Drive cho notebook v3, v4 |
| `so_do_quy_trinh.png`, `.svg` | Sơ đồ quy trình của đề tài, để chèn vào báo cáo |
| `so_do_quy_trinh.html` | Bản trang web của sơ đồ, kèm ví dụ một khuyến nghị thật và bảng mô tả từng khối |

## Cách chạy lại thực nghiệm v4

Mọi tính toán nặng chạy trên **Google Colab (GPU T4)**, không chạy trên laptop.

1. Upload `pilot_colab_v4.zip` và `chay_tren_colab_v4.ipynb` vào `MyDrive/De_Tai_Nong_San/` trên Google Drive. Thư mục `pilot_v2/` (kèm `runs/v2/`) cũng phải có trên Drive, vì ô kiểm chứng so sánh với kết quả CPU của v2.
2. Mở notebook bằng Colab, chọn *Runtime → Change runtime type → T4 GPU*.
3. Chạy lần lượt từng ô. Nếu ô kiểm chứng in "CÓ SAI LỆCH", dừng lại kiểm tra trước khi chạy tiếp.

Nếu sửa mã trong `pilot/`, cần đóng gói lại thành file zip mới trước khi upload.

## Yêu cầu và quy ước của đề tài

- **Giáo sư chỉ chấp nhận bộ dữ liệu công khai đã được các bài báo uy tín công bố hoặc sử dụng.** Không tự thu thập dữ liệu, không dùng dữ liệu chỉ có trên Kaggle hoặc dữ liệu tổng hợp không có bài báo gốc. Khi đề xuất dữ liệu mới, ghi rõ bài báo gốc và tạp chí.
- Khi trình giáo sư, nên lập luận: đề tài dùng *bộ mô phỏng* IndPenSim đã công bố làm chuẩn so sánh (Goldrick 2015, 2019); ba nhóm nghiên cứu khác cũng tự sinh dữ liệu từ bộ mô phỏng này; bản của đề tài tái hiện đúng số liệu đã công bố (2.943 ± 823 kg so với 2.912 ± 786 kg).
- Văn bản gửi giáo sư viết bằng tiếng Việt, dùng **"Hệ thống hỗ trợ ra quyết định"** (viết đầy đủ, không rút gọn thành "Hệ hỗ trợ").

## Những thứ không có trong repo

| Nội dung | Ở đâu |
|---|---|
| 6 bài báo PDF tham khảo (bản quyền của nhà xuất bản) | Thư mục riêng, `1_Bai_bao_tham_khao/` |
| Lịch sử trao đổi với Claude, ghi chú bối cảnh đề tài | Thư mục riêng, `2_Lich_su_lam_viec_voi_Claude/` |
| Thư mục `__pycache__/` | Không cần: Python tự sinh lại khi chạy |
| Trang sơ đồ trên claude.ai | Thuộc tài khoản Claude cá nhân của người làm trước, người khác không mở được. Nội dung giống hệt `so_do_quy_trinh.html`. |

## Nguồn dữ liệu và giấy phép

- **IndPenSim:** Goldrick và cộng sự (2015), *J. Biotechnol.* 193:70–82; Goldrick và cộng sự (2019), *Comput. Chem. Eng.* 130:106471, doi:10.1016/j.compchemeng.2019.05.037. Bộ dữ liệu 100 mẻ chính thức: https://data.mendeley.com/datasets/pdnjz7zz5x/2 (đề tài không dùng trực tiếp).
- **PenSimPy:** Quartic.ai, giấy phép MIT (`pilot/pensimpy/LICENSE_PenSimPy_MIT.txt`).
- **E-nose thịt bò:** Wijaya, Sarno, Zulaika (2018), *Data in Brief*. Mendeley Data, doi:10.17632/mwmhh766fc.3, giấy phép CC BY 4.0.
