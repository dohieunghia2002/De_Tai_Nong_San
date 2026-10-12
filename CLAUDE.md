# CLAUDE.md

Hướng dẫn cho Claude Code khi làm việc với repo này. Người dùng có thể sửa file này.

## Bối cảnh

- Đề tài do giáo sư giao: *"Nghiên cứu mô hình hỗ trợ ra quyết định điều hành sản xuất theo thời gian thực tích hợp AI giải thích được (XAI-driven DSS) trong chuỗi chế biến nông sản."*
- Ngày 12/10/2026, người làm ban đầu bàn giao dự án cho một đồng nghiệp. Người đang làm việc với Claude là **người nhận bàn giao**, có thể chưa nắm hết chi tiết, nên giải thích rõ khi cần.
- Đọc `README.md` trước: tình trạng hiện tại, kết quả v4, các việc còn mở, cấu trúc thư mục. Chi tiết phương pháp và 6 lỗi đã sửa trong PenSimPy nằm trong `pilot/README.md`.
- Ngoài repo còn thư mục riêng `De_Tai_Nong_San_FILE_RIENG_TU` (không có trên GitHub): 6 bài báo PDF tham khảo và `lich_su_trao_doi.md` (toàn bộ trao đổi trước đây, khoảng 150 KB, gồm yêu cầu của giáo sư và phần trả lời 5 câu hỏi về đề tài). Khi cần, hỏi người dùng đường dẫn tới thư mục này; chỉ đọc phần cần thiết.

## Quy tắc bắt buộc

- **Dữ liệu:** giáo sư chỉ chấp nhận bộ dữ liệu công khai đã được bài báo uy tín công bố hoặc sử dụng. Không tự thu thập; không dùng dữ liệu chỉ có trên Kaggle, dữ liệu tổng hợp hay preprint không có bài báo gốc. Khi đề xuất dữ liệu, ghi bài báo gốc và tạp chí, và nêu rõ nếu dữ liệu thuộc loại không được chấp nhận.
- **Tính toán nặng:** không tự chạy mô phỏng, sinh dữ liệu hay huấn luyện trên máy người dùng. Mặc định đưa notebook hoặc script để chạy trên Google Colab (GPU T4). Chỉ chạy cục bộ các kiểm tra nhanh (vài giây), và mô tả trước khi chạy. Không tự làm chuỗi bước lớn ngoài yêu cầu. Lý do: người làm trước từng phản đối khi Claude chạy 120 mẻ mô phỏng trên laptop mà không hỏi.
- **Ngôn ngữ:** mọi tin nhắn hiển thị cho người dùng viết bằng tiếng Việt, kể cả cập nhật ngắn giữa các bước. Văn bản gửi giáo sư viết bằng tiếng Việt, dùng **"Hệ thống hỗ trợ ra quyết định"** (viết đầy đủ, không rút gọn thành "Hệ hỗ trợ").
- **Repo công khai:** không commit bài báo PDF (bản quyền, `.gitignore` đã chặn), thông tin cá nhân hay khóa API.

## Mã nguồn và cách chạy

- `pilot/` là bản đang dùng (v4). `pilot_v2/` là bản v2 chạy CPU; giữ nguyên vì `validate_gpu.py` so sánh với `pilot_v2/runs/v2/`.
- `jaxsim.py` (mô phỏng GPU bằng JAX) phải tái hiện đúng từng bước bản CPU (`sim.py`). Sau khi sửa bộ mô phỏng, chạy `validate_gpu.py` và chỉ dùng kết quả khi nó in "KHỚP".
- Không thay `pilot/pensimpy/` bằng bản PenSimPy trên PyPI: bản trong repo đã sửa lỗi.
- **Quy trình một lần thực nghiệm mới (vN):**
  1. Đóng gói mã thành `pilot_colab_vN.zip`, bên trong là thư mục `pilot_vN/`; tạo notebook `chay_tren_colab_vN.ipynb` theo mẫu bản v4 và đặt `PILOT_EXP = 'vN'`. Phải dùng tên mới: notebook bỏ qua bước giải nén nếu thư mục `pilot_vN/` đã có trên Drive.
  2. Người dùng upload zip và notebook vào `MyDrive/De_Tai_Nong_San/` (cùng chỗ với `pilot_v2/`), chạy trên Colab.
  3. Ô cuối notebook tạo `ket_qua_pilot_vN.zip`. Người dùng tải về, giải nén vào `Ket_qua_vN/` trong repo, rồi nhờ Claude phân tích.
- Số liệu tổng hợp của mỗi lần chạy nằm trong `runs/vN/results/summary.json`; dùng file này khi viết báo cáo, không ước lượng số liệu.
- `.gitattributes` đặt `* -text` để giữ nguyên từng byte của file dữ liệu và mô hình.

## Tính mới của đề tài (khảo sát ngày 02/10/2026)

Các công trình gần nhất trên IndPenSim:
- Mlinarič & Glavan, ERK 2025, kèm luận văn Đại học Ljubljana: DSS khuyến nghị hành động mỗi 24 giờ, không có XAI.
- Li, Qiu & You, *CES* 2024: bộ điều khiển học tăng cường, chỉ có phân tích KernelSHAP tổng thể.
- Luận văn ICESI 2025 và bài MLOps soft sensor, *C&CE* 2025 (cùng nhóm INRAE Toulouse): cảm biến mềm LSTM có SHAP, LIME.
- Multi-scale TCN, *Systems & Control Transactions* 2026: SHAP cho cửa sổ cảnh báo; nêu "khép vòng tới can thiệp" là hướng tương lai.
- Liu & Lou, *ISA Transactions* 2024: chẩn đoán lỗi bằng phản thực tế trên PenSim V2.

Khoảng trống đề tài nhắm tới: giải thích đối chiếu (Shapley so với công thức vận hành) cho từng khuyến nghị, và **kiểm chứng độ trung thực của lời giải thích bằng cách chạy lại bộ mô phỏng IndPenSim**. Dữ liệu e-nose thịt bò (Wijaya 2018) dùng để kiểm chứng trên dữ liệu thực phẩm thật.
