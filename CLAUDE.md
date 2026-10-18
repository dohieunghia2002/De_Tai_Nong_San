# CLAUDE.md

Hướng dẫn cho Claude Code khi làm việc với repo này. Người dùng có thể sửa file này.

## Bối cảnh

- Đề tài do giáo sư giao: *"Nghiên cứu mô hình hỗ trợ ra quyết định điều hành sản xuất theo thời gian thực tích hợp AI giải thích được (XAI-driven DSS) trong chuỗi chế biến nông sản."*
- Ngày 12/10/2026, người làm ban đầu bàn giao dự án cho một đồng nghiệp. Người đang làm việc với Claude là **người nhận bàn giao**, có thể chưa nắm hết chi tiết, nên giải thích rõ khi cần.
- Ngày 18/10/2026, đề tài chuyển từ mô phỏng lên men penicillin (IndPenSim) sang **hỗ trợ ra quyết định thu hoạch xoài từ phổ NIR**. Mã và kết quả cũ đã xóa khỏi repo; nếu cần xem lại, lấy từ lịch sử git (commit `14a3857` trở về trước).
- **Đặc tả đầy đủ của đề tài mới nằm trong `docs/PROMPT_ClaudeCode_DSS_XAI_xoai.md`. Đọc file này trước khi viết code:** dữ liệu, kiến trúc M1–M5, các bước thực nghiệm, mốc số liệu cần tái hiện, cấu trúc repo mong muốn.
- `README.md` ghi tình trạng hiện tại và cấu trúc thư mục.

## Quy tắc bắt buộc

- **Dữ liệu:** giáo sư chỉ chấp nhận bộ dữ liệu công khai đã được bài báo uy tín công bố hoặc sử dụng. Không tự thu thập, không tự sinh dữ liệu; không dùng dữ liệu chỉ có trên Kaggle, dữ liệu tổng hợp hay preprint không có bài báo gốc. Khi đề xuất dữ liệu, ghi bài báo gốc và tạp chí, và nêu rõ nếu dữ liệu thuộc loại không được chấp nhận. Bộ dữ liệu hiện dùng (Anderson et al. 2020, *Postharvest Biology and Technology*) đạt yêu cầu này.
- **Không bịa số liệu:** mọi con số trong báo cáo phải lấy từ output thực tế (`results/summary.json`). Chưa chạy thì ghi "chưa có kết quả".
- **Tính toán nặng:** không tự huấn luyện mạng nơ-ron (ANN, 1D-CNN) trên máy người dùng; viết notebook Google Colab (GPU T4). Chỉ chạy cục bộ các kiểm tra nhanh (vài giây), và mô tả trước khi chạy. Làm từng bước nhỏ; sau mỗi bước in kết quả kiểm tra (số dòng, số quả, sai số so với mốc) và dừng lại nếu lệch mốc. Không tự làm chuỗi bước lớn ngoài yêu cầu. Lý do: người làm trước từng phản đối khi Claude chạy mô phỏng dài trên laptop mà không hỏi.
- **Ngôn ngữ:** mọi tin nhắn hiển thị cho người dùng viết bằng tiếng Việt, kể cả cập nhật ngắn giữa các bước. Code, tên biến, comment viết bằng tiếng Anh. Văn bản gửi giáo sư viết bằng tiếng Việt, dùng **"Hệ thống hỗ trợ ra quyết định"** (viết đầy đủ, không rút gọn thành "Hệ hỗ trợ").
- **Repo công khai:** không commit file PDF, file dữ liệu gốc `data/*.csv` (`.gitignore` đã chặn cả hai), thông tin cá nhân hay khóa API.

## Dữ liệu và mã nguồn

- Dữ liệu: `data/MangoDMC_NIR_Data_v3.csv` (12.011 phổ × 315 cột). Chỉ dùng file v3, không dùng v4 ở giai đoạn này. Giữ nguyên phân chia có sẵn trong cột `Set`, **không chia ngẫu nhiên**.
- Mỗi quả được quét 2–2,6 lần và file không có mã quả. Tạo mã quả bằng `groupby(['Set','Pop','Type','Cultivar','DM'])`. RMSEP tính theo phổ (như bài gốc); quyết định đưa ra theo quả (trung bình dự đoán các phổ của quả).
- `scripts/pilot_dss.py` là pilot ban đầu (PLS và tầng quyết định), dùng làm tham chiếu. Mã chính viết trong `src/` theo cấu trúc ở mục 6 của đặc tả. Chạy từ thư mục gốc repo bằng Python của venv: `.venv/Scripts/python -m pytest`, `.venv/Scripts/python -m src.step1_baseline`. **Chỉ cài thư viện vào `.venv/`** (tạo từ `requirements.txt`, phiên bản cố định), không cài vào Python chung của máy; thư viện mới phải thêm vào `requirements.txt` đúng phiên bản đã dùng. Khi console Windows báo lỗi mã hóa tiếng Việt, đặt `PYTHONIOENCODING=utf-8`.
- Ngưỡng conformal q dùng định nghĩa chuẩn: giá trị thứ ⌈(n+1)(1−α)⌉ khi xếp |phần dư| của các quả tập Tuning theo thứ tự tăng dần. Pilot dùng `np.quantile` có nội suy, nên ở mùa vụ 5, T = 15 có đúng 1 quả cho kết quả khác. Test `test_step2_with_pilot_quantile_matches_pilot_exactly` chứng minh đó là khác biệt duy nhất.
- Quy ước bias: `mean(dự đoán − thật)`. Với quy ước này, bias mùa vụ 4 tái hiện đúng dấu của Anderson 2020 (−0,07). Walsh 2024 báo bias PLS mùa vụ 5 là +1,56, ngược dấu với kết quả ở đây (−1,48), nên quy ước của Walsh có thể ngược lại; chưa xác nhận.
- Cố định random seed. Lưu mọi kết quả ra CSV/JSON trong `results/`, kèm `results/summary.json`; dùng file này khi viết báo cáo, không ước lượng số liệu.
- Không tự gán ý nghĩa hóa học cho bước sóng nếu không có nguồn trích dẫn.
- `.gitattributes` đặt `* -text` để giữ nguyên từng byte của file dữ liệu và mô hình.

## Đóng góp của đề tài

Không nằm ở việc dự đoán DMC chính xác hơn (bài toán đã bão hòa, RMSEP 0,77–1,01 %FW), mà ở **tầng quyết định**: biết khi nào "chưa quyết" dựa trên độ bất định đã hiệu chỉnh; xử lý lệch mùa vụ bằng hiệu chỉnh độ lệch theo lô; giải thích từng quyết định và **kiểm chứng độ trung thực của lời giải thích**; đánh giá bằng chỉ số ở mức quyết định. Hướng mới chưa có khảo sát tài liệu về tính mới; cần làm trước khi khẳng định tính mới với giáo sư.
