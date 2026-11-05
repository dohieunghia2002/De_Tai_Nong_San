# CLAUDE.md

Hướng dẫn cho Claude Code khi làm việc với repo này. Người dùng có thể sửa file này.

## Tình trạng hiện tại (03/11/2026): đọc `docs/BAN_GIAO_SESSION.md` trước tiên

- Hướng đang làm: **WeedsGalore**, phân vùng cỏ dại từ ảnh UAV đa phổ và quyết định phun thuốc theo ô lưới. Code nằm trong `src/weeds/`, notebook trong `notebooks/colab_weeds_pilot.ipynb`.
- Lần chạy thử thứ nhất: không đạt vì mIoU thấp; tầng quyết định thì đạt.
- Việc tiếp theo: chạy lần 2 với batch 8.
- Phần xoài NIR bên dưới vẫn giữ nguyên: dùng để viết bài báo, và các quy tắc chung vẫn áp dụng.

## Bối cảnh

- Đề tài do giáo sư giao: *"Nghiên cứu mô hình hỗ trợ ra quyết định điều hành sản xuất theo thời gian thực tích hợp AI giải thích được (XAI-driven DSS) trong chuỗi chế biến nông sản."*
- Ngày 12/10/2026, người làm ban đầu bàn giao dự án cho một đồng nghiệp. Người đang làm việc với Claude là **người nhận bàn giao**, có thể chưa nắm hết chi tiết, nên giải thích rõ khi cần.
- Ngày 18/10/2026, đề tài chuyển từ mô phỏng lên men penicillin (IndPenSim) sang **hỗ trợ ra quyết định thu hoạch xoài từ phổ NIR**. Mã và kết quả cũ đã xóa khỏi repo; nếu cần xem lại, lấy từ lịch sử git (commit `14a3857` trở về trước).
- **Đặc tả đầy đủ của đề tài mới nằm trong `docs/PROMPT_ClaudeCode_DSS_XAI_xoai.md`. Đọc file này trước khi viết code:** dữ liệu, kiến trúc M1–M5, các bước thực nghiệm, mốc số liệu cần tái hiện, cấu trúc repo mong muốn.
- `README.md` ghi tình trạng hiện tại và cấu trúc thư mục.

## Quy tắc bắt buộc

- **Dữ liệu:** giáo sư chỉ chấp nhận bộ dữ liệu công khai đã được bài báo uy tín công bố hoặc sử dụng. Không tự thu thập, không tự sinh dữ liệu; không dùng dữ liệu chỉ có trên Kaggle, dữ liệu tổng hợp hay preprint không có bài báo gốc. Khi đề xuất dữ liệu, ghi bài báo gốc và tạp chí, và nêu rõ nếu dữ liệu thuộc loại không được chấp nhận. Bộ dữ liệu hiện dùng (Anderson et al. 2020, *Postharvest Biology and Technology*) đạt yêu cầu này.
- **Dữ liệu đa phương tiện (thầy nhận xét ngày 18/10/2026):** dữ liệu chỉ có bảng số (CSV) như bộ NIR thì "đề tài yếu", không được đánh giá là bỏ nhiều công. Luận văn cần dữ liệu **ảnh hoặc video**; thầy gợi ý **dữ liệu UAV (drone)**, hướng tới thiết kế thu hoạch tự động. Phần NIR đã làm có thể dùng để viết bài báo.
- **Không bịa số liệu:** mọi con số trong báo cáo phải lấy từ output thực tế (`results/summary.json`). Chưa chạy thì ghi "chưa có kết quả".
- **Tính toán nặng:** không tự huấn luyện mạng nơ-ron (ANN, 1D-CNN) trên máy người dùng; viết notebook Google Colab (GPU T4). Chỉ chạy cục bộ các kiểm tra nhanh (vài giây), và mô tả trước khi chạy. Làm từng bước nhỏ; sau mỗi bước in kết quả kiểm tra (số dòng, số quả, sai số so với mốc) và dừng lại nếu lệch mốc. Không tự làm chuỗi bước lớn ngoài yêu cầu. Lý do: người làm trước từng phản đối khi Claude chạy mô phỏng dài trên laptop mà không hỏi.
- **Giới hạn Colab miễn phí (người dùng ghi nhận ngày 18/10/2026):** mỗi phiên GPU T4 chỉ dùng được tối đa khoảng **5 tiếng**, sau đó có thể bị tạm ngừng cấp GPU. Notebook phải:
  - chạy phần tải và tiền xử lý dữ liệu trên runtime CPU;
  - lưu dữ liệu gọn và checkpoint mỗi epoch lên Drive, chạy tiếp được sau khi bị ngắt;
  - dừng an toàn trước khoảng 4,5 tiếng;
  - đo thời gian một epoch trước khi lên kế hoạch số thí nghiệm.
- **Ngôn ngữ:** mọi tin nhắn hiển thị cho người dùng viết bằng tiếng Việt, kể cả cập nhật ngắn giữa các bước. Code, tên biến, comment viết bằng tiếng Anh. Văn bản gửi giáo sư viết bằng tiếng Việt, dùng **"Hệ thống hỗ trợ ra quyết định"** (viết đầy đủ, không rút gọn thành "Hệ hỗ trợ").
- **Repo công khai:** không commit file PDF, file dữ liệu gốc `data/*.csv` (`.gitignore` đã chặn cả hai), thông tin cá nhân hay khóa API.

## Dữ liệu và mã nguồn

- Dữ liệu: `data/MangoDMC_NIR_Data_v3.csv` (12.011 phổ × 315 cột). Chỉ dùng file v3, không dùng v4 ở giai đoạn này. Giữ nguyên phân chia có sẵn trong cột `Set`, **không chia ngẫu nhiên**.
- Mỗi quả được quét 2–2,6 lần và file không có mã quả. Tạo mã quả bằng `groupby(['Set','Pop','Type','Cultivar','DM'])`. RMSEP tính theo phổ (như bài gốc); quyết định đưa ra theo quả (trung bình dự đoán các phổ của quả).
- `scripts/pilot_dss.py` là pilot ban đầu (PLS và tầng quyết định), dùng làm tham chiếu. Mã chính viết trong `src/` theo cấu trúc ở mục 6 của đặc tả. Chạy từ thư mục gốc repo bằng Python của venv: `.venv/Scripts/python -m pytest`, `.venv/Scripts/python -m src.step1_baseline`. **Chỉ cài thư viện vào `.venv/`** (tạo từ `requirements.txt`, phiên bản cố định), không cài vào Python chung của máy; thư viện mới phải thêm vào `requirements.txt` đúng phiên bản đã dùng. Khi console Windows báo lỗi mã hóa tiếng Việt, đặt `PYTHONIOENCODING=utf-8`.
- Ngưỡng conformal q dùng định nghĩa chuẩn: giá trị thứ ⌈(n+1)(1−α)⌉ khi xếp |phần dư| của các quả tập Tuning theo thứ tự tăng dần. Pilot dùng `np.quantile` có nội suy, nên ở mùa vụ 5, T = 15 có đúng 1 quả cho kết quả khác. Test `test_step2_with_pilot_quantile_matches_pilot_exactly` chứng minh đó là khác biệt duy nhất.
- Quy ước bias: `mean(dự đoán − thật)`, giống Anderson 2020 (bias mùa vụ 4 là −0,07). **Walsh 2024 dùng quy ước ngược lại**, `mean(thật − dự đoán)`. Điều này đã xác nhận qua `utils/evaluation.py` trong repo GitHub của Walsh, và qua Bảng 1 của Walsh, nơi PLS của Anderson có bias +0,07. Khi dùng số của Walsh phải đổi dấu bias.
- **Bước 3 (ANN, 1D-CNN) chạy trên Colab** bằng `notebooks/colab_cnn.ipynb`. Notebook clone repo từ GitHub, nên phải push code trước khi người dùng chạy. Người dùng upload CSV vào `MyDrive/De_Tai_Nong_San/data/`, chạy notebook, rồi giải nén `ket_qua_buoc3.zip` vào thư mục gốc repo. Venv cục bộ không có TensorFlow, nên `src/models_nn.py` chỉ được kiểm tra bằng bước chạy thử (mục 5) trong notebook. Siêu tham số lấy theo code của Walsh (github.com/jemjemwalsh/research-deep-learning-nirs); mốc 0,77 của đặc tả không so sánh được với cách chia tập cố định (xem README).
- Cố định random seed. Lưu mọi kết quả ra CSV/JSON trong `results/`, kèm `results/summary.json`; dùng file này khi viết báo cáo, không ước lượng số liệu.
- Không tự gán ý nghĩa hóa học cho bước sóng nếu không có nguồn trích dẫn.
- `.gitattributes` đặt `* -text` để giữ nguyên từng byte của file dữ liệu và mô hình.

## Đóng góp của đề tài

Không nằm ở việc dự đoán DMC chính xác hơn (bài toán đã bão hòa, RMSEP 0,77–1,01 %FW), mà ở **tầng quyết định**: biết khi nào "chưa quyết" dựa trên độ bất định đã hiệu chỉnh; xử lý lệch mùa vụ bằng hiệu chỉnh độ lệch theo lô; giải thích từng quyết định và **kiểm chứng độ trung thực của lời giải thích**; đánh giá bằng chỉ số ở mức quyết định. Hướng mới chưa có khảo sát tài liệu về tính mới; cần làm trước khi khẳng định tính mới với giáo sư.
