# CLAUDE.md

Hướng dẫn cho Claude Code khi làm việc với repo này. Người dùng có thể sửa file này.

## Tình trạng hiện tại (05/11/2026): đọc `docs/BAN_GIAO_SESSION.md` trước tiên

- Đề tài: **WeedsGalore**. Phân vùng cỏ dại từ ảnh UAV đa phổ, rồi quyết định **phun / không phun / chưa quyết** cho từng ô lưới bằng conformal. Code trong `src/weeds/`, notebook Colab là `notebooks/colab_weeds_pilot.ipynb`.
- Thí nghiệm thử:
  - lần 1 (batch 2, `results/weeds_pilot/`) không đạt vì mIoU thấp;
  - lần 2 (batch 8, `results/weeds_pilot_b8/`) đạt cả 3 tiêu chí (GO).
- Ngày 05/11/2026, phần xoài NIR đã được xóa khỏi nhánh main để tập trung vào đề tài mới. Nếu cần viết bài báo NIR, lấy lại từ tag git `xoai-nir`.

## Bối cảnh

- Đề tài do giáo sư giao: *"Nghiên cứu mô hình hỗ trợ ra quyết định điều hành sản xuất theo thời gian thực tích hợp AI giải thích được (XAI-driven DSS) trong chuỗi chế biến nông sản."*
- Ngày 12/10/2026, người làm ban đầu bàn giao dự án cho một đồng nghiệp. Người đang làm việc với Claude là **người nhận bàn giao**, có thể chưa nắm hết chi tiết, nên giải thích rõ khi cần.
- Các hướng đã bỏ vẫn còn trong lịch sử git:
  - lên men penicillin (IndPenSim): commit `14a3857` trở về trước;
  - thu hoạch xoài từ phổ NIR (Anderson 2020): tag `xoai-nir`, gồm Bước 1–2 đã chạy và Bước 3 (ANN, CNN) chưa chạy. Thầy nhận xét dữ liệu chỉ có CSV thì "đề tài yếu", nhưng phần này có thể dùng để viết bài báo.
- Hướng hiện tại theo gợi ý của thầy về dữ liệu UAV (18/10/2026). Ba điểm còn chờ thầy xác nhận nằm ở mục 1 của `docs/BAN_GIAO_SESSION.md`.

## Quy tắc bắt buộc

- **Dữ liệu:** giáo sư chỉ chấp nhận bộ dữ liệu công khai đã được bài báo uy tín công bố hoặc sử dụng.
  - Không tự thu thập, không tự sinh dữ liệu.
  - Không dùng dữ liệu chỉ có trên Kaggle, dữ liệu tổng hợp, hay preprint không có bài báo gốc.
  - Khi đề xuất dữ liệu, ghi bài báo gốc và nơi công bố, và nêu rõ nếu dữ liệu thuộc loại không được chấp nhận.
  - WeedsGalore có bài báo gốc ở hội nghị WACV 2025 (IEEE/CVF). Thầy chưa xác nhận có chấp nhận dữ liệu công bố ở hội nghị không.
- **Dữ liệu đa phương tiện (thầy nhận xét ngày 18/10/2026):** luận văn cần dữ liệu **ảnh hoặc video**, tốt nhất là **UAV**. Dữ liệu chỉ có bảng số thì "đề tài yếu".
- **Không bịa số liệu:** mọi con số trong báo cáo phải lấy từ file kết quả thật trong `results/`, ví dụ `results/weeds_pilot_b8/segmentation_metrics.json`, `go_no_go.json`, `decision_metrics.csv`. Chưa chạy thì ghi "chưa có kết quả".
- **Tính toán nặng:**
  - Không huấn luyện mạng nơ-ron trên máy người dùng; viết notebook Google Colab (GPU T4).
  - Chỉ chạy cục bộ các kiểm tra nhanh (vài giây đến khoảng một phút), và mô tả trước khi chạy.
  - Làm từng bước nhỏ, in kết quả kiểm tra sau mỗi bước. Không tự làm chuỗi bước lớn ngoài yêu cầu.
  - Lý do: người làm trước từng phản đối khi Claude chạy mô phỏng dài trên laptop mà không hỏi.
- **Giới hạn Colab miễn phí:** mỗi phiên GPU T4 dùng được tối đa khoảng **5 tiếng**. Notebook phải:
  - chạy phần tải và tiền xử lý dữ liệu được trên runtime CPU;
  - lưu dữ liệu gọn và checkpoint lên Drive, chạy tiếp được sau khi bị ngắt;
  - dừng an toàn trước giới hạn (hiện đặt 3,5 giờ);
  - đo thời gian một epoch trước khi lên kế hoạch số thí nghiệm.
- **Ngôn ngữ:**
  - Mọi tin nhắn hiển thị cho người dùng viết bằng tiếng Việt, kể cả cập nhật ngắn giữa các bước.
  - Code, tên biến, comment viết bằng tiếng Anh.
  - Văn bản gửi giáo sư viết bằng tiếng Việt, dùng **"Hệ thống hỗ trợ ra quyết định"** (viết đầy đủ, không rút gọn thành "Hệ hỗ trợ").
- **Repo công khai:** không commit file PDF, dữ liệu (`data/`), file xác suất `probs_*.npz`, file zip kết quả, thông tin cá nhân hay khóa API; `.gitignore` đã chặn các file này. Không gửi email hay thông tin cá nhân của người dùng tới dịch vụ bên ngoài.

## Dữ liệu và mã nguồn

- **Dữ liệu:** `data/weedsgalore/weedsgalore-dataset.zip`, 337 MB, CC BY 4.0, tải từ https://doidata.gfz.de/weedsgalore_e_celikkan_2024/.
  - 156 ô ảnh 600×600, 5 dải phổ R, G, B, NIR, RE (uint16), mặt nạ phân vùng 6 lớp.
  - Bài toán 3 lớp gộp mọi loài cỏ thành một lớp, giống bài báo gốc.
  - Giữ nguyên cách chia train/val/test 104/26/26 của tác giả, **không chia lại**.
  - Ghép vị trí qua các ngày bằng `location_blocks`: trong file chia tập val và test, mỗi khối id có ngày giảm dần là một vị trí.
- **Chạy test:** từ thư mục gốc repo, bằng Python của venv:
  `WEEDSGALORE_ZIP=data/weedsgalore/weedsgalore-dataset.zip .venv/Scripts/python -m pytest`
  - **Chỉ cài thư viện vào `.venv/`** (tạo từ `requirements.txt`, phiên bản cố định), không cài vào Python chung của máy.
  - Thư viện mới phải thêm vào `requirements.txt` đúng phiên bản đã dùng.
  - Khi console Windows báo lỗi mã hóa tiếng Việt, đặt `PYTHONIOENCODING=utf-8`.
- **Huấn luyện trên Colab** bằng `notebooks/colab_weeds_pilot.ipynb`.
  - Notebook nhúng sẵn mã nguồn `src/weeds/`. **Sửa `src/weeds/` thì phải chạy lại `scripts/make_weeds_notebook.py`.**
  - Mỗi lần chạy mới dùng thư mục kết quả mới trên Drive, vì `train_done.json` làm notebook bỏ qua huấn luyện. Giải nén kết quả vào `results/weeds_pilot_<tên lần chạy>/`.
  - Thiết lập của mỗi lần chạy (batch, lr, seed) được ghi vào `train_done.json`. Cố định random seed.
- **Tầng quyết định** (`src/weeds/analyze.py`) chỉ dùng numpy và pandas, chạy cục bộ được.
  - Thiết lập chính: ô lưới 200 điểm ảnh (0,5 m), τ = 1%, α = β = 0,1, gộp 3 ngày trước khi phun. **τ = 1% chỉ là minh họa, chưa có nguồn.**
  - Hiệu chỉnh chéo 2 phần theo vị trí trên tập test, vì tập val đã dùng để chọn mô hình.
- **Ảnh xem trước cho thầy:** `scripts/export_weeds_images.py` xuất ảnh JPG của cả 156 ô ảnh vào `data/weedsgalore/anh_xem_truoc/`.
- `.gitattributes` đặt `* -text` để giữ nguyên từng byte của file dữ liệu và mô hình.

## Đóng góp của đề tài

Đóng góp không nằm ở việc phân vùng chính xác hơn bài báo gốc, mà ở **tầng quyết định**:
- quyết định phun / không phun / chưa quyết cho từng ô lưới, có bảo đảm conformal theo lớp về tỉ lệ bỏ sót và tỉ lệ phun nhầm;
- hiệu chỉnh lại khi sang đợt bay mới;
- giải thích từng quyết định và **kiểm chứng độ trung thực của lời giải thích** (che vùng, có đối chứng ngẫu nhiên);
- đánh giá bằng chỉ số ở mức quyết định: giảm thuốc, tỉ lệ chưa quyết, bỏ sót, phun nhầm (theo Melki 2025).

Các bài gần nhất (Celikkan 2023, 2025; Melki 2025) được so sánh ở mục 3 của file bàn giao.

Lưu ý: khi mô hình phân vùng đã tốt, chính sách plug-in đã gần bằng oracle. Giá trị của conformal là giảm bỏ sót, đổi lại phải kiểm tra thêm một số ô; phải trình bày đúng như vậy.
