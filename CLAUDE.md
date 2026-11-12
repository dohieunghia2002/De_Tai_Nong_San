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
- **Đối sánh với bài báo:** `notebooks/colab_weeds_authors.ipynb` (sinh bởi cùng script) đánh giá các mô hình công bố sẵn của tác giả (`ckpts.zip`) bằng `src/weeds/authors.py`. Nạp checkpoint của bên thứ ba luôn dùng `weights_only=True`. Kết quả khớp bài báo; mô hình phân vùng chính của đề tài là mô hình công bố của tác giả.
- **Huấn luyện thêm (seed, lịch huấn luyện):** `notebooks/colab_weeds_runs.ipynb` chạy các lần trong `src/weeds/runs.py`, tự chạy tiếp qua nhiều phiên Colab; kết quả giải nén vào `results/weeds_runs/`.
- **XAI:** `src/weeds/explain.py` giải thích quyết định của từng ô (Integrated Gradients từ ảnh "đất trống", Grad-CAM, mức quan trọng từng dải phổ) và kiểm chứng độ trung thực bằng phép che vùng có đối chứng ngẫu nhiên. Notebook: `notebooks/colab_weeds_xai.ipynb`; kết quả giải nén vào `results/weeds_xai/`.
  - Test chạy cả quy trình mất khoảng 2 phút, chỉ chạy khi đặt `WEEDS_SLOW_TESTS=1`. Phải chạy test này trước khi đưa code XAI đã sửa lên Colab.
  - Đã chạy ngày 09/11/2026. Kết quả phân tích nằm cuối mục 5 của file bàn giao: IG trung thực, Grad-CAM không trung thực ở mức ô lưới, IG không hơn bản đồ xác suất cỏ của chính mô hình.
  - **Giải thích phản thực** (hướng chính từ ngày 10/11): `src/weeds/counterfactual.py` tính "bỏ những vùng cỏ nào thì ô hết phun, thêm bao nhiêu cỏ thì ô phải phun" từ bản đồ xác suất, chạy cục bộ. Kết quả ở `results/weeds_counterfactual/`. Ước lượng trùng với mô hình chấm lại ở 96,5% trường hợp (95,1% nếu bỏ các trường hợp trùng hiển nhiên) và hơi lạc quan ở ô sát ngưỡng. Phần kiểm chứng bằng mô hình nằm ở `src/weeds/counterfactual_check.py` và `notebooks/colab_weeds_cfcheck.ipynb`; kết quả giải nén vào `results/weeds_cfcheck/`. Đã chạy ngày 10/11/2026 (khoảng 1 giây mỗi ô trên T4): ước lượng đúng chính xác k ở 46/104 ô PHUN, lệch không quá 1 vùng ở 78% ô và thường lạc quan. **Lời giải thích đưa vào báo cáo dùng k đã được mô hình kiểm chứng.** Chi tiết ở cuối mục 5 và mục 6, 2b của file bàn giao.
- **Ảnh xem trước cho thầy:** `scripts/export_weeds_images.py` xuất ảnh JPG của cả 156 ô ảnh vào `data/weedsgalore/anh_xem_truoc/`. `scripts/make_pipeline_figure.py` vẽ sơ đồ xử lý trên một ô ảnh test thật. `scripts/make_counterfactual_figure.py` vẽ hình giải thích phản thực cho 3 ô tiêu biểu (`data/weedsgalore/hinh_giai_thich_phan_thuc.png`, matplotlib); ví dụ được chọn theo quy tắc cố định, không chọn bằng mắt; `--simple` vẽ bản gọn cho khổ A4 (`hinh_giai_thich_don_gian.png`).
- **Báo cáo tiến độ gửi thầy:** `docs/BAO_CAO_TIEN_DO.md` chỉ có chữ và được commit. Bản có hình để xuất PDF do `scripts/embed_report_images.py` sinh vào `data/weedsgalore/BAO_CAO_TIEN_DO_co_hinh.md`, không commit vì chứa ảnh của bộ dữ liệu. Sửa chữ ở bản trong `docs/` rồi chạy lại script.
- **Bản demo tiến độ cho thầy:** `demo/app.py` (Streamlit 1.65), chạy bằng `.venv/Scripts/streamlit run demo/app.py` rồi mở http://localhost:8501.
  - **Trang đầu "Chạy thử hệ thống"** (`demo/views/pipeline.py`, phần tính toán ở `demo/pipeline.py`, người dùng chọn ngày 12/11/2026): nhận một ô ảnh 5 dải (chọn từ bộ dữ liệu hoặc upload 5 file PNG 16 bit), chạy mô hình tác giả `dlv3p_msi_3` **trên CPU của laptop** (chỉ suy luận, không huấn luyện), rồi hiện kết quả từng module. Kiểm chứng phản thực bằng mô hình và IG chỉ chạy khi bấm nút, có báo trước thời gian. Trọng số nằm ở `data/weedsgalore/ckpts/` (không commit).
  - Ngưỡng: ô test của 3 đợt bay trước ngày phun dùng ngưỡng của phần vị trí còn lại (đúng như báo cáo); ô khác dùng ngưỡng hiệu chỉnh trên cả 216 ô test.
  - Các trang "Kết quả đánh giá" chỉ đọc `results/` và tính lại quyết định từ bản đồ xác suất bằng `src/weeds`. Phần tính toán nằm ở `demo/core.py` (không dùng Streamlit). Phần vẽ dùng lại hàm của `scripts/make_counterfactual_figure.py`.
  - Thiết lập nào có trong `decision_metrics.csv` thì trang tự đối chiếu với file (`core.saved_difference`); thiết lập khác thì ghi "đang khám phá". Chữ hiển thị không được gõ số cứng: số phải đọc từ file kết quả.
  - Test: `tests/test_demo.py`, `tests/test_demo_pipeline.py`. Test chạy mọi trang chỉ chạy khi đặt `WEEDS_SLOW_TESTS=1`; phải chạy test này sau khi sửa demo.
- **Khi ghi file bằng Python trên Windows, dùng `write_bytes` hoặc `newline='\n'`.** Repo dùng ký tự xuống dòng LF, còn `Path.write_text` mặc định đổi sang CRLF.
- `.gitattributes` đặt `* -text` để giữ nguyên từng byte của file dữ liệu và mô hình.

## Đóng góp của đề tài

Đóng góp không nằm ở việc phân vùng chính xác hơn bài báo gốc, mà ở **tầng quyết định**:
- quyết định phun / không phun / chưa quyết cho từng ô lưới, có bảo đảm conformal theo lớp về tỉ lệ bỏ sót và tỉ lệ phun nhầm;
- hiệu chỉnh lại khi sang đợt bay mới;
- giải thích từng quyết định và **kiểm chứng độ trung thực của lời giải thích** (che vùng, có đối chứng ngẫu nhiên);
- đánh giá bằng chỉ số ở mức quyết định: giảm thuốc, tỉ lệ chưa quyết, bỏ sót, phun nhầm (theo Melki 2025).

Các bài gần nhất (Celikkan 2023, 2025; Melki 2025) được so sánh ở mục 3 của file bàn giao.

Lưu ý: khi mô hình phân vùng đã tốt, chính sách plug-in đã gần bằng oracle. Giá trị của conformal là giảm bỏ sót, đổi lại phải kiểm tra thêm một số ô; phải trình bày đúng như vậy.
