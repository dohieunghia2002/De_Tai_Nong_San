# Bàn giao giữa các phiên làm việc với Claude (cập nhật 14/11/2026)

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
   - **Ngày 05/11, thầy góp ý về input/output:** "output không phải là model quyết định; output phải xác định: kết quả phân lớp / vùng...". Vì vậy output được trình bày thành hai kết quả:
     - output 1: kết quả phân vùng (segmentation) cho từng điểm ảnh;
     - output 2: kết quả phân lớp (classification) cho từng ô 0,5 m: phun / không phun / chưa chắc.
     Người dùng đã gửi thầy sơ đồ xử lý vẽ trên một ô ảnh test thật (`scripts/make_pipeline_figure.py`, ảnh lưu tại `data/weedsgalore/so_do_pipeline.png`) và bài báo gốc.
   - **Yêu cầu của thầy (05/11):** "chạy thử rồi đối sánh với bài báo có dataset đó, xem thế nào; nếu ổn thì viết rồi hẹn vô gặp mình họp chốt".
   - **Ngày 13/11** người dùng gửi thầy kết quả đối sánh, báo cáo PDF và video demo. Thầy trả lời: "triển khai lên Overleaf bài báo luôn", "làm tiếp cho phần này".
   - **Chỉ đạo của thầy (14/11, qua Zalo):**
     - viết **bài báo** khoảng 12–15 trang trên Overleaf để thầy nộp, vì thời gian phản biện rất lâu;
     - trong bài phải có thực nghiệm và **đối sánh kết quả với các nghiên cứu khác**;
     - đề án (luận văn) làm sau: người dùng chưa đăng ký đề cương. Thứ tự là bài báo → nộp → viết đề cương → phát triển dần thành đề án.

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

## 3. Tính mới (đã đọc kỹ ba bài gần nhất; PDF nằm trong `docs/`)

PDF bài WACV 2025 tải ngày 05/11/2026 từ CVF Open Access: `docs/Celikkan_2025_WeedsGalore_WACV.pdf`. Đã đối chiếu: Bảng 3 ghi DeepLabv3+ ảnh 5 dải, 3 lớp, mIoU 82,90; phần cài đặt ghi batch 8, Adam lr 0,001.

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
- **Ngưỡng phun:** đã tra cứu ngày 14/11, xem mục 3.2. **Ngưỡng τ hiện tại (1%) vẫn chỉ là minh họa.**

### 3.1. Tra cứu tính mới của giải thích phản thực (14/11/2026)

Cách tra: công cụ tìm kiếm web, chủ yếu đọc tóm tắt. **Chưa tra Google Scholar / Scopus**, nên khi viết bài dùng "theo hiểu biết của chúng tôi".

**Đã có người làm, phải trích dẫn, không được nhận là mới:**
- **Xóa tập vùng ảnh nhỏ nhất đến khi kết quả đổi.** Câu "bỏ k vùng 5 cm thì ô hết PHUN" của đề tài thuộc họ này:
  - SEDC: Vermeire và cộng sự, *Pattern Analysis and Applications* 2022, doi:10.1007/s10044-021-01055-y, arXiv:2004.07511. Xóa tập phân đoạn ảnh nhỏ nhất làm đổi lớp dự đoán;
  - Fong & Vedaldi, ICCV 2017 ("meaningful perturbation");
  - Dabkowski & Gal, NeurIPS 2017.
- **Phản thực cho tập dự đoán conformal:** Maalej, Sönströd & Johansson, COPA 2025, PMLR 266:405–424. Dữ liệu bảng, phân loại; tóm tắt không nói đến trạng thái từ chối / chưa quyết.
- **Phản thực cho quyết định từ chối (reject option):** Artelt, Brinkrolf, Visser & Hammer, IJCCI 2022 (bộ phân loại LVQ); Artelt, Visser & Hammer, ESANN 2022 (không phụ thuộc mô hình).
- **Phản thực cho mô hình phân vùng ảnh:** OCTET, Zemni và cộng sự, CVPR 2023 (ảnh xe tự lái).
- **Phản thực trong nông nghiệp:** Morales & Sheppard, IJCNN 2023 (đường cong đáp ứng phân bón, dữ liệu bảng).

**Chưa tìm thấy, có thể nêu là đóng góp:**
- phản thực cho **quyết định phun cỏ** từ ảnh UAV (các bài XAI về tìm cỏ hầu như chỉ dùng Grad-CAM);
- tổ hợp: quyết định 3 mức có bảo đảm conformal + phản thực ở mức quyết định với phép xóa có nghĩa ngoài ruộng (thay cỏ bằng đất) + ước lượng nhanh từ bản đồ xác suất, có mô hình kiểm chứng + chỉ chỗ cần ra xem cho ô CHƯA CHẮC.

**Cách viết đề nghị:** "áp dụng giải thích phản thực dạng xóa vùng (như SEDC) cho quyết định phun 3 mức có bảo đảm conformal…", không viết "đề xuất phương pháp phản thực mới".

### 3.2. Tra cứu ngưỡng phun và kích thước ô (14/11/2026)

| Bài | Đã đọc | Cây | Ngưỡng phun | Ô / vùng phun |
|---|---|---|---|---|
| Allmendinger, Spaeth, Saile, Peteinatos & Gerhards, *Plants* 13(15):2164, 2024, doi:10.3390/plants13152164 | toàn văn (PMC11314123) | ngô | **0** (thấy cỏ là phun): "Due to the low competitive ability of maize to suppress weeds [2], a zero weed threshold was used" | béc cách 0,5 m, vùng phun nhỏ nhất 1 m × 1 m (0,5 m trước và sau mỗi cây cỏ). Ảnh 1,4 mm/điểm ảnh. Phun sớm tiết kiệm 47% thuốc, năng suất không giảm có ý nghĩa |
| Sapkota, Stenger, Ostlie & Flores, *Sci Rep* 13:6548, 2023 | bản arXiv v1 (2301.07519) | ngô | ô có ít nhất 1 cây cỏ thì phun | ô 1,67 × 10 ft, vệt phun mỗi béc 0,5 m; tiết kiệm 26% diện tích |
| Longchamps, Panneton, Simard & Leroux, *Weed Science* 62(1):177–185, 2014 | tóm tắt | (tóm tắt không ghi) | ngưỡng kinh tế **0,06–0,31% diện tích cỏ**, do chuyên gia tư vấn đánh giá hơn 2.500 ảnh | ảnh 2 m × 3 m |
| San Martín, Andújar, Barroso, Fernández-Quintanilla & Dorado, *Weed Technology* 30:888–897, 2016, doi:10.1614/WT-D-16-00039.1 | toàn văn | ngô (Tây Ban Nha) | 6 và 12 cây/m² (theo các ngưỡng đã công bố) | ô 0,75 × 0,5 m; mô phỏng máy phun béc cách 0,5 m |
| Peña và cộng sự, *PLoS ONE* 8(10):e77151, 2013 | toàn văn | ngô (UAV, 2 cm/điểm ảnh) | **không chốt ngưỡng**; chia mức cỏ: không cỏ, <5%, 5–20%, >20% | 1 m × khoảng cách hàng (0,7 m); "lưới và ngưỡng tùy theo máy phun" |
| López-Granados và cộng sự, *Precision Agriculture* 2016, doi:10.1007/s11119-015-9415-8 | đoạn trích tìm kiếm | hướng dương | thử 7 ngưỡng 0–15%, bước 2,5% | – |
| Nikolić, Pappalardo, De Marchi & Masin 2025, "From detection to action…" | đoạn trích tìm kiếm (toàn văn bị chặn) | ngô (đông bắc Ý) | thử >1%, >5%, >10% điểm ảnh là cỏ | 0,25 m², 2 m², 3 m². **Chưa xác nhận tạp chí** |
| Ali, Streibig, Christensen & Andreasen, *Weed Research* 55:26–33, 2015, "Image-based thresholds for weeds in maize fields" | chưa đọc | ngô | – | – |

**Ý chính rút ra từ San Martín 2016 (toàn văn):**
- ngưỡng kinh tế nói chung rất thấp (trích Longchamps 2014);
- với loài cỏ sinh sản mạnh hoặc kháng thuốc nên dùng ngưỡng 0 (trích Norsworthy 2014);
- tăng ngưỡng thì tiết kiệm thuốc hơn nhưng bỏ sót nhiều hơn.

**Kết luận:**
- Với ngô, tài liệu nghiêng về ngưỡng rất thấp: 0, hoặc 0,06–0,31% diện tích cỏ. τ = 1% là mức lỏng; nguồn ủng hộ duy nhất (Nikolić 2025) chưa xác nhận được.
- `results/weeds_authors/dlv3p_msi_3/decision_metrics.csv` đã có τ = 0,1%, 0,5%, 1%, 2% (ô 100 và 200 điểm ảnh). τ = 0,1% nằm trong khoảng của Longchamps. **τ = 0 chưa tính.**
- Ngưỡng của Longchamps đo trên ảnh 2 m × 3 m; trên ô 0,25 m², 0,1% chỉ là khoảng 2,5 cm² cỏ, nên không chuyển thẳng ngưỡng giữa hai kích thước ô được.
- Ô 0,5 m khớp khoảng cách béc phun 0,5 m (Allmendinger 2024, Sapkota 2023, San Martín 2016). Theo chiều xe chạy, máy thật cần khoảng 1 m vì độ trễ đóng mở béc (Allmendinger 2024).
- **Đã quyết (người dùng, 14/11):** τ chính = 0,1%, báo cáo thêm độ nhạy theo 7 mức τ (mục 3.3).

### 3.3. Đổi τ chính sang 0,1% và phân tích độ nhạy (14/11/2026)

**Code:**
- `decision.needs_spray`: τ = 0 (`ANY_WEED`) nghĩa là "có cỏ là phun", tức `w > 0`; định nghĩa cũ `w >= 0` sẽ coi cả ô không có cỏ là cần phun.
- `analyze.TAUS` = (0; 0,06%; 0,1%; 0,31%; 0,5%; 1%; 2%), có ghi nguồn trong comment. `PRIMARY['tau']` = 0,001.
- `analyze.main` ghi file bằng LF trên mọi hệ điều hành.

**Kết quả đã tính lại trên máy** (chỉ numpy và pandas, từ bản đồ xác suất đã lưu):
- `decision_metrics.csv` và `go_no_go.json` của 4 thư mục có `probs_*.npz` (2 mô hình tác giả, 2 lần chạy thử). Các dòng của 4 mức τ cũ trùng file cũ (sai khác ≤ 4,4·10⁻¹⁶).
- **Sự cố ngày 14/11 (đã sửa):** `go_no_go.json` tính lại từ `decision_metrics.csv` đọc bằng pandas ghi `lam_lo`/`lam_hi` thành chữ, ví dụ `"(0.0023, 0.0015)"`, vì CSV lưu tuple thành chữ. Notebook XAI vì vậy dừng ở `decision_check` với lỗi numpy `ufunc 'subtract' … <U44`.
  - Cách sửa: `analyze.per_fold` đổi chữ về danh sách số (kể cả `inf`); 4 thư mục có `probs_*.npz` được ghi lại bằng lệnh `analyze`.
  - Hai test mới bắt lỗi này: `test_go_no_go_reads_the_thresholds_back_from_csv`, và `test_go_no_go_is_what_the_xai_notebook_checks_against` (chạy đúng phép đối chiếu của notebook trên file thật).
- 12 lần tự huấn luyện không có `probs_*.npz` trên máy, nên không có các mức τ mới. `go_no_go.json` và `runs_summary.csv` được tính lại ở τ = 0,1% từ `decision_metrics.csv` có sẵn; các cột khác của `runs_summary.csv` giữ nguyên. (Từ 15/11 đã thay bằng kết quả tính lại trên Drive với điểm s mới, đủ 7 mức τ: mục 3.7.)
- **Tiêu chí C2 đổi (người dùng yêu cầu, 14/11):** từ "giảm ≥ 30%" (đặt cho τ = 1%) sang "giảm ít nhất 60% mức giảm tối đa có thể (oracle)". Hệ số 0,6 = 30% / 50%, tức mức tối đa ở τ = 1%, nên ở τ = 1% hai tiêu chí giống hệt nhau. Ở τ = 0,1%, mức tối đa là 10,2% nên yêu cầu là ≥ 6,11%.
  - Lý do: ở τ = 0,1% không mô hình nào đạt được mức cũ, kể cả khi biết trước đáp án.
  - Kết luận GO/NO-GO của cả 16 thư mục không đổi so với τ = 1% với tiêu chí cũ: 14 GO; 2 NO-GO vì C1 (lần chạy thử 1, và es_seed1 ở epoch cuối).
  - Trang Tổng quan của demo vẫn gọi đây là "tiêu chí đề ra trước khi chạy thử": cần ghi chú việc đổi C2 khi cập nhật demo.

**Bảng độ nhạy, mô hình chính** (5 dải; 216 ô, ô 50 cm, gộp 3 đợt bay, hiệu chỉnh chéo). DT = đề tài (conformal 3 mức), NCĐ = ngưỡng cố định; số trong ngoặc là số ô:

| τ | Cần / không cần phun | Giảm tối đa | DT: giảm / CHƯA CHẮC / bỏ sót / phun nhầm | NCĐ: bỏ sót / phun nhầm |
|---|---|---|---|---|
| có cỏ là phun | 209 / 7 | 3,2% | 13,9% / 186 / **11,0% (23)** / 0% (0) | 0% / 100% (7) |
| 0,06% | 201 / 15 | 6,9% | 15,7% / 86 / 9,5% (19) / 0% (0) | 0,5% / 26,7% (4) |
| **0,1%** | 194 / 22 | 10,2% | 15,7% / 2 / 6,7% (13) / 4,5% (1) | 2,1% / 13,6% (3) |
| 0,31% | 164 / 52 | 24,1% | 29,2% / 21 / 8,5% (14) / 5,8% (3) | 4,3% / 17,3% (9) |
| 0,5% | 144 / 72 | 33,3% | 37,5% / 11 / 9,0% (13) / 5,6% (4) | 3,5% / 9,7% (7) |
| 1% | 108 / 108 | 50,0% | 49,5% / 9 / 3,7% (4) / 4,6% (5) | 3,7% / 7,4% (8) |
| 2% | 75 / 141 | 65,3% | 65,7% / 21 / 1,3% (1) / 0% (0) | 1,3% / 7,8% (11) |

- **Mô hình RGB:** đề tài giữ cả hai tỉ lệ ≤ 10% ở mọi mức τ; ở 0,1% và 0,06% phải đẩy 47 và 123 ô sang CHƯA CHẮC. Ngưỡng cố định vượt giới hạn ở 5/7 mức: phun nhầm 100% ở 2 mức thấp nhất, 50% ở 0,1%; bỏ sót 21,3%, 20,8%, 16,7% ở 0,31%, 0,5%, 1%.
- **Mọi mô hình có kết quả** (4 mô hình ở mức mới, 16 ở mức cũ), số mô hình có bỏ sót hoặc phun nhầm vượt 10%:
  - đề tài: 2/4, 1/4, 3/16, 1/4, 3/16, 1/16, 0/16 (theo thứ tự τ ở bảng trên); mức vượt lớn nhất 13,9%;
  - ngưỡng cố định: 4/4, 4/4, 16/16, 4/4, 7/16, 5/16, 2/16; mức vượt lớn nhất 100%.
- **Lưu ý khi viết:**
  - Ở τ thấp, "giảm" của đề tài lớn hơn mức tối đa là do bỏ sót, không phải tiết kiệm thật.
  - Ở τ = 0,1% chỉ có 22 ô không cần phun, nên 1 ô = 4,5 điểm phần trăm; các trường hợp vượt (13,6% = 3/22 ô) chỉ cách giới hạn 1 ô.
  - "Có cỏ là phun" và 0,06% gần như không còn bài toán ở ruộng này: hầu hết ô cần phun, và bảo đảm phải đổi bằng rất nhiều ô CHƯA CHẮC.
- **Ước lượng phản thực nhanh ở τ = 0,1%** (trên máy, từ bản đồ float16, chưa kiểm chứng bằng mô hình):
  - quyết định 180 PHUN / 2 CHƯA CHẮC / 34 KHÔNG PHUN (ở 1% là 104 / 9 / 103);
  - ngưỡng phun 4,2–5,9 cm² (ở 1% là 24,8–33,0 cm²);
  - số vùng cần bỏ để hết PHUN: trung vị 11 (tứ phân vị 5–22, lớn nhất 57), ở 1% là 9 (5–21, 48); 11 ô "mong manh" ở cả hai mức.

### 3.4. Kết quả XAI và phản thực ở τ = 0,1% (15/11/2026)

Người dùng chạy `colab_weeds_xai.ipynb` (thư mục Drive `xai_authors_msi3_tau0.001`). Kết quả được giải nén vào `results/weeds_xai/`; bản τ = 1% chép sang `data/ket_qua_cu_tau1/` (không commit).

- **Đối chiếu quyết định:** 180 PHUN / 34 KHÔNG PHUN / 2 CHƯA CHẮC. Khớp `go_no_go.json` (sai khác 0,0); chấm lại không đổi ô nào.
- **Độ đầy đủ của IG:** gần như không đổi so với τ = 1% (sai số tương đối trung vị 0,26%), vì IG tính theo ô, không phụ thuộc τ.
- **Kiểm chứng độ trung thực trên 182 ô PHUN và CHƯA CHẮC** (ở τ = 1% là 113 ô). Mức che 1% số vùng của ô ảnh, tức 9 vùng = 9% diện tích một ô, là mức dùng trong báo cáo:

| Cách chọn vùng để xóa | s giảm (τ 0,1% / 1%) | Ô PHUN bị đổi (0,1% / 1%) | Thắng ngẫu nhiên trong ô (0,1% / 1%) |
|---|---|---|---|
| IG | 62,5% / 63,5% | 30,6% / 45,2% | 99,5% (181/182) / 100% |
| Bản đồ xác suất cỏ của mô hình | 68,6% / 66,6% | 40,6% / 46,2% | 99,5% / 100% |
| Grad-CAM | 5,2% / 9,3% | 1,1% / 3,8% | 49,5% / 45,1% |
| Ngẫu nhiên trong ô | −1,6% / 4,7% | 0,7% / 3,3% | – |

  - **Kết luận giữ nguyên so với τ = 1%:** IG trung thực (thắng ngẫu nhiên cả ô ảnh ở 179/182 ô); Grad-CAM không giải thích được từng ô (khoảng 50%, như tung đồng xu); IG không hơn bản đồ xác suất cỏ của chính mô hình.
  - Tỉ lệ ô PHUN bị đổi thấp hơn vì ngưỡng phun thấp hơn, nên phải xóa gần hết cỏ của ô mới hết PHUN.
- **Phản thực, ước lượng nhanh** (`results/weeds_counterfactual/`, chạy trên máy 0,08 giây; module giờ ghi LF):
  - 180 ô PHUN: số vùng cần bỏ để hết PHUN có trung vị 11 (tứ phân vị 5–22, lớn nhất 57); 11 ô chỉ cần 1 vùng, 44 ô tối đa 4 vùng.
  - 2 ô CHƯA CHẮC: bỏ 1 vùng là KHÔNG PHUN.
  - 34 ô KHÔNG PHUN: thêm 1,4–5,6 cm² cỏ dự đoán (trung vị 3,3) là thành PHUN.
- **Ước lượng nhanh khớp mô hình kém hơn hẳn:**
  - quyết định trùng ở 84,9% trường hợp (78,2% nếu chỉ tính trường hợp quyết định có đổi), so với 96,5% và 95,1% ở τ = 1%;
  - 129/137 trường hợp lệch là ước lượng lạc quan;
  - lý do: ngưỡng phun chỉ khoảng 4–6 cm² cỏ dự đoán, trong khi sai số trung vị của ước lượng 1–2 cm².
  - **Lời giải thích trong bài phải dùng k đã được mô hình kiểm chứng** (bước 3 ở mục 6, 3b).

### 3.5. Kiểm chứng phản thực bằng mô hình ở τ = 0,1% (15/11/2026)

Người dùng chạy `colab_weeds_cfcheck.ipynb` (thư mục Drive `cfcheck_authors_msi3_tau0.001`; 5 251 lượt chạy mô hình, 4,9 phút, 1,6 giây mỗi ô). Kết quả ở `results/weeds_cfcheck/`; bản τ = 1% ở `data/ket_qua_cu_tau1/`. Mã băm `decisions.csv` và giá trị đất khớp kết quả XAI; 182 ô; chấm lại không đổi quyết định ô nào.

| | τ = 0,1% | τ = 1% |
|---|---|---|
| Ô PHUN: k (bỏ đến vùng thứ mấy thì hết PHUN) theo mô hình = ước lượng | 59/180 (33%) | 46/104 (44%) |
| Ô PHUN: lệch không quá 1 vùng | 97/180 (54%) | 81/104 (78%) |
| Ô PHUN "mong manh" (ước lượng ≤ 2 vùng): đúng chính xác / lệch ≤ 1 | 16/22 / 20/22 | 14/18 / 18/18 |
| Ô mô hình cần nhiều hơn ước lượng từ 20 vùng trở lên | 11 | 0 |
| Ô bỏ hết 100 vùng vẫn chưa KHÔNG PHUN | 2 | 0 |
| Ô quyết định đổi ngược lại khi bỏ thêm vùng (về PHUN / khỏi KHÔNG PHUN) | 30 / 37 | 5 / 5 |
| Từng bước bỏ vùng: quyết định của ước lượng trùng mô hình | 68,0% | 91,7% |

**Nguyên nhân: ngưỡng nằm sát "mức nền" của mô hình.**
- Thay toàn bộ ô bằng đất trống (7 ô quét tới k = 100), mô hình vẫn chấm 3,1 cm² cỏ dự đoán (trung vị; lớn nhất 5,5 cm²). Đó là tổng các xác suất cỏ rất nhỏ trên 40 000 điểm ảnh.
- Ngưỡng phun ở τ = 0,1% là 4,2–5,9 cm², ngưỡng không phun 3,75–4,34 cm².
- Vì vậy quyết định nằm ở mức nhiễu của mô hình: có ô (`2023-05-30_0099` ô 3, ô 4, ô 7; `2023-05-25_0681` ô 1; `2023-05-25_0109` ô 1) ước lượng chỉ cần 3–8 vùng mà mô hình cần 89–100 vùng.
- Sai số tuyệt đối của ước lượng không đổi (trung vị 3,6 cm², ở τ = 1% là 3,9 cm²); chỉ là so với ngưỡng nhỏ thì nó thành lớn.

**Ý nghĩa khi viết bài:**
- Bảo đảm conformal vẫn giữ (mục 3.3), nhưng lời giải thích "bỏ k vùng" kém ổn định hơn ở τ thấp.
- Với các ô PHUN mong manh, lời giải thích vẫn đáng tin (20/22 lệch ≤ 1 vùng).
- Đây là hạn chế của cách tính điểm s bằng trung bình xác suất: ngưỡng phải cao hơn mức nền của mô hình ở kích thước ô. Một hướng sửa (chưa làm): trừ mức nền đất trống khỏi s, hoặc tính diện tích cỏ từ lớp dự đoán (argmax) thay vì xác suất.

### 3.6. Thử điểm s tính từ nhãn dự đoán (15/11/2026; mới thử, chưa đổi code)

**Điểm thử:** s_hard = tỉ lệ điểm ảnh của ô mà mô hình xếp vào lớp cỏ (argmax). Trừ một hằng số khỏi s không có tác dụng: ngưỡng conformal dịch theo đúng hằng số đó, nên quyết định giữ nguyên.

**Tài liệu:**
- Popordanoska, Bertels, Vandermeulen, Maes & Blaschko, MICCAI 2021 (arXiv:2112.12560): trung bình xác suất từng điểm ảnh cho diện tích (thể tích) không lệch khi mô hình được hiệu chỉnh tốt trên từng ảnh; độ lệch bị chặn bởi sai số hiệu chỉnh. Mức nền của s là dấu hiệu hiệu chỉnh chưa tốt ở diện tích rất nhỏ.
- Bản đồ phun từ phân vùng cỏ thường tính độ phủ cỏ của ô bằng tỉ lệ điểm ảnh được xếp vào lớp cỏ: Huang và cộng sự, *Sensors* 18(10):3299, 2018 (ngưỡng độ phủ 0–0,25, theo đoạn tóm tắt); Nikolić 2025 (mục 3.2, chưa xác nhận tạp chí).

**Tầng quyết định** (từ bản đồ đã lưu, trên máy):
- Ô không có cỏ theo nhãn (7 ô): điểm cũ trung vị 1,02 cm² (95% số ô dưới 3,07); điểm mới 0,00 cm² (95% dưới 0,40).
- Mô hình chính, τ = 0,1%:
  - đề tài: giảm 16,2%, 7 ô CHƯA CHẮC, bỏ sót 7,2%, phun nhầm 4,5% (điểm cũ: 15,7% / 2 / 6,7% / 4,5%);
  - ngưỡng cố định: bỏ sót 6,2%, phun nhầm 4,5% (điểm cũ: 2,1% / 13,6%).
- Số mô hình (trên 4) vượt giới hạn 10%, theo 7 mức τ (0; 0,06%; 0,1%; 0,31%; 0,5%; 1%; 2%):
  - đề tài, điểm mới: 2, 0, 0, 0, 2, 0, 0 (điểm cũ: 2, 1, 1, 1, 1, 1, 0);
  - ngưỡng cố định, điểm mới: 4, 2, 2, 3, 3, 2, 0 (điểm cũ: 4, 4, 4, 4, 2, 2, 1).
- RGB, τ = 0,1%: đề tài bỏ sót 0% nhưng 36,1% ô CHƯA CHẮC (điểm cũ 21,8%); ngưỡng cố định bỏ sót 28,9%.

**Phản thực, ô `2023-05-30_0099` ô 3** (mô hình chạy trên CPU):
- điểm mới 14,0 cm², ngưỡng 3,62 / 4,38 cm²;
- ước lượng và mô hình cùng thành KHÔNG PHUN ở k = 2 (điểm cũ: ước lượng 3, mô hình 96);
- nhưng ở k = 4 mô hình lại PHUN (4,50 cm²), và khi đã bỏ hết 8 vùng có cỏ, mô hình vẫn thấy 2,38 cm² điểm ảnh cỏ mới, có thể là vết mép của các ô vuông đất trống dùng để che.

### 3.7. Đã đổi điểm s sang tỉ lệ điểm ảnh cỏ (15/11/2026, người dùng chọn phương án A)

**Code:**
- `decision.predicted_weed(p_crop, p_weed)`: lớp có xác suất cao nhất là cỏ (argmax; bằng nhau thì lớp đứng trước), sau khi làm tròn hai xác suất về float16 như `train.evaluate` lưu. `cell_table(ids, y3, p_crop, p_weed, cell)`.
- `analyze.SCORE = 'argmax'`, `PRIMARY['score']`, cột `score` trong `decision_metrics.csv`. `go_no_go` và `runs.decision_row` từ chối bảng không có cột này (điểm cũ). Thêm `analyze.write_results`.
- `explain.py` (XAI_VERSION 3): `weed_pixels`, `cell_fractions`, `cell_scores` (điểm s, float64; khớp từng bit với `cell_table` trên CPU, còn trên GPU torch chia bằng cách nhân với 1/cell² nên có thể lệch 1 đơn vị ở chữ số cuối, vì vậy các phép so dùng `decision.TIE`), `cell_prob_scores` (P(cỏ) trung bình cho IG). Phép xóa vùng và mức quan trọng dải đo bằng s; cách xếp hạng `predicted_weed` = số điểm ảnh cỏ × 1024 + khối P(cỏ). Cột mới `p_model` trong `xai_cells.csv`.
- `counterfactual.py`: xếp hạng vùng theo số điểm ảnh cỏ, bằng nhau thì theo P(cỏ) (`ranked_patches(..., tie=)`); ước lượng = trừ số điểm ảnh; module ghi LF.
- `counterfactual_check.py` (CHECK_VERSION 2): `weed_map` trả về điểm ảnh cỏ và P(cỏ); chấm lại bằng s.
- Demo (`ui.cells`, `pipeline.decide_tile/explain_estimate/estimate_steps`), `make_pipeline_figure.py`, `make_counterfactual_figure.py`: phần tính toán dùng điểm mới (chữ hiển thị chưa sửa).
- Notebook: thư mục Drive `xai_authors_msi3_tau0.001_argmax`, `cfcheck_authors_msi3_tau0.001_argmax`; notebook mới `colab_weeds_runs_rescore.ipynb` (CPU; metadata không xin GPU) tính lại quyết định của 12 mô hình tự huấn luyện (6 lần chạy, mỗi lần có mô hình chọn theo val và mô hình epoch cuối) trên Drive, gói `ket_qua_weeds_runs_argmax.zip`. Review ngày 15/11: notebook dừng ở mục 2, trước khi ghi gì lên Drive, nếu thiếu kết quả hay bản đồ xác suất của bất kỳ mô hình nào (ví dụ mở bằng tài khoản Google khác; phiên 1 và lần chạy 2 nằm trên Drive khác), và không ghi `runs_summary.csv` thiếu dòng. Đã chạy thử code của notebook trên một Drive giả: mỗi file tính lại khớp từng byte với `analyze` trên máy cho cùng bản đồ xác suất, zip có đúng 55 tên file như `results/weeds_runs/`, xuống dòng LF.

**Kiểm tra** (cập nhật 15/11 sau khi chạy XAI và phản thực: 105 test đạt, kể cả test chậm; 4 test dưới đây nay đạt). Lúc đổi điểm: 93 test nhanh đạt; 4 test lỗi đúng dự kiến, vì so với kết quả XAI, phản thực đang lưu bằng điểm cũ. Test chậm đạt: XAI toàn quy trình, mọi trang demo, upload. Test upload nay tự chọn ảnh "ngoài bộ dữ liệu" theo quy tắc cố định: ô val đầu tiên có đủ 3 quyết định, đổi tên. Mô hình chạy trên CPU khác bản đồ Colab ở 2–10 trên 360 000 điểm ảnh; điểm ô lệch tối đa 0,125 cm².

**Kết quả quyết định mới (τ = 0,1%, tính trên máy):**
- 5 dải: GO, giảm 16,2%, CHƯA CHẮC 3,2% (7 ô), bỏ sót 7,2%, phun nhầm 4,5%.
- RGB: NO-GO vì C3: CHƯA CHẮC 36,1%, bỏ sót 0%, phun nhầm 4,5%, giảm 9,7%.
- Lần chạy thử 1: NO-GO (C1, và CHƯA CHẮC 31,9%). Lần chạy thử 2: GO, giảm 17,1%, CHƯA CHẮC 0,9%, bỏ sót 8,2%, phun nhầm 4,5%.
- **12 mô hình tự huấn luyện** (tính lại trên Drive bằng `colab_weeds_runs_rescore.ipynb`, 15/11, giải nén vào `results/weeds_runs/`). Kiểm tra trước khi giải nén: 30 file json huấn luyện giống từng byte bản trên máy (đúng các mô hình của phiên 2), mỗi `go_no_go.json` khớp kết luận tính lại từ `decision_metrics.csv` của nó, `runs_summary.csv` giữ nguyên các cột huấn luyện.
  - Kết luận không đổi ở cả 12 dòng: 11 GO, 1 NO-GO (`es_seed1` epoch cuối, vì C1: mIoU 66,93).
  - 11 mô hình khác nhau (10 mô hình của `runs_summary.csv` và lần chạy 2): 10 GO; giảm 13,4–19,0%; CHƯA CHẮC 0–6,9%; phun nhầm 0–9,1% (0–2 trên 22 ô không cần phun); bỏ sót 3,6–10,3% (7–20 trên 194 ô cần phun).
  - `es_seed2` (chọn theo val) bỏ sót 20 ô, hơn mức α × 194 = 19,4 ô đúng một ô. Bảo đảm conformal chỉ đúng theo kỳ vọng, nên một lần đo trên 194 ô có thể vượt; phải báo cáo đúng như vậy. Các ô KHÔNG PHUN của mọi mô hình chỉ chứa 0,35–0,84% tổng diện tích cỏ (`missed_weed_area`).
  - So với điểm cũ (cùng τ = 0,1%, 12 dòng), điểm mới thay đổi như sau. Phun nhầm 4,5–13,6% thành 0–9,1%; hai mô hình phun nhầm 13,6% (3/22 ô, vượt β) nay còn 4,5%. CHƯA CHẮC 0,5–26,4% thành 0–6,9%; `e100_seed2` (chọn theo val) giảm từ 26,4% xuống 3,2%. Bỏ sót 6,2–9,8% thành 3,6–10,3%; giảm 14,4–18,5% thành 13,4–19,0%.
- **XAI** (Colab 15/11, thư mục Drive `xai_authors_msi3_tau0.001_argmax`, kết quả ở `results/weeds_xai/`):
  - Quyết định 174 PHUN, 35 KHÔNG PHUN, 7 CHƯA CHẮC, khớp tuyệt đối kết quả đã báo cáo; 181 ô được kiểm chứng.
  - Kết luận như trước:
    - IG và điểm ảnh cỏ của mô hình làm s giảm mạnh hơn đối chứng ngẫu nhiên ở 95–100% số ô.
    - Grad-CAM không hơn ngẫu nhiên trong ô (chỉ thắng `random_cell` ở 30–55% số ô).
    - IG không hơn điểm ảnh cỏ của chính mô hình: ở mọi mức xóa, s giảm ít hơn.
  - Số cụ thể ở `xai_summary.json`. Bản điểm cũ (gồm cả `xai_maps.npz`) lưu ở `data/ket_qua_cu_tau0.001_diem_cu/` (không commit).
- **Giải thích phản thực** (`results/weeds_counterfactual/`, chạy trên máy) so với điểm cũ:
  - ước lượng khớp mô hình chấm lại 92,3% (điểm cũ 84,9%);
  - chỉ tính các dòng có đổi quyết định: 89,2% (78,2%);
  - kết luận hết PHUN khớp 95,4% (88,8%);
  - sai số trung vị khi xóa 10% vùng: 0,06 cm² (0,82 cm²).
- **Kiểm chứng phản thực bằng mô hình** (Colab 16/11, thư mục Drive `cfcheck_authors_msi3_tau0.001_argmax`, kết quả ở `results/weeds_cfcheck/`; 181 ô, 4343 lượt chạy mô hình, 3,6 phút), so với điểm cũ (182 ô):
  - k hết PHUN (174 ô PHUN): ước lượng đúng 68 ô (39,1%; điểm cũ 32,8%), lệch không quá 1 vùng ở 114 ô (65,5%; 53,9%). Mô hình cần nhiều vùng hơn ước lượng ở 56 ô (32,2%; 46,7%), ít hơn ở 50 ô.
  - Ô PHUN mong manh (ước lượng ≤ 2 vùng): đúng 20/25 (80%), lệch không quá 1 vùng 23/25.
  - k thành KHÔNG PHUN: đúng 64/177 (36,2%), lệch không quá 1 vùng 113/177 (63,8%); 4 ô đã bỏ hết vùng vẫn không thành KHÔNG PHUN.
  - 30 ô PHUN (17,2%) lại thành PHUN khi bỏ thêm vùng sau lúc đã hết PHUN; 24 ô rời KHÔNG PHUN.
  - Kết luận giữ nguyên: ước lượng nhanh chỉ gần đúng vì mô hình nhìn cả ngữ cảnh quanh vùng bị bỏ, nên lời giải thích trong báo cáo dùng k đã được mô hình kiểm chứng. Điểm mới làm ước lượng khớp hơn.
  - Kiểm tra trước khi giải nén:
    - cấu hình đúng, kể cả mã băm `decisions.csv` của lần chạy XAI;
    - bảng tóm tắt tính lại từ các file csv khớp;
    - không ô nào đổi quyết định ở k = 0;
    - điểm ở k = 0 trùng từng điểm ảnh với điểm XAI chấm lại. Hai bên chỉ lệch khoảng 1e-17 vì XAI chia trên GPU bằng cách nhân với 1/40000, còn cfcheck chia bằng numpy;
    - so với bản đồ đã lưu: 16 ô lệch 1–2 điểm ảnh (đã biết từ XAI), 1 ô khác k ước lượng, 2 ô khác thứ tự vùng, đều nằm trong 16 ô đó.
  - Bản điểm cũ lưu ở `data/ket_qua_cu_tau0.001_diem_cu/weeds_cfcheck/`.
- **Hai lỗi do đổi sang điểm đếm điểm ảnh, sửa ngày 15/11.** Lần review notebook XAI trước đó đã bỏ sót cả hai. Điểm và ngưỡng có thể bằng nhau đúng (hòa), và sai số làm tròn lại quyết định trường hợp hòa:
  - (a) `explain.faithfulness` tính "thắng đối chứng" khi hòa tùy theo làm tròn. Tính lại trên máy từ csv thì lệch 1/181 ô ở Grad-CAM, mức xóa 0,5%. Nay hòa không tính là thắng. Summary của lần chạy Colab không đổi.
  - (b) `decide_conformal` so điểm trên đường trừ dần vùng bị bỏ với ngưỡng đọc từ csv, nên 7/216 ô có k phản thực lệch một vùng. Nay dùng `decision.TIE` = 1e-9.
  - Kiểm chứng: `decision_metrics.csv` và `go_no_go.json` của 4 thư mục có bản đồ xác suất giống từng byte; phản thực trùng với phép tính đúng tuyệt đối trên lưới điểm ảnh; các test mới bắt được cả hai lỗi trên code cũ.
- **Plug-in ở τ = "có cỏ là phun" (sửa 16/11, người dùng đồng ý):** trước đó dùng `s >= 0` nên phun mọi ô, giống phun toàn ruộng; nay phun khi mô hình thấy ít nhất một điểm ảnh cỏ (`s > 0`), như cách định nghĩa "cần phun" ở mức này. Chỉ các dòng plug-in ở τ = 0 đổi; mọi dòng khác và `go_no_go.json` giữ nguyên từng byte. Đã tính lại 4 thư mục có bản đồ xác suất. Ví dụ mô hình 5 dải của tác giả, thiết lập chính: phun 95,8%, giảm 4,2%, bỏ sót 1,9%, phun nhầm 28,6%. 12 thư mục tự huấn luyện chờ chạy lại `colab_weeds_runs_rescore.ipynb` (notebook nay xóa zip cũ trên Drive trước khi tính).

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

- `src/weeds/authors.py`: đánh giá mô hình công bố sẵn của tác giả bằng code của đề tài.
  - Nguồn: `ckpts.zip` (887 MB) trên trang dữ liệu, chứa 6 mô hình. Module dùng 4 mô hình thường: ảnh 5 dải và RGB, bài toán 3 lớp và 6 lớp. Hai mô hình MC dropout (`dlv3p_do_*`) chưa dùng.
  - Cách nạp: mạng của tác giả là DeepLabV3Plus theo VainF, chỉ khác tên tham số. Module đổi tên rồi nạp với `strict=True` và `weights_only=True`.
  - Cách đánh giá: dùng **chính hàm `train.evaluate`** đã cho ra kết quả lần chạy 2 (hàm này đã được mở rộng cho 6 lớp, cho RGB và cho tùy chọn fp32/fp16). Mỗi mô hình được đánh giá hai lần:
    - fp32, như `evaluate.py` của tác giả, để so với Bảng 3 và Bảng 4;
    - fp16, như lần chạy 2, để tách riêng phần chênh lệch do fp16.
  - Với mô hình 3 lớp, module lưu xác suất (từ lần đánh giá fp16) theo đúng định dạng của `train.py`, để chạy tiếp được `src.weeds.analyze`.
  - Tiêu chí C1 trong `analyze.go_no_go` so mIoU với số của bài báo cho đúng loại đầu vào (khóa `paper_test`, ví dụ RGB là 79,33 − 5). Các lần chạy của đề tài vẫn so với 82,90 − 5 = 77,90.
- Ngày 06/11 đã review lại code và sửa các lỗi review tìm ra; kết quả lần 2 tính lại bằng code mới khớp hoàn toàn với `decision_metrics.csv` và `go_no_go.json` đã lưu. Các thay đổi:
  - `decision.crossfit_decisions` trả về quyết định cho từng ô; `evaluate_crossfit` và script vẽ sơ đồ cùng dùng hàm này.
  - Lời giải thích trên sơ đồ đúng cho cả trường hợp ngưỡng không "bắt chéo", và cả khi ngưỡng là vô cực.
  - Bước giải nén checkpoint kiểm tra kích thước file.
- Đã đọc code của tác giả (github.com/GFZ/weedsgalore):
  - lệnh mẫu huấn luyện 100 epoch cố định, batch 8, lưu checkpoint mỗi 100 vòng lặp, rồi chọn theo val;
  - tính ở fp32, không dùng fp16;
  - ảnh đầu vào chia 65535; RGB là 3 dải R, G, B đầu tiên.

**Notebook và test:**
- `notebooks/colab_weeds_pilot.ipynb` được sinh bởi `scripts/make_weeds_notebook.py`, nhúng sẵn các module. **Sửa `src/weeds` thì phải chạy lại script sinh notebook.**
  - Biến `BATCH` ở mục 1 quyết định batch, thư mục kết quả trên Drive (`run_dlv3p_msi_seed0_b<BATCH>`) và tên gói kết quả (`ket_qua_weeds_pilot_b<BATCH>.zip`, giải nén ra `results/weeds_pilot_b<BATCH>/`).
  - Nếu chạy lần mới với cùng batch (ví dụ seed khác), phải đổi tên thư mục và tên gói trong script, nếu không sẽ bị bỏ qua hoặc ghi đè.
- `notebooks/colab_weeds_authors.ipynb` (sinh bởi cùng script): tải `ckpts.zip` lên Drive, đánh giá 4 mô hình của tác giả, chạy tầng quyết định trên 2 mô hình 3 lớp, rồi đóng gói thành `ket_qua_weeds_authors.zip`. Giải nén gói này vào `results/weeds_authors/`.
- Trên runtime T4 chỉ cần bấm Run all; lỗi ở bước nào thì dừng ở bước đó.
- `src/weeds/runs.py` định nghĩa các lần huấn luyện thêm và hàm tổng hợp; `train.py` có thêm `--lr-schedule cosine`, lưu trạng thái lịch learning rate trong checkpoint để chạy tiếp đúng; `train_done.json` ghi thêm epochs, patience, lr_schedule. `notebooks/colab_weeds_runs.ipynb` sinh bởi cùng script.
- `src/weeds/counterfactual.py`: giải thích phản thực ở mức quyết định, tính từ bản đồ xác suất, không chạy lại mô hình (xem mục 6, 2b).
- `src/weeds/counterfactual_check.py`: cho mô hình kiểm chứng giải thích phản thực trên Colab; notebook `notebooks/colab_weeds_cfcheck.ipynb`.
- Có 68 test trong `tests/` (các file `test_weeds_*.py` và `test_pipeline_figure.py`). Bộ mặc định chạy khoảng 2 phút; test chạy cả quy trình XAI chỉ chạy khi đặt `WEEDS_SLOW_TESTS=1`. Chạy test cần dữ liệu bằng `WEEDSGALORE_ZIP=data/weedsgalore/weedsgalore-dataset.zip .venv/Scripts/python -m pytest`. Venv đã có torch 2.6.0 bản CPU.

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

**Đối sánh bằng mô hình công bố của tác giả** (`results/weeds_authors/`, chạy trên Colab ngày 06/11/2026):
- **Code đánh giá của đề tài được kiểm chứng.** Cả 4 mô hình ra đúng số của bài báo, lệch dưới 0,005 điểm mIoU, và mọi IoU từng lớp khớp đến 2 chữ số thập phân. Số liệu (fp32, `authors_vs_paper.json`):
  - ảnh 5 dải, 3 lớp: 82,90 (bài báo 82,90);
  - RGB, 3 lớp: 79,33 (79,33);
  - ảnh 5 dải, 6 lớp: 55,52 (55,52);
  - RGB, 6 lớp: 50,81 (50,81).
- Đánh giá bằng fp16 chỉ lệch so với fp32 dưới 0,002 điểm. Vì vậy chênh lệch 3,65 điểm của lần chạy 2 **hoàn toàn nằm ở khâu huấn luyện**. Mô hình 5 dải do đề tài tự huấn luyện chỉ ngang mô hình RGB của tác giả.
- Ma trận nhầm lẫn của mô hình tác giả (ảnh 5 dải) khớp Hình 4 của bài: ngô 0,079 / 0,792 / 0,130; cỏ 0,070 / 0,006 / 0,924. Mô hình đề tài bỏ sót thành nền nhiều hơn: ngô 15,5%, cỏ 12,0%.
- **Tầng quyết định**, thiết lập chính (216 ô, 108 ô cần phun), `decision_metrics.csv`:

| Mô hình | Chính sách | Giảm thuốc | Chưa quyết | Bỏ sót | Phun nhầm |
|---|---|---|---|---|---|
| Đề tài, 5 dải (79,25) | conformal / plug-in | 49,5% / 51,4% | 6,5% / 0% | 0,9% / 5,6% | 1,9% / 2,8% |
| Tác giả, 5 dải (82,90) | conformal / plug-in | 49,5% / 48,1% | 4,2% / 0% | 3,7% / 3,7% | 4,6% / 7,4% |
| Tác giả, RGB (79,33) | conformal / plug-in | 51,4% / 58,3% | 3,2% / 0% | 6,5% / **16,7%** | 3,7% / 0% |

- **Nhận xét:**
  - Với cả 3 mô hình, conformal giữ tỉ lệ bỏ sót và phun nhầm dưới α = β = 0,1, và tỉ lệ chưa quyết chỉ 3–7%.
  - Plug-in phụ thuộc vào mô hình: với mô hình RGB, plug-in bỏ sót 16,7% (18 ô). Đó là vì mô hình này đoán thiếu cỏ, nên ngưỡng 1% cố định là sai; conformal tự hạ ngưỡng xuống khoảng 0,6–0,8% và kéo tỉ lệ bỏ sót về 6,5%. Đây là luận điểm chính cho tầng quyết định.
  - mIoU cao hơn không có nghĩa là quyết định tốt hơn. Mô hình tác giả hơn 3,65 điểm mIoU nhưng bỏ sót 4 ô so với 1 ô của mô hình đề tài. Tuy nhiên chênh lệch chỉ vài ô nên chưa kết luận được mô hình nào tốt hơn ở mức quyết định.
  - Hiệu chỉnh riêng từng ngày yếu như nhau với cả 3 mô hình (25/5 và 6/6 có 85–93% ô chưa quyết), nên đây là giới hạn của dữ liệu, không phải của mô hình.

**Huấn luyện thêm** (`results/weeds_runs/`, đủ 6 lần chạy, Colab ngày 08/11/2026; batch 8, ảnh 5 dải; khoảng 17,3 giây mỗi epoch; tổng 3,26 giờ huấn luyện):

| Lần chạy | mIoU test, mô hình val cao nhất (epoch) | mIoU test, mô hình epoch cuối (epoch) |
|---|---|---|
| 100 epoch cố định, seed 0 | 79,60 (99) | 79,60 (99) |
| 100 epoch cố định, seed 1 | 81,74 (89) | 80,77 (99) |
| 100 epoch cố định, seed 2 | 81,53 (86) | 83,26 (99) |
| 100 epoch, lr cosine, seed 0 | 81,49 (99) | 81,49 (99) |
| Dừng sớm, seed 0 (lần chạy 2, `results/weeds_pilot_b8/`) | 79,25 (149) | chưa đánh giá |
| Dừng sớm, seed 1 | 81,20 (71) | **66,93** (101) |
| Dừng sớm, seed 2 | **83,57** (133) | 83,74 (163) |

- **Kết luận chính: khoảng cách 3,65 điểm là dao động ngẫu nhiên của quá trình huấn luyện, không phải code sai.**
  - 7 mô hình chọn theo val có mIoU test 79,25–83,57, trung bình 81,20 (độ lệch chuẩn 1,44).
  - `es_seed2` vượt bài báo (83,57 so với 82,90).
- **Cùng seed vẫn ra kết quả khác:** GPU dùng thuật toán không tất định (`cudnn.benchmark`).
  - Phiên 2 đã huấn luyện lại từ đầu cả 4 lần chạy 100 epoch đã xong ở phiên 1, không chạy tiếp; có thể do thư mục `runs` trên Drive không còn, ví dụ khi dùng tài khoản Google khác.
  - Số của phiên 1 (chỉ còn ghi lại, file đã bị ghi đè khi giải nén zip mới) lệch so với phiên 2 từ −2,90 đến +3,89 điểm cho cùng seed:
    - seed 0: 81,76 (phiên 1) so với 79,60 (phiên 2);
    - seed 1: 77,85 so với 81,74;
    - seed 2: 84,42 so với 81,53;
    - cosine: 81,74 so với 81,49.
  - Gộp cả 11 mô hình chọn theo val của hai phiên: trung bình 81,29, độ lệch chuẩn 1,86, khoảng 77,85–84,42; 2 mô hình đạt từ 82,90 trở lên.
  - **Khi báo cáo chỉ dùng số của phiên 2**, vì phiên 1 không còn file.
- **Chọn theo val an toàn hơn lấy epoch cuối.** Mô hình epoch cuối dao động mạnh hơn nhiều: độ lệch chuẩn 6,25, và `es_seed1` ở epoch cuối chỉ đạt 66,93 (IoU ngô 48,4). Giả thuyết "chọn val may mắn" bị bác bỏ. Learning rate cosine làm hai cách chọn trùng nhau.
- **Dao động tập trung ở ngày 15/6** (2 ô ảnh sau khi phun): mIoU 55,3–80,5 giữa 12 mô hình. Các ngày khác: 25/5 63,3–72,6; 30/5 71,3–78,5; 6/6 75,6–85,9 (`test_by_date`).
- **Tầng quyết định vững với cả 11 mô hình khác nhau** (mIoU test 66,93–83,74, thiết lập chính). Sửa ngày 11/11: trước đây ghi 13 là sai. `runs_summary.csv` có 12 dòng, nhưng ở `e100_seed0` và `e100_cosine_seed0` mô hình chọn theo val chính là mô hình epoch cuối (cùng epoch 99), nên chỉ có 10 mô hình, cộng lần chạy 2 thành 11. Câu "giữa 12 mô hình" ở trên cũng là 12 dòng, tức 10 mô hình. Các khoảng giá trị không đổi:
  - giảm thuốc 49,5–51,4%;
  - chưa quyết 0–7,9%;
  - bỏ sót tối đa 8,3%, phun nhầm tối đa 7,4%, cả hai đều dưới α = β = 10%.
  - Các số trên ở τ = 1% với điểm s cũ. Ở thiết lập chính hiện tại (τ = 0,1%, điểm mới) xem mục 3.7: một mô hình bỏ sót 10,3%, vượt α đúng một ô.
- Lần chạy 2 không có trên Drive của phiên Colab này, nên mô hình epoch cuối của nó chưa được đánh giá.

**Hạn chế:** lịch 100 epoch và lịch dừng sớm mỗi lịch có 3 seed, lịch cosine chỉ có 1; kết quả huấn luyện không tất định, nên cần báo cáo theo phân bố, không theo một lần chạy; tập test nhỏ (26 ô ảnh, 8 vị trí), nên các tỉ lệ bỏ sót và phun nhầm chỉ dựa trên 1–3 ô lưới; mIoU chỉ vượt ngưỡng C1 1,35 điểm.

**Giải thích quyết định (XAI)** (`results/weeds_xai/`, Colab ngày 09/11/2026; mô hình tác giả `dlv3p_msi_3`, fp16, IG 32 điểm, vùng che 20×20 điểm ảnh tức 5 cm × 5 cm, 10 lần ngẫu nhiên, seed 0; số liệu từ `xai_summary.json`, `xai_cells.csv`, `xai_deletion.csv`, `xai_maps.npz`):
- **Các phép kiểm tra đều đạt:**
  - quyết định tính lại khớp số đã báo cáo, chênh 0 (104 PHUN, 103 KHÔNG PHUN, 9 CHƯA CHẮC);
  - điểm số chấm lại lệch tối đa 8·10⁻⁶, không ô nào đổi quyết định;
  - ảnh "đất trống" có s gần 0 ở mọi ô (dưới 5·10⁻⁶);
  - tính đầy đủ của IG: sai số tương đối trung vị 0,26% (riêng các ô PHUN/CHƯA CHẮC: 0,12%, tối đa 4,7%). Sai số tương đối lớn nhất (18%) rơi vào một ô KHÔNG PHUN có s chỉ 0,03%.
- **Độ trung thực** (113 ô PHUN và CHƯA CHẮC; tỉ lệ đổi quyết định tính trên 104 ô PHUN). Tỉ lệ che tính theo 900 vùng của cả ô ảnh: 0,5% là 5 vùng, bằng 5% diện tích một ô lưới; 1% là 9 vùng; 2% là 18 vùng.

| Cách chọn vùng che | Mức giảm s, che 0,5% / 1% / 2% | Ô PHUN bị đổi quyết định | Thắng che ngẫu nhiên trong ô |
|---|---|---|---|
| Integrated Gradients | 45,5% / 63,5% / 82,7% | 25,0% / 45,2% / 67,3% | 100% / 100% / 100% |
| Grad-CAM | 5,3% / 9,3% / 17,2% | 1,9% / 3,8% / 9,6% | 38% / 45% / 54% |
| Xác suất cỏ của mô hình (mốc) | 47,3% / 66,6% / 85,1% | 26,9% / 46,2% / 69,2% | 100% / 100% / 100% |
| Ngẫu nhiên trong ô | 3,1% / 4,7% / 10,9% | 2,5% / 3,3% / 5,5% | – |
| Ngẫu nhiên trên cả ô ảnh | 0,35% / 0,53% / 1,05% | 0,29% / 0,19% / 0,58% | – |

- **Nhận xét:**
  - **IG trung thực.** IG thắng cả hai đối chứng ngẫu nhiên ở 113/113 ô tại mọi mức che từ 0,5% đến 5%. Ở mức 10%, đối chứng "ngẫu nhiên trong ô" che 90/100 vùng của ô, gần như xóa cả ô, nên so sánh ở mức này không còn ý nghĩa (IG vẫn thắng 93,8% số ô).
  - **Grad-CAM không trung thực ở mức ô lưới:** không hơn che ngẫu nhiên trong ô. Nguyên nhân: chỉ khoảng 10% phần dương của bản đồ Grad-CAM nằm trong ô đang quyết định, bằng đúng tỉ lệ diện tích 1/9, tức là bản đồ không định vị theo ô. Với IG, tỉ lệ này là 73% (ô PHUN) đến 88% (ô CHƯA CHẮC), tính theo trung vị. Đây là kết quả âm, nên báo cáo.
    - Đã kiểm tra (10/11) rằng đây là bản chất của phương pháp, không phải lỗi code. Bản đồ Grad-CAM của các ô khác nhau trong cùng một ô ảnh gần như giống hệt nhau: hệ số tương quan trung bình theo ô ảnh có trung vị 0,965 (thấp nhất 0,817, 24 ô ảnh). Với IG, hệ số này là 0,002. Grad-CAM gộp gradient thành một trọng số cho mỗi kênh trên cả bản đồ đặc trưng, nên nó chỉ ra mọi chỗ "giống cỏ" trên ô ảnh.
    - Claude đề xuất (10/11, chờ người dùng đồng ý): **không sửa Grad-CAM để chạy lại.** Code đúng công thức gốc; sửa sau khi đã thấy kết quả là chỉnh phương pháp cho đến khi qua được phép kiểm chứng. Nếu cần giải thích nguyên nhân trong bài viết, có thể thêm LayerCAM (Jiang và cộng sự 2021, *IEEE Transactions on Image Processing*), phương pháp giữ gradient theo từng vị trí, như một phương pháp thứ tư được khai báo rõ. Việc này cần chạy lại trên Colab, ưu tiên thấp.
  - **IG không trung thực hơn mốc "xác suất cỏ"**: mốc giảm s nhiều hơn một chút ở mọi mức che, và IG chỉ hơn mốc ở 11–28% số ô. Điều này dễ hiểu, vì s chính là trung bình xác suất cỏ trong ô. Vì vậy bản đồ vùng trình bày cho người dùng có thể lấy thẳng từ xác suất cỏ; IG có thêm ba điều: chia đóng góp theo dải phổ, cho thấy 12–27% đóng góp dương nằm ngoài ô (ngữ cảnh xung quanh), và tổng đóng góp bằng đúng s − s(đất trống). Khi viết phải trình bày đúng như vậy.
  - Theo ngày: mức giảm s khi che 1% tương tự nhau (60–71%), nhưng tỉ lệ đổi quyết định khác nhau: 25/5 100% (chỉ 2 ô), 30/5 77% (30 ô), 6/6 31% (72 ô). Nguyên nhân là ô ngày 6/6 nhiều cỏ (s trung vị 6,5%, so với ngưỡng khoảng 1%).
- **Dải phổ:**
  - Theo IG, **RE (red edge)** đóng góp lớn nhất ở 105/113 ô PHUN/CHƯA CHẮC. Trung vị tỉ lệ đóng góp: RE 70%, NIR 30%, R 9%, G −3%, B −10%.
  - Phép thay từng dải bằng giá trị đất xếp RE đầu ở 68/113 ô (R 24, NIR 18). Dải đứng đầu trùng với IG ở 60% số ô.
  - **Phép thay từng dải tạo phổ không có thật**: ở 57/103 ô KHÔNG PHUN, thay RE bằng giá trị đất đẩy s vượt 1%, tức mô hình "thấy cỏ" ở nơi không có. Vì vậy lời giải thích theo dải phổ nên dựa vào IG; phép thay dải chỉ dùng tham khảo.
  - IG đo so với ảnh đất trống, nên trả lời câu hỏi "vì sao là cỏ mà không phải đất", **không** trả lời "vì sao là cỏ mà không phải ngô".

**Giải thích phản thực, kiểm chứng bằng mô hình** (`results/weeds_cfcheck/`, Colab ngày 10/11/2026; mô hình tác giả `dlv3p_msi_3`, fp16, vùng 5 cm, quét thêm 2 vùng; số liệu từ `cfcheck_summary.json`, `cfcheck_cells.csv`, `cfcheck_steps.csv`). Cách làm ở mục 6, 2b.
- **Các phép kiểm tra đều đạt:**
  - Colab dùng đúng `results/weeds_xai/decisions.csv` (mã băm trùng);
  - không ô nào đổi quyết định khi mô hình chấm lại;
  - ước lượng trên Colab (bản đồ fp32) trùng với `results/weeds_counterfactual/` (bản đồ float16 trên máy) ở cả 113 ô, cả hai giá trị k, và thứ tự vùng trùng nhau. Như vậy chính các con số của bước 2 đã được kiểm chứng.
- **Thời gian:** 2.273 lần chạy mô hình, 119 giây trên T4, tức 1,05 giây mỗi ô (0,052 giây mỗi lần chạy). Ước lượng chỉ mất 0,42 ms mỗi ô trên laptop.
- **104 ô PHUN, "bỏ bao nhiêu vùng thì hết PHUN" (k theo đúng thứ tự vùng của lời giải thích):**
  - ước lượng đúng chính xác ở 46 ô (44%), thiếu vùng ở 49 ô, thừa vùng ở 9 ô;
  - phân bố mô hình trừ ước lượng: −3: 1, −2: 4, −1: 4, 0: 46, +1: 31, +2: 11, +3: 2, +4: 4, +5: 1. Lệch không quá 1 vùng ở 81/104 ô (78%), không quá 2 vùng ở 96/104 ô (92%);
  - sai lệch tương đối có trung vị 0, phân vị 90% là 17,7%, lớn nhất 42,9%. Sai lệch tăng theo k: ô có k ước lượng ≤ 2 lệch trung bình +0,22 vùng, ô có k > 20 lệch +1,22 vùng;
  - theo ngày: 30/5 đúng 19/30 ô, 6/6 đúng 26/72 ô, 25/5 đúng 1/2 ô;
  - **các quyết định "mong manh" là có thật:** mô hình xác nhận 11 ô PHUN hết phun chỉ khi bỏ 1 vùng 5 cm (ước lượng cũng ra 11 ô), 14 ô với tối đa 2 vùng (ước lượng 18), 25 ô với tối đa 4 vùng (ước lượng 26). Trong 18 ô có k ước lượng ≤ 2, ước lượng đúng 14 ô và thiếu 1 vùng ở 4 ô.
- **Câu giải thích "bỏ k vùng này thì ô hết PHUN", lấy k = k ước lượng + m, đúng theo mô hình ở:** m = 0: 54/104 ô (51,9%); m = 1: 85/104 (81,7%); m = 2: 95/104 (91,3%); m = 3: 98/104 (94,2%); m = 4: 98/101 ô đã quét (97,0%). Với câu "... thì KHÔNG PHUN" (113 ô): 56,6%, 74,3%, 87,6%, 91,6% (98/107), 94,0% (94/100). **Lưu ý:** m được chọn trên chính tập test, nên đây là con số trong mẫu.
- **9 ô CHƯA CHẮC:** 8 ô đúng chính xác. Bỏ 1 vùng thì mô hình cho KHÔNG PHUN, đúng như ước lượng. Ô còn lại ước lượng 2 vùng, mô hình cần 3. Như vậy câu "chỉ cần ra kiểm tra 1–3 vùng 5 cm này" đã được mô hình xác nhận cho cả 9 ô.
- **Tính cộng dồn của ước lượng không đúng hoàn toàn (ảnh hưởng của vùng xung quanh):**
  - mức giảm s của ước lượng lớn hơn mô hình khoảng 4–8% (trung vị tỉ số: 1,08 ở k = 1, 1,04 ở k = 10); sai số trung vị theo đường cong là −3,4 cm² (ước lượng thấp hơn), phân vị 95% của trị tuyệt đối là 14,5 cm². Sai số lớn hơn ở ngày 6/6, khi có nhiều cỏ;
  - ở 20 ô, bỏ thêm một vùng lại làm s **tăng** hơn 1 cm² (lớn nhất 10,1 cm²); ví dụ `2023-05-30_0720` ô 2: bỏ vùng thứ 2 làm s tăng từ 48,8 lên 56,0 cm²;
  - 7 ô có quyết định đổi rồi đổi ngược lại khi bỏ thêm vùng (5 ô quay lại PHUN, 5 ô rời KHÔNG PHUN, có ô thuộc cả hai). Tất cả đều xảy ra sát ngưỡng.
- **Độ trùng quyết định dọc đường quét:** 91,7% trên 2.256 trường hợp, 81,4% trên 1.003 trường hợp có đổi quyết định. Con số này thấp hơn 95,1% ở bước 1 vì phép quét dồn vào vùng sát ngưỡng; hai con số đo trên hai tập trường hợp khác nhau, không so trực tiếp được.
- **Kết luận cho luận văn:**
  - lời giải thích phản thực đưa vào báo cáo nên lấy **k đã được mô hình kiểm chứng** (từ phép quét), không lấy k ước lượng;
  - ước lượng là xấp xỉ nhanh, lệch tối đa 1 vùng ở 78% ô, nhưng thường lạc quan;
  - kiểm chứng bằng mô hình tốn khoảng 1 giây mỗi ô trên T4, chấp nhận được cho các ô người vận hành cần xem.

## 6. Việc tiếp theo

1. **Đối sánh với bài báo** (thầy yêu cầu ngày 05/11), làm theo từng bước:
   - **Bước 1: đã xong ngày 06/11.** Code đánh giá khớp bài báo; kết quả ở mục 5.
   - **Bước 2 (người dùng yêu cầu ngày 06/11, notebook đã sẵn sàng, chưa chạy):** `notebooks/colab_weeds_runs.ipynb` chạy 6 lần huấn luyện định nghĩa trong `src/weeds/runs.py`, theo thứ tự ưu tiên:
     - 3 seed với lịch của tác giả: 100 epoch cố định, learning rate không đổi;
     - 1 lần 100 epoch với learning rate giảm dần (cosine);
     - 2 seed với lịch của lần chạy 2 (dừng sớm).
     Tổng khoảng 4 giờ GPU, 2 phiên Colab; notebook tự chạy tiếp phiên sau. Kết quả: `ket_qua_weeds_runs.zip`, giải nén vào `results/weeds_runs/`, chỉ gồm json và csv.
     - Đã đối chiếu kỹ code huấn luyện của tác giả: loss, tăng cường ảnh, DataLoader, khởi tạo mô hình đều giống. Chỉ khác lịch huấn luyện, cách chọn checkpoint và fp32/A100. fp32 không thử vì batch 8 ở fp32 nhiều khả năng không vừa bộ nhớ T4.
     - Lần chạy 2: mIoU val tốt nhất trong 100 epoch đầu là 83,19 (epoch 94), cả quá trình là 83,94 (epoch 149). Như vậy chỉ rút về 100 epoch chưa chắc giúp ích.
   - Mô hình phân vùng chính trong báo cáo gửi thầy là mô hình công bố của tác giả; các lần tự huấn luyện là phép kiểm tra khả năng tái hiện.
   - **Phát hiện khi review (06/11): có thể đang chọn mô hình "may mắn" trên val.**
     - Mô hình của tác giả: val 79,53, test 82,90 (RGB: val 74,26, test 79,32).
     - Lần chạy 2 thì ngược lại: val 83,94, test 79,25.
     - Chọn epoch có val cao nhất trong 180 epoch dao động mạnh có thể chọn trúng một epoch may mắn trên val. Vì vậy mỗi lần chạy giờ được đánh giá thêm **mô hình ở epoch cuối** (`--stage test --weights last`, kết quả trong `<run>/last/`). Mô hình epoch cuối của lần chạy 2 cũng được đánh giá ở đầu notebook.
     - `segmentation_metrics.json` có thêm `test_by_date` (mIoU test theo từng ngày bay) và `model` (mô hình nào, epoch nào).
   - **Sửa sau review (06/11):**
     - Đồng hồ phiên tính từ lúc máy Colab khởi động (`/proc/uptime`).
     - Cuối notebook gọi `drive.flush_and_unmount()`.
     - Chạy tiếp với thiết lập khác checkpoint thì báo lỗi.
     - Trạng thái bộ sinh ngẫu nhiên được lưu trong `last.pt`.
     - "patience" đếm lại khi `best.pt` mới hơn `last.pt`.
     - Epoch cuối không bị tính là "hết giờ".
     - `runs.py` không còn phụ thuộc giá trị mặc định của `train.py` (dùng `LEGACY`).
     - Bảng tổng hợp có cột batch và cột cách chọn mô hình, và xử lý được trường hợp chưa có lần chạy nào xong.
   - **Báo cáo gửi thầy (đã soạn ngày 06/11, dùng mô hình của tác giả làm chính theo lựa chọn của người dùng):** https://claude.ai/artifact/J1XPMkEg2d4pwGXoFKVeCs. Gồm tóm tắt, input/output kèm sơ đồ, dữ liệu, đối sánh, tầng quyết định kèm biểu đồ bỏ sót/phun nhầm, tính mới, hạn chế và kế hoạch, 6 điểm xin thầy chốt. Ngày 08/11 đã cập nhật báo cáo với kết quả 7 lần tự huấn luyện (thêm biểu đồ) và kết quả tầng quyết định trên 15 mô hình. Việc tiếp theo: người dùng gửi thầy và hẹn họp chốt.
2. **Chờ thầy trả lời** ba điểm ở mục 1.4.
2b. **XAI (người dùng yêu cầu làm ngay ngày 08/11, trước khi thầy chốt): đã chạy trên Colab ngày 09/11, kết quả và phân tích ở cuối mục 5.**
   - Module `src/weeds/explain.py`, notebook `notebooks/colab_weeds_xai.ipynb`. Mô hình được giải thích là mô hình công bố của tác giả (`dlv3p_msi_3`), trên 216 ô của 24 ô ảnh test trước ngày phun.
   - **Giải thích:**
     - Integrated Gradients từ ảnh "đất trống" (trung vị các điểm ảnh nền của tập train, từng dải), 32 điểm trên đường tích phân; tổng đóng góp phải bằng s(ảnh) − s(đất trống).
     - Grad-CAM trên đặc trưng của bộ giải mã.
     - Bản đồ xác suất cỏ của mô hình, dùng làm mốc so sánh.
     - Mức quan trọng của từng dải phổ, đo bằng cách thay từng dải bằng giá trị đất.
   - **Kiểm chứng độ trung thực** (chỉ cho các ô PHUN và CHƯA CHẮC):
     - xóa (thay bằng đất) 0,5–10% số vùng 20×20 điểm ảnh mà mỗi phương pháp xếp cao nhất;
     - so với xóa cùng số vùng ngẫu nhiên trên cả ô ảnh và ngay trong ô lưới (10 lần mỗi loại);
     - chỉ số: mức giảm tương đối của s, tỉ lệ thắng đối chứng ngẫu nhiên, tỉ lệ ô PHUN bị đổi quyết định.
   - Kết quả lưu theo từng ô ảnh trong `parts/`, nên chạy tiếp được nếu bị ngắt. Gói kết quả: `ket_qua_weeds_xai.zip`, giải nén vào `results/weeds_xai/`. Riêng `xai_maps.npz` (khoảng 20 MB) đã bị gitignore.
   - **Đã review và sửa (08/11):**
     - Gradient lấy theo **tổng** xác suất cỏ trong ô rồi chia lại, vì trung bình trên 40.000 điểm ảnh tràn về 0 ở fp16; có tùy chọn `--no-amp`.
     - Bản đồ lưu ở fp32.
     - Mốc "xác suất cỏ" chỉ tính trong ô.
     - Quyết định tính một lần (`decisions.csv`) và đối chiếu với `results/weeds_authors/dlv3p_msi_3/go_no_go.json`, giá trị này được nhúng vào notebook.
     - Thiết lập cố định theo thư mục (`xai_config.json`, `XAI_VERSION`).
     - Tỉ lệ đổi quyết định tính theo điểm số chấm lại.
     - Kiểm tra lưới che vùng.
     - Phép che ngẫu nhiên trên cả ô ảnh dùng chung cho mọi ô.
     - `DeepLabV3Plus.features/head` nằm trong lớp mô hình.
     - Hàm dùng chung `analyze.primary_decisions` (cả `make_pipeline_figure.py` cũng dùng). Hàm này ra đúng quyết định đã báo cáo cho mô hình tác giả: 104 PHUN, 103 KHÔNG PHUN, 9 CHƯA CHẮC, chênh 0. Như vậy có 113 ô được kiểm chứng độ trung thực.
     - Kết quả lần 2 và của mô hình tác giả tính lại khớp hoàn toàn với file đã lưu.
   - **Phân tích độ trung thực: đã xong ngày 10/11** (cuối mục 5).
   - **Hướng mới, người dùng đồng ý ngày 10/11: giải thích phản thực (counterfactual) ở mức quyết định** (Wachter và cộng sự 2018, *Harvard Journal of Law & Technology*). Ví dụ: "bỏ k đám cỏ 5 cm × 5 cm này thì ô không còn PHUN"; "ô CHƯA CHẮC cách ngưỡng phun Z cm² cỏ"; "ô KHÔNG PHUN cần thêm V cm² cỏ mới phải phun". Lý do chọn:
     - quyết định chỉ là ngưỡng trên s, nên câu hỏi người vận hành cần trả lời là "đổi gì thì quyết định đổi";
     - thay cỏ bằng đất là thay đổi có thật ngoài ruộng;
     - nếu tính được từ bản đồ xác suất thì giải thích có ngay, phù hợp yêu cầu thời gian thực.
     IG giữ lại cho phần dải phổ, Grad-CAM giữ làm đối chứng âm. **Chưa tra cứu tài liệu, chưa được nói là mới.**
   - **Bước 1 (10/11, chạy trên máy, chỉ đọc file): ước lượng phản thực từ bản đồ xác suất có khớp với mô hình không.**
     - Cách làm: ước lượng s sau khi xóa bằng phép trừ, s − (tổng xác suất cỏ của các vùng bị xóa)/200². So với s do mô hình tính lại trong `xai_deletion.csv`, phương pháp `predicted_weed`, vì phương pháp này xóa đúng các vùng có xác suất cỏ cao nhất trong ô. Dùng `results/weeds_authors/dlv3p_msi_3/probs_test.npz`, file bị gitignore và chỉ có trên máy.
     - Kết quả trên 113 ô × 5 mức che:
       - quyết định sau khi xóa trùng với mô hình ở 545/565 trường hợp (96,5%);
       - với ô PHUN, dự đoán "còn phun hay không" trùng ở 97,3%;
       - sai số trung vị khoảng 1–3,5 cm² cỏ dự đoán (s × 2500 cm²), phân vị 95% tối đa 13,6 cm², trong khi ngưỡng phun tương đương 28–33 cm².
     - **Ước lượng lệch có hệ thống về một phía:** nó đánh giá mức giảm s cao hơn mô hình khoảng 1–6% (trung vị theo mức che). 18/20 trường hợp lệch là ước lượng cho rằng ô đã hết PHUN trong khi mô hình vẫn PHUN hoặc CHƯA CHẮC. Các trường hợp lệch đều nằm sát ngưỡng.
     - Kết luận: ước lượng dùng được để hiển thị ngay, nhưng số vùng tối thiểu k có thể thiếu ở ô sát ngưỡng. Bản dùng trong luận văn nên được mô hình kiểm chứng lại, mỗi ô vài lần chạy xuôi trên Colab, bắt đầu từ k ước lượng.
   - **Bước 2 (10/11, chạy trên máy): giải thích phản thực ước lượng cho cả 216 ô.**
     - Module `src/weeds/counterfactual.py` (chỉ dùng numpy và pandas) và 8 test trong `tests/test_weeds_counterfactual.py`.
     - Lệnh chạy: `python -m src.weeds.counterfactual --probs results/weeds_authors/dlv3p_msi_3/probs_test.npz --decisions results/weeds_xai/decisions.csv --deletion results/weeds_xai/xai_deletion.csv --out results/weeds_counterfactual`.
     - Kết quả nằm trong `results/weeds_counterfactual/`, gồm `counterfactual_cells.csv` (từng ô, kèm danh sách vùng cần bỏ) và `counterfactual_summary.json`. Phần đối chiếu với mô hình tái hiện đúng số của bước 1.
     - Diện tích tính là diện tích cỏ dự đoán, tức tổng xác suất cỏ nhân diện tích điểm ảnh 0,0625 cm². Mỗi vùng 5 cm × 5 cm rộng 25 cm².
     - **104 ô PHUN:** số vùng tối thiểu phải bỏ để hết PHUN có trung vị 9 (nhỏ nhất 1, tứ phân vị 4,75–21, lớn nhất 48). Diện tích cỏ dự đoán trong các vùng đó có trung vị 72,5 cm² (4,5–651).
       - Có 11 ô chỉ cần bỏ 1 vùng, 18 ô bỏ tối đa 2 vùng, 26 ô bỏ tối đa 4 vùng. Đây là các quyết định "mong manh", cũng là chỗ ước lượng dễ lệch nhất; bước 3 cần kiểm chứng kỹ các ô này.
       - Trung vị theo ngày: 25/5 là 3 vùng (2 ô), 30/5 là 4,5 vùng, 6/6 là 13 vùng.
     - **9 ô CHƯA CHẮC:**
       - thêm 2,6–7,8 cm² cỏ (trung vị 5,3) thì ô sẽ PHUN;
       - bớt 0,2–5,6 cm² (trung vị 2,0) thì KHÔNG PHUN;
       - bỏ 1 vùng là đủ (1 ô cần 2 vùng).
       Như vậy mỗi ô CHƯA CHẮC chỉ cách quyết định khoảng một vùng 5 cm. Danh sách vùng cho người kiểm tra biết chính xác cần ra xem chỗ nào.
     - **103 ô KHÔNG PHUN:** cần thêm 3,5–32,6 cm² cỏ (trung vị 21,8) mới phải phun. Phía thêm cỏ chỉ tính theo quy tắc, không kiểm chứng bằng mô hình được.
     - Thời gian: 216 ô mất 0,09 giây trên laptop (0,42 ms mỗi ô), không cần GPU, không cần chạy lại mô hình. Thời gian của IG chưa đo.
   - **Review bước 1 và 2 (10/11, người dùng yêu cầu): thuật toán đúng, kết quả giữ nguyên.**
     - Kiểm tra độc lập từ điểm ảnh, bằng vòng lặp và mặt nạ điểm ảnh thay cho cách tính của module, trên cả 216 ô: 0 lỗi. Cụ thể:
       - các vùng liệt kê nằm trong đúng ô và là các vùng có nhiều cỏ dự đoán nhất;
       - bỏ k vùng thì đổi quyết định; không tập k − 1 vùng nào trên lưới cố định làm đổi được;
       - các mức "thêm/bớt cm²" đúng là ranh giới của quy tắc;
       - diện tích tính đúng.
     - s tính từ bản đồ trùng với `decisions.csv` (lệch 1·10⁻¹⁶). Thứ tự đánh số ô khớp với `analyze.primary_decisions`.
     - Kích thước điểm ảnh 2,5 mm khớp với Bảng 1 của bài báo gốc (GSD 2,5).
     - Bước 1 (script tạm) và phần đối chiếu của bước 2 (module) là hai cách viết độc lập và ra cùng số.
     - Đã sửa nhỏ, không đổi kết quả:
       - cách tính số vùng che viết giống hệt `explain.py`;
       - bỏ cách dùng NaN làm khóa của dict;
       - ghi rõ phạm vi trong docstring (lưới cố định, chỉ bỏ vùng trong ô, chỉ đối chiếu ở 5 mức che).
     - Thêm chỉ số: bỏ 155 trường hợp cả hai cách đều không đổi quyết định (trùng hiển nhiên), còn 410 trường hợp, trùng 95,1%. **Nên báo cáo con số 95,1% cùng với 96,5%.** Các trường hợp không độc lập, vì 113 ô × 5 mức che.
     - Còn lại, không phải lỗi:
       - phép đối chiếu chỉ ở 5 mức che cố định, chưa ở đúng k của từng ô (vì vậy cần bước 3);
       - bản đồ lưu ở float16 có thể xếp hạng khác bản fp32 trên Colab khi hai vùng gần bằng nhau, nhưng khi đó ước lượng chỉ thay đổi tối đa bằng hiệu khối lượng giữa hai vùng.
   - **Bước 3: đã chạy trên Colab ngày 10/11, kết quả ở `results/weeds_cfcheck/`, phân tích ở cuối mục 5.**
     - Module `src/weeds/counterfactual_check.py` (GPU) và 6 test trong `tests/test_weeds_counterfactual_check.py` (CPU, khoảng 25 giây). Notebook: `notebooks/colab_weeds_cfcheck.ipynb`, sinh bởi `scripts/make_weeds_notebook.py`.
     - **Cách làm:** với mỗi ô PHUN/CHƯA CHẮC (113 ô), xếp hạng các vùng 5 cm trong ô theo xác suất cỏ của mô hình (fp32, như mốc `predicted_weed`). Lần lượt thay 1, 2, 3, ... vùng bằng đất rồi cho mô hình chấm lại. Dừng khi mô hình đã cho KHÔNG PHUN và k đã vượt cả hai số ước lượng thêm 2 vùng (`--extra`), hoặc khi đã bỏ hết 100 vùng của ô.
     - **Ghi lại:**
       - k theo mô hình (hết PHUN, thành KHÔNG PHUN) cạnh k ước lượng. k theo mô hình là vùng thứ mấy trong đúng thứ tự mà lời giải thích chỉ ra, nên nó kiểm chứng câu "bỏ k vùng này thì ô hết phun". **Nó không phải số vùng ít nhất đối với mô hình**: vì mô hình nhìn cả vùng xung quanh, một tập vùng khác có thể làm đổi quyết định với ít vùng hơn. Khi viết phải trình bày đúng như vậy;
       - đường cong s của mô hình và của ước lượng ở mọi k đã quét;
       - quyết định có đổi ngược lại khi bỏ thêm vùng không (`reverts_*`);
       - mức tăng s lớn nhất giữa hai bước liên tiếp.
     - **Đầu vào:** `decisions.csv` và `soil_baseline.json` của `results/weeds_xai` được nhúng vào notebook lúc sinh, để kiểm chứng đúng các quyết định đã phân tích. Notebook chỉ cần tập test.
     - **Khối lượng:** khoảng 2.200 lần chạy mô hình nếu mô hình khớp ước lượng, bằng khoảng 1/4 phép che của notebook XAI (khoảng 8.200 lần). Mục 5 chạy thử 1 ô ảnh và in thời gian ước tính, là cận dưới.
     - Kết quả: `ket_qua_weeds_cfcheck.zip` (csv và json), giải nén vào `results/weeds_cfcheck/`. Sau khi có kết quả, đối chiếu thêm k ước lượng trên Colab (bản đồ fp32) với k trong `results/weeds_counterfactual/` (bản đồ float16 lưu trên máy).
     - **Review (10/11, người dùng yêu cầu):** thuật toán quét đúng. Đã sửa:
       - câu chữ về ý nghĩa của k (xem trên);
       - thiết lập lưu cho thư mục kết quả nay ghi cả mã băm của `decisions.csv` và giá trị đất, để không trộn kết quả của đầu vào khác;
       - mô tả thời gian ước tính.
       Thêm test kiểm chứng độc lập với mạng DeepLab ngẫu nhiên: tự xếp hạng vùng và chấm từng ảnh một cho mọi k. Điểm số trong test này tăng ở 8/16 bước, quyết định đổi rồi đổi ngược lại; thứ tự vùng, điểm số, k và cờ đổi ngược đều khớp với phép quét theo lô.
     - Test với mô hình giả theo từng điểm ảnh: ước lượng khớp mô hình, k tính tay đúng, điều kiện dừng đúng. Riêng ô toàn đất có ngưỡng rất nhỏ thì ước lượng đạt KHÔNG PHUN còn mô hình thì không, vì ước lượng coi đất có xác suất cỏ bằng 0. Đây đúng là loại sai lệch cần đo.
   - **Hình giải thích (10/11):** `scripts/make_counterfactual_figure.py` vẽ `data/weedsgalore/hinh_giai_thich_phan_thuc.png` (2200 × 1793 điểm ảnh, nằm trong `data/` nên không commit). Script chạy khoảng 10 giây trên máy, dùng matplotlib 3.10.9 (đã thêm vào `requirements.txt`).
     - **Mỗi ví dụ một hàng:** ô ảnh 1,5 m có viền ô lưới; ô phóng to với các vùng cần bỏ (đánh số khi không quá 9 vùng); bản đồ xác suất cỏ; đường cong lượng cỏ dự đoán theo số vùng bỏ (mô hình chấm lại và ước lượng nhanh, nền tô theo vùng quyết định); đóng góp IG của từng dải tính bằng cm²; ba câu giải thích: vì sao, đổi gì thì quyết định đổi (k đã kiểm chứng), dải phổ và nhãn thật.
     - **Ví dụ được chọn theo quy tắc cố định** (hàm `pick_examples`), bỏ qua các ô có quyết định đổi ngược lại:
       1. PHUN mong manh: ô có nhiều cỏ dự đoán nhất trong các ô mà mô hình xác nhận bỏ 1 vùng là hết PHUN → `2023-05-30_0649` ô 5 (đánh số từ 1). Bỏ 1 vùng thì thành CHƯA CHẮC, bỏ 2 vùng thì KHÔNG PHUN; nhãn thật 1,38%.
       2. CHƯA CHẮC: ô có nhiều cỏ dự đoán nhất trong các ô chỉ cần bỏ 1 vùng → `2023-05-30_0649` ô 6. Bỏ 1 vùng thì KHÔNG PHUN; nhãn thật 0,94%, tức ô thực ra không cần phun.
       3. Ngày phun gần nhất (6/6), ước lượng thiếu từ 2 vùng trở lên, lấy ô có k ở giữa → `2023-06-06_0770` ô 1. Mô hình cần 29 vùng, ước lượng nhanh 27 vùng; nhãn thật 13,62%.
     - **Kiểm tra trước khi vẽ:** điểm số từ bản đồ xác suất trên máy khớp với Colab, và thứ tự vùng khớp với `results/weeds_counterfactual`. 6 test trong `tests/test_counterfactual_figure.py`.
     - **Màu:** màu CHƯA CHẮC đổi từ `#f5aa00` (màu của hình sơ đồ cũ) sang `#c98500`, để bộ ba màu quyết định đạt bộ kiểm tra màu: dải độ sáng, độ phân biệt với người mù màu và tương phản 3:1. Hình sơ đồ cũ giữ nguyên.
     - Đóng góp IG hiển thị bằng cm² cỏ dự đoán vì các dải cộng lại đúng bằng lượng cỏ dự đoán của ô. Hiển thị theo % thì ra số lớn hơn 100%, vì có dải đóng góp âm.
     - **Bản gọn cho báo cáo (11/11):** `--simple` vẽ `data/weedsgalore/hinh_giai_thich_don_gian.png` cho khổ A4. Cùng 3 ví dụ; mỗi ví dụ gồm ô 50 cm (chỗ máy cho là cỏ tô đỏ, các vùng đánh số), đường cong chỉ của mô hình chấm lại, và 3 dòng chữ (vì sao, đổi gì thì quyết định đổi, nhãn thật). Bản này được nhúng vào `docs/BAO_CAO_TIEN_DO.md`; bản chi tiết vẫn giữ. Phần vẽ đường cong đã tách thành hàm `draw_curve` dùng chung; bản chi tiết vẽ lại cho ra đúng từng byte như trước.
   - **Báo cáo tiến độ bằng Markdown (11/11, người dùng yêu cầu):** `docs/BAO_CAO_TIEN_DO.md`, văn phong đơn giản, gồm 4 mục: Đặt vấn đề, Mô tả bài toán, Phương pháp đề xuất, Kết quả.
     - Bản trong `docs/` chỉ có chữ và được commit; hình là đường dẫn tới `data/weedsgalore/so_do_pipeline_tac_gia.png` và `hinh_giai_thich_don_gian.png`.
     - Để xuất PDF: chạy `python scripts/embed_report_images.py`, script sinh `data/weedsgalore/BAO_CAO_TIEN_DO_co_hinh.md` có hình nhúng sẵn (không commit vì chứa ảnh của bộ dữ liệu). Sửa chữ thì sửa bản trong `docs/` rồi chạy lại script.
     - Đã review 4 lần theo yêu cầu người dùng; mọi câu có số liệu hoặc khẳng định đều được kiểm bằng file kết quả, code, hoặc PDF của bài báo (Celikkan 2025 trang 4 và 8, Celikkan 2023 trang 1, Melki 2025 trang 1–3 và 7–10, code của tác giả trên GitHub).
   - **Việc tiếp theo:** gửi báo cáo cho thầy; có thể cập nhật thêm artifact báo cáo cũ (https://claude.ai/artifact/J1XPMkEg2d4pwGXoFKVeCs).
0. **Việc chính từ 14/11: viết bài báo 12–15 trang trên Overleaf** theo chỉ đạo của thầy (mục 1), có thực nghiệm đối sánh với các nghiên cứu khác. Đề án làm sau.
3. **Ngưỡng phun:** đã tra cứu (mục 3.2) và đổi τ chính sang 0,1% (mục 3.3). Còn đọc Ali 2015 và toàn văn Nikolić 2025.
3c. **Chạy lại với điểm s mới (mục 3.7):** (1) người dùng chạy `colab_weeds_runs_rescore.ipynb` (CPU, độc lập với các bước sau; **xong 15/11**, đã kiểm tra và giải nén, kết quả ở mục 3.7) và `colab_weeds_xai.ipynb` (GPU; **xong 15/11**, in đúng `216 cells, decisions {'spray': 174, 'no_spray': 35, 'defer': 7}`, lệch 0 so với kết quả đã báo cáo; review trước khi chạy đã bỏ sót lỗi hòa, xem mục 3.7; zip mang tên lần chạy, `ket_qua_weeds_xai_tau0.001_argmax.zip`, cfcheck là `ket_qua_weeds_cfcheck_tau0.001_argmax.zip`, vì zip cũ cùng tên vẫn còn trên Drive); (2) Claude giải nén (với zip của `runs_rescore`: trước khi giải nén đè, so `segmentation_metrics.json`, `best.json`, `train_done.json`, `history.json` trong zip với bản trên máy; phải giống từng byte, khác là mô hình của phiên hay tài khoản khác), chạy `src.weeds.counterfactual`, sinh lại `colab_weeds_cfcheck.ipynb` (**xong 15/11**: notebook đã mở khóa, kiểm chứng 181 ô, thư mục Drive `cfcheck_authors_msi3_tau0.001_argmax`); (3) người dùng chạy notebook đó (**xong 16/11**, kết quả ở mục 3.7). Review ngày 16/11 không thấy lỗi logic. Phép thử: chạy `counterfactual_check.run()` trên 24 ô ảnh test thật với một mô hình giả lập trả về đúng bản đồ đã lưu và biến vùng bị che thành đất. Kết quả: k của mô hình bằng ước lượng ở 174/174 và 181/181 ô, trùng `results/weeds_counterfactual/` ở cả 181 ô, điều kiện dừng và số lượt đều đúng (3671 lượt = 23 ô ảnh có ô được kiểm chứng + 3648 bước); (3b) **việc cho bước 4, khi nhận kết quả cfcheck:** (i) `demo/views/explanation.py` gọi `check_example` với bản đồ P(cỏ), tức điểm cũ, nên phải truyền điểm ảnh cỏ `predicted_weed(p_crop, p_weed)` (**đã sửa 16/11**: trước khi sửa, test mọi trang demo báo lỗi ở trang này; sau khi sửa, toàn bộ 105 test đạt); (ii) `SCORE_TOL` = 1e-5 (trong `make_counterfactual_figure.py`) nhỏ hơn một điểm ảnh, trong khi 16/216 ô có điểm do mô hình chạy lại trên Colab (lô 1 ảnh) lệch bản đồ đã lưu (lô 2 ảnh) 1–2 điểm ảnh, và thứ tự vùng có thể đảo ở chỗ hai vùng bằng số điểm ảnh cỏ với tổng P(cỏ) rất sát, nên `check_example` phải cho phép vài điểm ảnh và các trường hợp đó; (4) người dùng đồng ý 16/11, làm từng phần:
   - 4.1 xong: `check_example` cho lệch tới 2 điểm ảnh.
   - 4.2 xong: chữ trong demo theo τ = 0,1% và điểm mới; sửa lỗi sập trang ở 4 ô không bao giờ ra KHÔNG PHUN; thanh chọn τ có đủ 7 mức.
   - 4.3 xong: vẽ lại 3 hình. Ngưỡng nhỏ có chữ số thập phân; khung IG ghi "xác suất cỏ"; sơ đồ chọn ô ảnh theo quy tắc (`2023-05-25_0750`); sửa số thứ tự ví dụ.
   - 4.4 (báo cáo tiến độ) **chờ người dùng quyết định cách viết**, vì câu chuyện đổi ở τ = 0,1% (số từ `decision_metrics.csv`, ô 50 cm, gộp 3 đợt bay, hiệu chỉnh chéo):
     - 194/216 ô cần phun, mức giảm tối đa 10,2%.
     - Mô hình 5 dải: conformal giảm 16,2%, CHƯA CHẮC 3,2% (7 ô), bỏ sót 7,2% (14/194), phun nhầm 4,5% (1/22). Ngưỡng cố định: giảm 15,3%, bỏ sót 6,2% (12/194), phun nhầm 4,5%. Tức là conformal không hơn ngưỡng cố định, và cả hai giảm nhiều hơn mức tối đa vì có bỏ sót.
     - Mô hình RGB: ngưỡng cố định bỏ sót 28,9% (56/194); conformal bỏ sót 0% nhưng CHƯA CHẮC 36,1% (78 ô), vượt C3.
     - Ở τ = 1% (điểm mới), conformal vẫn hơn: mô hình 5 dải bỏ sót 2,8% so với 6,5%; RGB 6,5% so với 21,3%.
3b. **Chạy lại XAI ở τ = 0,1% (bước 1–3 xong ngày 15/11, kết quả ở mục 3.4 và 3.5; bước 4 chuyển sang 3c vì đã đổi điểm s):**
   1. Người dùng chạy `notebooks/colab_weeds_xai.ipynb`; kết quả ghi vào thư mục Drive mới `xai_authors_msi3_tau0.001`. Notebook đối chiếu quyết định với `go_no_go.json` mới, và xai_config.json ghi cả thiết lập chính. Tải `ket_qua_weeds_xai.zip`, giải nén vào thư mục gốc repo (ghi đè `results/weeds_xai/`). Bản cũ của các file csv/json còn trong git; riêng `xai_maps.npz` thì không, muốn giữ thì đổi tên thư mục cũ trước khi giải nén.
   2. Claude chạy `src.weeds.counterfactual` trên máy (lệnh ở mục 2b), rồi chạy lại `scripts/make_weeds_notebook.py`. Notebook kiểm chứng phản thực chỉ nhúng quyết định khi `results/weeds_xai/xai_config.json` có đúng thiết lập chính; nếu chưa, notebook dừng ở mục 4.
   3. Người dùng chạy `notebooks/colab_weeds_cfcheck.ipynb` (thư mục Drive mới `cfcheck_authors_msi3_tau0.001`), giải nén vào `results/weeds_cfcheck/`.
   4. Claude cập nhật hình (`make_counterfactual_figure.py`, `make_pipeline_figure.py` có chữ "ngưỡng minh họa", và in τ với 0 chữ số thập phân nên sẽ ra "0%"), demo (`ui.SOURCE`, trang Tổng quan, Dữ liệu, Giải thích, Chạy thử: còn ghi "τ = 1%" hoặc in "0%"), báo cáo; chạy lại test (kể cả `WEEDS_SLOW_TESTS=1`).
4. **Thiết kế cách hiệu chỉnh lại khi sang đợt bay mới:** hiệu chỉnh riêng từng ngày hiện có quá ít mẫu.
5. **Mô-đun giải thích (XAI)** và phép kiểm chứng độ trung thực: đã chạy (mục 2b). Còn lại phần hình, lời giải thích bằng chữ và đưa vào báo cáo.

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
