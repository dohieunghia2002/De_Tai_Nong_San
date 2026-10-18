# Đề tài: Hệ thống hỗ trợ ra quyết định thu hoạch xoài theo thời gian thực từ phổ NIR, tích hợp AI giải thích được (XAI-driven DSS)

Bạn là kỹ sư ML giúp tôi viết code cho luận văn thạc sĩ **định hướng ứng dụng**. Đọc kỹ toàn bộ file này trước khi viết code.

## 1. Quy tắc làm việc (bắt buộc)

- Trả lời và báo cáo bằng **tiếng Việt**. Code, tên biến, comment bằng tiếng Anh.
- **Không chạy tính toán nặng trên máy tôi.** Phần huấn luyện mạng nơ-ron phải viết thành **notebook Google Colab (GPU T4 miễn phí)**. Trên máy local chỉ chạy các kiểm tra nhanh (vài giây) và phải nói trước khi chạy.
- **Không bịa số liệu.** Mọi con số trong báo cáo phải lấy từ output thực tế. Nếu chưa chạy thì ghi "chưa có kết quả".
- **Không tự sinh dữ liệu.** Chỉ dùng bộ dữ liệu công khai mô tả ở mục 3.
- Làm từng bước nhỏ. Sau mỗi bước, in ra kiểm tra (số dòng, số quả, sai số so với mốc) và dừng lại để tôi xem nếu có sai lệch.
- Cố định random seed. Lưu mọi kết quả ra CSV/JSON trong `results/`, kèm `results/summary.json`.

## 2. Ý tưởng đề tài

**Bài toán thực tế:** hợp tác xã và nhà đóng gói xoài (doanh nghiệp vừa và nhỏ) cần quyết định quả nào **đủ độ chín để thu hoạch hoặc phân hạng**. Độ chín được đo bằng **hàm lượng chất khô (dry matter content, DMC, %FW)**. Đo bằng cách cắt quả thì chậm và phá mẫu; máy **NIR cầm tay** cho kết quả tức thì nhưng có sai số, đặc biệt khi sang **mùa vụ mới**.

**Đóng góp của đề tài không nằm ở việc dự đoán DMC chính xác hơn** (bài toán này đã bão hòa, các bài trước đạt RMSEP 0,77–1,01). Đóng góp nằm ở **tầng quyết định**:

1. Biết **khi nào nên "chưa quyết"** (defer: chuyển sang kiểm tra phá mẫu hoặc chờ thêm), dựa trên độ bất định đã hiệu chỉnh.
2. Xử lý **lệch mùa vụ**: khi sang mùa mới, mô hình lệch hệ thống và conformal thông thường mất hiệu lực → hiệu chỉnh độ lệch theo lô bằng một số ít quả đo phá mẫu.
3. **Giải thích quyết định** (vì sao "chưa thu hoạch", vì sao "chưa quyết"), có **kiểm chứng độ trung thực** của lời giải thích.
4. Đánh giá bằng **chỉ số ở mức quyết định** (độ đúng, tỉ lệ báo "đủ chín" sai, tỉ lệ quả tự quyết được), không chỉ RMSEP.

## 3. Dữ liệu

- Nguồn: Anderson, Walsh, Subedi, Walsh — *Mango DMC and NIR spectra*, Mendeley Data, DOI 10.17632/46htwnp833 (bản v5, CC BY 4.0). Bài báo gốc: Anderson et al. 2020, *Postharvest Biology and Technology* 168:111202. Bài đánh giá CNN: Walsh et al. 2024, *Spectrochimica Acta Part A* 311:124003 (code: github.com/jemjemwalsh/research-deep-learning-nirs).
- **Dùng file `MangoDMC_NIR_Data_v3.csv`** (12.011 dòng × 315 cột). Không dùng `MangoDMC_NIR_Data_v4.csv` ở giai đoạn này.
- Cột: `Set, Season, Region, Date, Type, Cultivar, Pop, Temp, DM`, tiếp theo là 306 cột phổ hấp thụ có tên là bước sóng `285, 288, …, 1200` (nm, bước 3 nm). Nhãn là `DM`.
- Phân chia có sẵn (giữ nguyên, **không chia ngẫu nhiên**):

| Set | Phổ | Quả | Vai trò |
|---|---|---|---|
| Cal | 7.413 | 2.866 | huấn luyện |
| Tuning | 2.830 | 1.086 | huấn luyện / hiệu chỉnh |
| Val Ext | 1.448 | 725 | kiểm thử 1 = mùa vụ 4 (Anderson 2020) |
| Val Ext 2 | 320 | 160 | kiểm thử 2 = mùa vụ 5, năm 2019, quần thể 128 (R2E2) và 129 (KP), toàn quả xanh cứng (Walsh 2024) |

- **Mỗi quả được quét 2–2,6 lần** (phổ trùng lặp). File không có cột mã quả; tạo mã quả bằng `groupby(['Set','Pop','Type','Cultivar','DM'])` (cách này đã được đối chiếu: Tuning 1.086 và Val Ext 725 khớp chính xác Bảng 1 của bài gốc; Cal lệch 2 quả). Quyết định được đưa ra **theo quả**: lấy trung bình dự đoán của các phổ cùng một quả.
- Viết test kiểm tra các con số trong bảng trên khi load dữ liệu.

## 4. Kiến trúc hệ thống (đầu vào → các mô-đun → đầu ra)

**Đầu vào:** phổ NIR của một quả (hoặc một lô quả) + giống + ngưỡng thu hoạch.

- **M1 – Tiền xử lý:** Savitzky–Golay đạo hàm bậc 2 (polyorder 2, window 17) áp dụng trên **toàn phổ**, sau đó cắt về **684–990 nm** (còn **103 biến**). Cho phép cấu hình thêm SNV, đạo hàm bậc 1, window 13 để thử nghiệm.
- **M2 – Mô hình dự đoán DMC (mô-đun được huấn luyện):**
  - PLS 11 thành phần (baseline, chạy local được).
  - ANN 1 lớp ẩn 5 nơ-ron sigmoid (như Anderson 2020).
  - 1D-CNN (theo Walsh 2024) — chạy trên Colab.
- **M3 – Độ bất định và lệch mùa vụ:**
  - Split conformal: huấn luyện trên Cal, hiệu chỉnh trên Tuning, ở mức quả; khoảng dự đoán 90%.
  - Hiệu chỉnh độ lệch theo lô: với mỗi quần thể (`Pop`) mới, lấy ngẫu nhiên k quả (k = 5, 10) có giá trị DMC thật, ước lượng bias = mean(dự đoán − thật), trừ bias cho các quả còn lại; đánh giá trên các quả còn lại; lặp 30 lần.
  - (Thử) phát hiện bất thường phổ: Q-residual / Hotelling T² so với tập huấn luyện.
- **M4 – Quyết định:** với ngưỡng T: `đủ chín` nếu khoảng dự đoán nằm hoàn toàn trên T, `chưa chín` nếu nằm hoàn toàn dưới T, ngược lại `chưa quyết`. Ngưỡng T cấu hình được (thử 14, 15, 16 %FW; **đây là ngưỡng minh họa**, ngưỡng theo giống sẽ bổ sung sau khi có nguồn). Có tham số chi phí: chi phí báo "đủ chín" sai (thu hoạch quả non), chi phí bỏ sót, chi phí một lần kiểm tra phá mẫu → chọn ngưỡng defer tối ưu chi phí.
- **M5 – Giải thích quyết định (XAI):**
  - Cho mỗi quả: lý do bằng lời gồm (a) khoảng cách dự đoán tới ngưỡng, (b) độ rộng khoảng bất định, (c) cờ lệch lô/bất thường phổ, (d) vùng bước sóng đóng góp chính (PLS: hệ số × phổ; CNN: SHAP hoặc Integrated Gradients), gộp theo dải bước sóng.
  - **Kiểm chứng độ trung thực**: deletion/perturbation test (che các dải được cho là quan trọng → dự đoán và quyết định phải thay đổi nhiều hơn so với che dải ngẫu nhiên); kiểm tra tính ổn định giữa hai phổ của cùng một quả.
  - Không tự gán ý nghĩa hóa học cho bước sóng nếu không có nguồn trích dẫn.

**Đầu ra:** quyết định (đủ chín / chưa chín / chưa quyết) + lời giải thích + chỉ số tin cậy.

## 5. Thực nghiệm và mốc kiểm tra

**Bước 1 – Tái hiện baseline (bắt buộc khớp trước khi làm tiếp):**
- PLS 11 LV, huấn luyện trên Cal + Tuning (10.243 phổ), M1 như trên.
- Mốc: **RMSEP mùa vụ 4 = 1,01 %FW** (Anderson 2020; pilot của tôi đạt 1,01). Mùa vụ 5: pilot đạt RMSEP 1,83, bias (dự đoán − thật) −1,48; Walsh 2024 báo PLS tốt nhất 1,87, bias 1,56 (khác dấu có thể do quy ước, cần kiểm tra).
- RMSEP tính **theo phổ** như bài gốc (không bỏ mẫu nào trong tập kiểm thử).

**Bước 2 – Tầng quyết định với PLS (local):** báo cáo cho mỗi tập kiểm thử và mỗi ngưỡng T: độ đúng, tỉ lệ báo "đủ chín" sai, tỉ lệ quả tự quyết được (coverage), độ đúng trên quả tự quyết, độ phủ thực tế của khoảng 90%. Mốc pilot (T = 15%, khoảng conformal ±1,26):

| | Mùa vụ 4 | Mùa vụ 5 |
|---|---|---|
| Không defer | đúng 93,0% | đúng 61,3% |
| Conformal defer | tự quyết 72%, đúng 99,6% | độ phủ khoảng chỉ 19% (hỏng) |
| Hiệu chỉnh lô k=5 + defer | tự quyết 71%, đúng 99,7% | tự quyết 53%, đúng 95,1%, báo "đủ chín" sai 4,9% |

Code mới phải tái hiện được các số này (sai lệch nhỏ do random thì ghi rõ).

**Bước 3 – Notebook Colab cho ANN và 1D-CNN:** mốc Walsh 2024: CNN RMSEP 0,77 (mùa 4), 1,18 (mùa 5); ANN 0,88–0,89 (mùa 4). Không bắt buộc vượt mốc; chỉ cần đạt mức tương đương và báo cáo trung thực. Sau đó chạy lại Bước 2 với CNN.

**Bước 4 – XAI và kiểm chứng độ trung thực** (M5).

**Bước 5 – Demo web đơn giản** (tải phổ lên → quyết định + giải thích), làm sau cùng.

## 6. Cấu trúc repo mong muốn

```
data/            # đặt MangoDMC_NIR_Data_v3.csv ở đây (không commit)
src/data.py      # load, tạo mã quả, kiểm tra số lượng
src/preprocess.py
src/models_pls.py
src/uncertainty.py   # conformal, bias correction, OOD
src/decision.py
src/explain.py
notebooks/colab_cnn.ipynb
tests/
results/
```

Bắt đầu bằng Bước 1: viết `src/data.py` + `src/preprocess.py` + `src/models_pls.py` và test, chạy kiểm tra số lượng và RMSEP rồi báo kết quả cho tôi.
