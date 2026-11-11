<!-- Bản này chỉ có chữ; hình là file trong data/weedsgalore/, chỉ có trên máy (không đưa lên GitHub). Để xuất PDF: python scripts/embed_report_images.py, rồi dùng data/weedsgalore/BAO_CAO_TIEN_DO_co_hinh.md. -->

# Báo cáo tiến độ: Hệ thống hỗ trợ ra quyết định phun thuốc trừ cỏ từ ảnh drone

*Viết ngày 11/11/2026*

**Tóm tắt:** Drone chụp ruộng ngô. Máy tìm cỏ trên ảnh, rồi quyết định cho từng ô 50 cm × 50 cm: **PHUN**, **KHÔNG PHUN** hay **CHƯA CHẮC** (cần người ra xem). Hiện tại:

- Hệ thống giảm khoảng **một nửa diện tích phải phun thuốc** so với phun cả ruộng. Trên dữ liệu kiểm tra, một nửa số ô cần phun (cỏ chiếm từ 1% diện tích trở lên), nên 50% cũng là mức giảm tối đa có thể.
- Trên dữ liệu kiểm tra, tỉ lệ bỏ sót và phun nhầm đều **dưới 5%**.
- Các quyết định PHUN và CHƯA CHẮC có **lời giải thích đã được kiểm chứng bằng mô hình**, ví dụ "bỏ các vùng cỏ này thì hệ thống không còn quyết định phun ô này".

---

## 1. Đặt vấn đề

- **Phun thuốc trừ cỏ cho cả ruộng gây tốn kém và hại môi trường.** Cách khắc phục là phun theo vị trí, chỉ phun chỗ cần (Celikkan và cộng sự, 2023).
- **Lượng cỏ khác nhau nhiều giữa các chỗ.** Trên dữ liệu của đề tài, một nửa số ô 50 cm × 50 cm có cỏ dưới 1% diện tích.
- **Cách làm:** drone chụp ảnh, máy tìm chỗ có cỏ, rồi chỉ phun những chỗ cần.
- **Khó khăn:** máy có lúc nhận sai, và có hai kiểu sai:
  - **bỏ sót**: chỗ cần phun nhưng không phun, nên cỏ còn lại làm giảm năng suất cây trồng (Celikkan và cộng sự, 2023);
  - **phun nhầm**: chỗ không cần phun vẫn bị phun, gây tốn thuốc.
- **Vì vậy đề tài đặt ra ba yêu cầu cho hệ thống:**
  1. giữ được mức sai trong giới hạn cho trước, ví dụ không quá 10%;
  2. khi không chắc thì nói "chưa chắc", để người ra kiểm tra, thay vì đoán bừa;
  3. giải thích được vì sao phun ô này, và lời giải thích phải phản ánh đúng lý do máy đưa ra quyết định, chứ không chỉ "trông hợp lý".
- **Ba nghiên cứu gần nhất mà đề tài đã đọc kỹ chưa làm đủ ba điều này:**
  - Celikkan và cộng sự (2023) vẽ bản đồ độ không chắc chắn cho ảnh cỏ và cây trồng, nhưng không có quy tắc quyết định;
  - bài gốc của bộ dữ liệu (Celikkan và cộng sự, 2025) có đo độ không chắc chắn của mô hình, nhưng chỉ gợi ý ý tưởng "không phun khi không chắc", chưa biến nó thành quy tắc quyết định;
  - Melki và cộng sự (2025, bản thảo chưa qua bình duyệt) có bảo đảm tỉ lệ sai bằng conformal, nhưng chỉ ra quyết định phun hoặc không phun, không có "chưa chắc", không có phần giải thích. Bài dùng ảnh màu RGB, mỗi ảnh ứng với vùng phun của một vòi phun, không phải ảnh đa phổ.

## 2. Mô tả bài toán

**Dữ liệu: WeedsGalore.** Bộ dữ liệu công khai, công bố tại hội nghị WACV 2025, giấy phép CC BY 4.0.

| | |
|---|---|
| Nơi chụp | Ruộng ngô ở Potsdam (Đức), năm 2023 |
| Cách chụp | Drone DJI P4 Multispectral bay ở độ cao 5 m; mỗi điểm ảnh ứng với 2,5 mm ngoài ruộng |
| Ảnh | 156 ô ảnh, mỗi ô 600 × 600 điểm ảnh (1,5 m × 1,5 m), 5 dải phổ: đỏ, lục, lam, cận hồng ngoại (NIR), rìa đỏ (RE) |
| Thời gian | 4 đợt bay: 25/5, 30/5, 6/6, 15/6. Ruộng được phun thuốc ngày 12/6 |
| Nhãn | Từng điểm ảnh được gán: đất / ngô / cỏ (bộ dữ liệu có 4 lớp cỏ, gồm 3 loài chính và nhóm "cỏ khác"; đề tài gộp làm một như bài báo gốc) |
| Chia tập | 104 ô ảnh để huấn luyện, 26 để chọn mô hình, 26 để kiểm tra (giữ đúng cách chia của tác giả) |

**Đầu vào:** ảnh drone 5 dải phổ.

**Đầu ra:**
1. **Bản đồ cỏ:** mỗi điểm ảnh là đất, ngô hay cỏ.
2. **Quyết định cho từng ô 50 cm × 50 cm:** PHUN / KHÔNG PHUN / CHƯA CHẮC, kèm lời giải thích.

**Khi nào một ô "cần phun":** khi cỏ chiếm từ 1% diện tích ô trở lên. *Ngưỡng 1% hiện chỉ để minh họa, chưa có tài liệu làm căn cứ.*

**Đo kết quả bằng 4 con số** (chỉ số giảm thuốc lấy theo Melki và cộng sự, 2025):

| Chỉ số | Ý nghĩa |
|---|---|
| Giảm thuốc | Diện tích không phải phun, so với phun cả ruộng. Ô CHƯA CHẮC được tính là đã có người ra xem, cần thì phun |
| Chưa chắc | Bao nhiêu ô phải nhờ người ra xem |
| Bỏ sót | Trong các ô cần phun, bao nhiêu ô bị quyết định KHÔNG PHUN |
| Phun nhầm | Trong các ô không cần phun, bao nhiêu ô bị quyết định PHUN |

## 3. Phương pháp đề xuất

Hệ thống gồm 3 bước:

![Sơ đồ xử lý trên một ô ảnh của tập kiểm tra, với mô hình của tác giả](../data/weedsgalore/so_do_pipeline_tac_gia.png)

**Bước 1. Tìm cỏ trên ảnh.**
- Dùng mạng học sâu DeepLabv3+, là mô hình mà tác giả bộ dữ liệu đã công bố.
- Kết quả là xác suất mỗi điểm ảnh là cỏ.

**Bước 2. Quyết định cho từng ô.**
- Tính **lượng cỏ dự đoán** trong ô: trung bình xác suất cỏ của các điểm ảnh trong ô, xem như ước lượng % diện tích ô là cỏ.
- So với **hai ngưỡng**:
  - trên ngưỡng phun thì **PHUN**;
  - dưới ngưỡng không phun thì **KHÔNG PHUN**;
  - nằm giữa hai ngưỡng thì **CHƯA CHẮC**.
- Hai ngưỡng **không đặt bằng tay**. Chúng được tính bằng phương pháp *conformal* trên các vị trí khác của ruộng, nhằm giữ bỏ sót không quá 10% và phun nhầm không quá 10%.
- Nhờ vậy, khi mô hình tìm cỏ tốt hay kém, ngưỡng tự điều chỉnh theo để giữ mức sai trong giới hạn đã đặt. Bảo đảm này chỉ gần đúng, không tuyệt đối, vì các ô nằm cạnh nhau thường giống nhau.

**Bước 3. Giải thích từng quyết định.**
- **"Đổi gì thì quyết định đổi"** (giải thích phản thực).
  - Với ô PHUN: "nếu không có các vùng cỏ được đánh dấu này thì hệ thống không còn quyết định PHUN".
  - Với ô CHƯA CHẮC: "bỏ vùng 5 cm × 5 cm này thì thành KHÔNG PHUN". Đó là chỗ người kiểm tra nên xem trước.
  - Mỗi câu được **kiểm chứng**: xóa đúng các vùng đó khỏi ảnh (thay bằng đất) rồi cho mô hình chấm lại.
- **Dải phổ nào quan trọng:** tính bằng phương pháp Integrated Gradients (IG; Sundararajan và cộng sự, ICML 2017).
- **Kiểm tra lời giải thích có trung thực không** (tức là phản ánh đúng lý do mô hình quyết định): xóa vùng mà lời giải thích chỉ ra, rồi so với xóa cùng số vùng chọn ngẫu nhiên. Lời giải thích trung thực thì xóa vùng của nó phải làm thay đổi kết quả nhiều hơn hẳn xóa ngẫu nhiên.

## 4. Kết quả đạt được hiện tại

Kết quả tính trên **216 ô** của tập kiểm tra, thuộc 3 đợt bay trước ngày phun thuốc. Trong đó có **108 ô thật sự cần phun** (theo nhãn, với ngưỡng 1%).

### 4.1. Tìm cỏ trên ảnh: khớp bài báo gốc

- Chạy mô hình của tác giả bằng code của đề tài cho ra **đúng số của bài báo**: mIoU 82,90% với ảnh 5 dải và 79,33% với ảnh màu thường (RGB). Như vậy code đánh giá của đề tài là đúng.
- Đề tài tự huấn luyện 7 lần, cùng batch 8 như bài báo, với 3 lịch huấn luyện (100 epoch như tác giả, learning rate giảm dần, dừng sớm). Kết quả: mIoU 79,25–83,57% (trung bình 81,20%).
  - Số của bài báo (82,90%) nằm trong khoảng này, nên chênh lệch chủ yếu do dao động ngẫu nhiên giữa các lần huấn luyện.
  - Còn một khác biệt chưa thử riêng: đề tài huấn luyện ở chế độ fp16 (nhanh hơn, tốn ít bộ nhớ GPU hơn), còn tác giả dùng fp32.

### 4.2. Quyết định phun: giảm một nửa diện tích phun, mức sai trong giới hạn

Mô hình của tác giả, ảnh 5 dải:

| Cách làm | Giảm thuốc | Chưa chắc | Bỏ sót | Phun nhầm |
|---|---|---|---|---|
| Phun cả ruộng | 0% | 0% | 0% | 100% |
| Biết trước đáp án (mức tốt nhất có thể) | 50,0% | 0% | 0% | 0% |
| Ngưỡng cố định 1% | 48,1% | 0% | 3,7% | 7,4% |
| **Đề tài (hai ngưỡng conformal)** | **49,5%** | **4,2% (9 ô)** | **3,7% (4 ô)** | **4,6% (5 ô)** |

**Ý chính:**
- Giảm diện tích phun gần bằng mức tốt nhất có thể (49,5% so với 50%).
- Chỉ 9/216 ô cần người ra xem.
- **Ưu điểm chính là tự điều chỉnh khi mô hình yếu hơn.** Với mô hình chỉ dùng ảnh màu thường (RGB), ngưỡng cố định 1% bỏ sót tới **16,7%** (18 ô), vượt xa mức cho phép. Cách của đề tài tự hạ ngưỡng và kéo bỏ sót về **6,5%** (7 ô).
- Thay bằng 11 mô hình tự huấn luyện khác nhau (mIoU 66,93–83,74%), kết quả vẫn ổn định: bỏ sót không quá 8,3%, phun nhầm không quá 7,4%.
- Với mô hình 5 dải (đã tốt), ngưỡng cố định cũng gần đạt mức tốt nhất. So với nó, cách mới **giảm phun nhầm** (7,4% → 4,6%) với cùng mức bỏ sót (3,7%), đổi lại phải ra xem 9 ô.

### 4.3. Lời giải thích: đã kiểm chứng là trung thực

**Kiểm tra độ trung thực của lời giải thích** trên 113 ô PHUN và CHƯA CHẮC. Mỗi lần xóa 9 vùng 5 cm × 5 cm, tổng diện tích bằng 9% một ô. IG và Grad-CAM chọn những vùng chúng xếp cao nhất trên cả ô ảnh 1,5 m, còn đối chứng chọn ngẫu nhiên ngay trong ô:

| Cách chọn vùng để xóa | Lượng cỏ dự đoán giảm | Ô PHUN bị đổi quyết định |
|---|---|---|
| Theo lời giải thích IG | 63,5% | 45% |
| Theo Grad-CAM (Selvaraju và cộng sự, ICCV 2017) | 9,3% | 4% |
| Chọn ngẫu nhiên trong ô | 4,7% | 3% |

- **IG trung thực:** ở **cả 113/113 ô**, xóa theo IG làm lượng cỏ dự đoán giảm nhiều hơn xóa ngẫu nhiên.
- **Grad-CAM không dùng được** để giải thích từng ô.
  - Tính trung bình, nó hơn chọn ngẫu nhiên một chút (9,3% so với 4,7%). Nhưng so từng ô thì nó chỉ hơn ở 45% số ô, gần như tung đồng xu.
  - Lý do: bản đồ Grad-CAM của 9 ô trong cùng một ô ảnh gần như giống hệt nhau (hệ số tương quan trung vị 0,965), nên nó không chỉ ra được vùng riêng của từng ô.
- *"Trung thực" nghĩa là lời giải thích phản ánh đúng cách mô hình quyết định. Nó không có nghĩa là mô hình nhận đúng cỏ.*
- Theo IG, dải phổ quan trọng nhất là **rìa đỏ (RE)**, đứng đầu ở 105/113 ô.

**Giải thích "đổi gì thì quyết định đổi"** (đã cho mô hình chấm lại để kiểm chứng):
- **11 ô PHUN "mong manh" (sát ngưỡng):** chỉ cần bỏ **một** vùng cỏ 5 cm là hệ thống không còn quyết định PHUN (ô chuyển sang CHƯA CHẮC hoặc KHÔNG PHUN). Người dùng nên được báo về những ô này.
- **Cả 9 ô CHƯA CHẮC:** bỏ 1 vùng (8 ô) hoặc 3 vùng (1 ô) là thành KHÔNG PHUN.
  - Vì vậy người kiểm tra biết ngay **nên xem chỗ nào trước**: 1–3 vùng 5 cm mà quyết định phụ thuộc vào.
  - Đây là chỗ xem trước, không thay cho việc xem cả ô, vì cỏ mà mô hình bỏ sót có thể nằm ở chỗ khác.
- **Hai cách tính:**
  - *ước lượng nhanh* (0,4 ms mỗi ô, không cần GPU) đúng chính xác ở 44% ô PHUN, lệch không quá 1 vùng ở 78% ô PHUN, và thường hơi lạc quan;
  - vì vậy lời giải thích chính thức dùng số **đã kiểm chứng bằng mô hình**, mất khoảng 1 giây mỗi ô trên GPU T4 của Colab.

Ví dụ ba ô, được chọn theo quy tắc cố định:

![Ví dụ giải thích cho ba ô](../data/weedsgalore/hinh_giai_thich_don_gian.png)

### 4.4. Hạn chế cần lưu ý

- **Tập kiểm tra nhỏ** (216 ô, 8 vị trí). Mỗi tỉ lệ bỏ sót hoặc phun nhầm chỉ dựa trên vài ô, nên có thể dao động.
- **Một ruộng, một mùa vụ.** Chưa kiểm tra trên ruộng khác.
- **Ngưỡng "cần phun" 1% chỉ là minh họa.** Cần tìm ngưỡng có căn cứ từ tài liệu về cỏ dại.
- **Kích thước ô 50 cm cũng chưa có căn cứ.** Nó nên khớp với vùng mà máy phun điều khiển riêng được. Thử ô 25 cm thì diện tích phun giảm 56,5% thay vì 49,5%, nhưng có 51 ô CHƯA CHẮC thay vì 9. Hai kết quả không so sánh trực tiếp hoàn toàn được: cùng ngưỡng 1%, nhưng ô nhỏ chỉ cần ít cỏ hơn là đã "cần phun".
- **Hiệu chỉnh riêng cho từng đợt bay còn yếu.** Ở ngày 25/5 và 6/6, 85% và 93% số ô thành CHƯA CHẮC, vì có quá ít ô thuộc một trong hai nhóm (cần phun / không cần phun) để tính ngưỡng.
