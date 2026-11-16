<!-- Bản cập nhật ngày 16/11/2026. Bản ngày 11/11/2026 (docs/BAO_CAO_TIEN_DO.md) giữ nguyên để đối chiếu; bản đã gửi thầy có hình cũ ở data/weedsgalore/BAO_CAO_TIEN_DO.pdf và BAO_CAO_TIEN_DO_co_hinh.md, hình cũ ở data/weedsgalore/hinh_bao_cao_11_11/. Bản này chỉ có chữ; hình là file trong data/weedsgalore/, chỉ có trên máy (không đưa lên GitHub). Để xuất PDF: python scripts/embed_report_images.py, rồi dùng data/weedsgalore/BAO_CAO_TIEN_DO_16_11_co_hinh.md. Mọi con số lấy từ results/. -->

# Báo cáo tiến độ: Hệ thống hỗ trợ ra quyết định phun thuốc trừ cỏ từ ảnh drone

*Bản cập nhật ngày 16/11/2026. Bản ngày 11/11/2026 được giữ nguyên để đối chiếu.*

**Tóm tắt:** Drone chụp ruộng ngô. Máy tìm cỏ trên ảnh, rồi quyết định cho từng ô 50 cm × 50 cm: **PHUN**, **KHÔNG PHUN** hay **CHƯA CHẮC** (cần người ra xem). So với bản 11/11, ngưỡng "cần phun" nay dựa theo tài liệu (cỏ chiếm từ 0,1% diện tích ô), và kết quả được tính ở 7 mức ngưỡng. Hiện tại:

- Ở ngưỡng 0,1%, **194/216 ô kiểm tra cần phun**, nên phun theo ô chỉ giảm được tối đa 10,2% diện tích so với phun cả ruộng.
- Với mô hình chính (ảnh 5 dải phổ), hệ thống bỏ sót 7,2% số ô cần phun và phun nhầm 1/22 ô không cần phun, trong giới hạn 10% đặt ra; 7 ô cần người ra xem. Ở ngưỡng này, cách quyết định của đề tài **chưa tốt hơn** một ngưỡng cố định (bỏ sót 6,2%).
- Ưu điểm của cách quyết định của đề tài thể hiện khi mô hình tìm cỏ yếu hơn: với mô hình chỉ dùng ảnh màu (RGB), ngưỡng cố định bỏ sót **28,9%** số ô cần phun, còn cách của đề tài **không bỏ sót ô nào**, đổi lại 36,1% số ô phải có người ra xem (vượt mức 30% đặt ra).
- Các quyết định PHUN và CHƯA CHẮC có **lời giải thích đã được kiểm chứng bằng mô hình**, ví dụ "bỏ các vùng cỏ này thì hệ thống không còn quyết định phun ô này".

## Thay đổi so với bản 11/11

| | Bản 11/11 | Bản 16/11 |
|---|---|---|
| Ngưỡng "cần phun" | Cỏ chiếm từ 1% diện tích ô, chỉ để minh họa | Cỏ chiếm từ 0,1% diện tích ô, trong khoảng ngưỡng kinh tế 0,06–0,31% (Longchamps và cộng sự, Weed Science 2014); kết quả có cho 7 mức |
| Lượng cỏ dự đoán của một ô | Trung bình xác suất cỏ của các điểm ảnh trong ô | Tỉ lệ điểm ảnh trong ô mà mô hình xếp vào lớp cỏ |
| Tiêu chí giảm thuốc | Giảm ít nhất 30% | Giảm ít nhất 60% mức giảm tối đa có thể (ở ngưỡng 1%, đúng bằng 30% như trước) |

- **Vì sao đổi cách tính lượng cỏ dự đoán:** với cách cũ, các xác suất cỏ rất nhỏ trên 40 000 điểm ảnh của một ô cộng lại thành một "mức nền", trong khi ngưỡng phun tính theo cách cũ ở mức 0,1% chỉ là 4,2–5,9 cm².
  - 7 ô không có cỏ theo nhãn vẫn được chấm 1,0 cm² cỏ (trung vị; cao nhất 3,4 cm²).
  - Trong phép kiểm chứng bằng mô hình, ô bị thay toàn bộ bằng đất trống vẫn được chấm khoảng 3,1 cm² (trung vị của 7 ô).
  - Với cách mới, chính 7 ô không có cỏ ở trên được chấm 0 cm² (trung vị; cao nhất 0,4 cm²).
- **Vì sao đổi tiêu chí giảm thuốc:** ở ngưỡng 0,1%, ngay cả khi biết trước ô nào cần phun cũng chỉ giảm được 10,2%, nên mức cố định 30% không cách nào đạt được.
- Các con số của bản 11/11 (ví dụ "giảm khoảng một nửa diện tích phun") là của ngưỡng 1% với cách tính cũ, không so trực tiếp được với bản này.

---

## 1. Đặt vấn đề

- **Phun thuốc trừ cỏ cho cả ruộng gây tốn kém và hại môi trường.** Cách khắc phục là phun theo vị trí, chỉ phun chỗ cần (Celikkan và cộng sự, 2023).
- **Mật độ cỏ khác nhau nhiều giữa các chỗ.** Trên tập kiểm tra của đề tài, một nửa số ô 50 cm × 50 cm có cỏ dưới 1% diện tích, nhưng chỉ 22/216 ô có cỏ dưới 0,1%.
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

**Khi nào một ô "cần phun":** khi cỏ chiếm từ 0,1% diện tích ô trở lên, tức khoảng 2,5 cm² trong ô 50 cm × 50 cm.
- Mức này nằm trong khoảng ngưỡng kinh tế 0,06–0,31% diện tích cỏ mà các chuyên gia tư vấn đặt khi đánh giá hơn 2.500 ảnh (Longchamps và cộng sự, Weed Science 2014; đề tài mới đọc phần tóm tắt).
- Lưu ý: ảnh trong nghiên cứu đó rộng 2 m × 3 m, còn ô của đề tài là 0,5 m × 0,5 m, nên chỉ mượn được mức phần trăm, không chuyển thẳng ngưỡng giữa hai kích thước được.
- Kết quả còn được tính ở 7 mức để xem kết luận có đổi theo ngưỡng không: có cỏ là phun (ngưỡng 0, như thí nghiệm phun cỏ trên ngô của Allmendinger và cộng sự, Plants 2024), 0,06%, 0,1%, 0,31%, 0,5%, 1%, 2%.

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
- Mỗi điểm ảnh được xếp vào lớp có xác suất cao nhất: đất, ngô hoặc cỏ.

**Bước 2. Quyết định cho từng ô.**
- Tính **lượng cỏ dự đoán** trong ô: tỉ lệ điểm ảnh của ô mà mô hình xếp vào lớp cỏ, xem như ước lượng % diện tích ô là cỏ.
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
- **Dải phổ nào quan trọng:** tính bằng phương pháp Integrated Gradients (IG; Sundararajan và cộng sự, ICML 2017). IG cần đạo hàm, nên nó giải thích xác suất cỏ của ô, đại lượng trơn đi cùng lượng cỏ dự đoán.
- **Kiểm tra lời giải thích có trung thực không** (tức là phản ánh đúng lý do mô hình quyết định): xóa vùng mà lời giải thích chỉ ra, rồi so với xóa cùng số vùng chọn ngẫu nhiên. Lời giải thích trung thực thì xóa vùng của nó phải làm thay đổi kết quả nhiều hơn hẳn xóa ngẫu nhiên.

## 4. Kết quả đạt được hiện tại

Kết quả tính trên **216 ô** của tập kiểm tra, thuộc 3 đợt bay trước ngày phun thuốc. Ở ngưỡng 0,1%, **194 ô thật sự cần phun** (theo nhãn) và 22 ô không cần.

### 4.1. Tìm cỏ trên ảnh: khớp bài báo gốc

- Chạy mô hình của tác giả bằng code của đề tài cho ra **đúng số của bài báo**: mIoU 82,90% với ảnh 5 dải và 79,33% với ảnh màu thường (RGB). Như vậy code đánh giá của đề tài là đúng.
- Đề tài tự huấn luyện 7 lần, cùng batch 8 như bài báo, với 3 lịch huấn luyện (100 epoch như tác giả, learning rate giảm dần, dừng sớm). Kết quả: mIoU 79,25–83,57% (trung bình 81,20%).
  - Số của bài báo (82,90%) nằm trong khoảng này, nên chênh lệch chủ yếu do dao động ngẫu nhiên giữa các lần huấn luyện.
  - Còn một khác biệt chưa thử riêng: đề tài huấn luyện ở chế độ fp16 (nhanh hơn, tốn ít bộ nhớ GPU hơn), còn tác giả dùng fp32.

### 4.2. Quyết định phun: mức sai trong giới hạn, nhưng với mô hình tốt thì chưa hơn ngưỡng cố định

Mô hình của tác giả, ảnh 5 dải, ngưỡng "cần phun" 0,1%:

| Cách làm | Giảm thuốc | Chưa chắc | Bỏ sót | Phun nhầm |
|---|---|---|---|---|
| Phun cả ruộng | 0% | 0% | 0% | 100% (22/22 ô) |
| Biết trước đáp án (mức tốt nhất có thể) | 10,2% | 0% | 0% | 0% |
| Ngưỡng cố định 0,1% | 15,3% | 0% | 6,2% (12/194 ô) | 4,5% (1/22 ô) |
| **Đề tài (hai ngưỡng conformal)** | **16,2%** | **3,2% (7 ô)** | **7,2% (14/194 ô)** | **4,5% (1/22 ô)** |

**Ý chính:**
- Ở ngưỡng 0,1%, cỏ có ở gần khắp các ô, nên phun theo ô giảm được ít: tối đa 10,2%.
- Hai cách quyết định đều "giảm" nhiều hơn mức tối đa (15,3% và 16,2%) là vì có bỏ sót: ô cần phun mà không phun cũng làm giảm diện tích phun. Vì vậy con số giảm thuốc phải đọc cùng tỉ lệ bỏ sót.
- Bỏ sót và phun nhầm đều dưới giới hạn 10%; mô hình đạt cả 3 tiêu chí đề ra. Ngưỡng phun hiệu chỉnh được là 3,4–4,4 cm² cỏ dự đoán và ngưỡng không phun 2,1–3,6 cm² trên ô 2 500 cm² (mỗi phần vị trí của ruộng có một cặp ngưỡng).
- **Với mô hình chính, ở ngưỡng này cách của đề tài chưa tốt hơn ngưỡng cố định:** bỏ sót nhiều hơn 2 ô (14 so với 12), phun nhầm bằng nhau, và có 7 ô phải ra xem.
- **Khi mô hình yếu hơn, cách của đề tài mới phát huy tác dụng.** Với mô hình RGB (mIoU 79,33%), ngưỡng cố định bỏ sót 28,9% (56/194 ô), vượt xa giới hạn; cách của đề tài không bỏ sót ô nào, nhưng 36,1% số ô (78 ô) phải ra xem, vượt tiêu chí "chưa chắc không quá 30%".

**Thay đổi ngưỡng "cần phun"** (mô hình 5 dải):

| Ngưỡng | Ô cần phun | Ngưỡng cố định: bỏ sót / phun nhầm | Đề tài: bỏ sót / phun nhầm / chưa chắc |
|---|---|---|---|
| Có cỏ là phun | 209/216 | 1,9% / 28,6% | 10,0% / 0% / 87,0% |
| 0,06% | 201/216 | 5,0% / 0% | 5,5% / 0% / 45,4% |
| **0,1%** | **194/216** | **6,2% / 4,5%** | **7,2% / 4,5% / 3,2%** |
| 0,31% | 164/216 | 7,9% / 9,6% | 9,1% / 5,8% / 9,7% |
| 0,5% | 144/216 | 8,3% / 8,3% | 9,0% / 5,6% / 4,6% |
| 1% | 108/216 | 6,5% / 5,6% | 2,8% / 2,8% / 5,6% |
| 2% | 75/216 | 1,3% / 6,4% | 1,3% / 0% / 8,8% |

- Với mô hình 5 dải, ở ngưỡng 0,31%, 0,5% và 2%, cách của đề tài phun nhầm ít hơn ngưỡng cố định (5,8%, 5,6% và 0% so với 9,6%, 8,3% và 6,4%), đổi lại bỏ sót bằng hoặc nhiều hơn một chút. Ở ngưỡng 1% nó hơn ở cả hai chỉ số (bỏ sót 2,8% so với 6,5%, phun nhầm 2,8% so với 5,6%). Ở ngưỡng 0,1% hai cách gần như nhau.
- Ở ngưỡng rất thấp (có cỏ là phun, 0,06%), gần như mọi ô đều cần phun: cả tập kiểm tra chỉ có 7 và 15 ô không cần phun.
  - Muốn đặt ngưỡng phun với giới hạn phun nhầm 10%, mỗi lần tính cần ít nhất 9 ô như vậy ở phần vị trí còn lại. Thiếu số đó thì không ô nào ở phần vị trí đang xét được quyết định PHUN.
  - Vì vậy cách của đề tài để 87,0% và 45,4% số ô ở trạng thái CHƯA CHẮC; ở mức "có cỏ là phun" không có ô PHUN nào.
- Tính trên 4 mô hình (mô hình 5 dải và RGB của tác giả, 2 lần chạy thử của đề tài) và 7 mức ngưỡng, số mô hình có bỏ sót hoặc phun nhầm vượt 10% là: với ngưỡng cố định 4, 2, 2, 3, 3, 2, 0; với cách của đề tài 2, 0, 0, 0, 2, 0, 0.
- Thay bằng 11 mô hình tự huấn luyện khác nhau (mIoU 66,93–83,74%), ở ngưỡng 0,1%: 10/11 mô hình đạt cả 3 tiêu chí (mô hình còn lại trượt vì mIoU thấp); bỏ sót 3,6–10,3%, phun nhầm 0–9,1%, chưa chắc 0–6,9%. Một mô hình bỏ sót 10,3% (20/194 ô), vượt giới hạn đúng một ô: bảo đảm conformal chỉ đúng theo trung bình, nên một lần đo có thể vượt.

### 4.3. Lời giải thích: đã kiểm chứng là trung thực

**Kiểm tra độ trung thực của lời giải thích** trên 181 ô PHUN và CHƯA CHẮC. Mỗi lần xóa 9 vùng 5 cm × 5 cm, tổng diện tích bằng 9% một ô. IG và Grad-CAM chọn những vùng chúng xếp cao nhất trên cả ô ảnh 1,5 m, còn đối chứng chọn ngẫu nhiên ngay trong ô:

| Cách chọn vùng để xóa | Lượng cỏ dự đoán giảm | Ô PHUN bị đổi quyết định |
|---|---|---|
| Theo lời giải thích IG | 71,5% | 36% |
| Theo Grad-CAM (Selvaraju và cộng sự, ICCV 2017) | 5,2% | 1% |
| Chọn ngẫu nhiên trong ô | 1,9% | 1% |
| *Mốc so sánh: theo chính các điểm ảnh cỏ của mô hình trong ô* | *78,0%* | *48%* |

- **IG trung thực:** ở 180/181 ô, xóa theo IG làm lượng cỏ dự đoán giảm nhiều hơn xóa ngẫu nhiên trong ô.
- **Grad-CAM không dùng được** để giải thích từng ô.
  - So từng ô, nó chỉ hơn xóa ngẫu nhiên trong ô ở 84/181 ô (46%), gần như tung đồng xu.
  - Lý do: bản đồ Grad-CAM của các ô trong cùng một ô ảnh gần như giống hệt nhau (hệ số tương quan trung vị 0,983), nên nó không chỉ ra được vùng riêng của từng ô.
- IG không hơn mốc "các điểm ảnh cỏ của mô hình" (71,5% so với 78,0%). Điều này dễ hiểu, vì lượng cỏ dự đoán chính là số điểm ảnh đó. IG có thêm thông tin về dải phổ: dải quan trọng nhất là **rìa đỏ (RE)**, đứng đầu ở 167/181 ô.
- *"Trung thực" nghĩa là lời giải thích phản ánh đúng cách mô hình quyết định. Nó không có nghĩa là mô hình nhận đúng cỏ.*

**Giải thích "đổi gì thì quyết định đổi"** (đã cho mô hình chấm lại để kiểm chứng):
- **11 ô PHUN "mong manh" (sát ngưỡng):** chỉ cần bỏ **một** vùng cỏ 5 cm là hệ thống không còn quyết định PHUN; 25 ô với tối đa hai vùng. Người dùng nên được báo về những ô này.
- **Cả 7 ô CHƯA CHẮC:** bỏ 1 vùng (5 ô), 2 vùng (1 ô) hoặc 3 vùng (1 ô) là thành KHÔNG PHUN.
  - Vì vậy người kiểm tra biết ngay **nên xem chỗ nào trước**: 1–3 vùng 5 cm mà quyết định phụ thuộc vào.
  - Đây là chỗ xem trước, không thay cho việc xem cả ô, vì cỏ mà mô hình bỏ sót có thể nằm ở chỗ khác.
- **Hai cách tính:**
  - *ước lượng nhanh* (trừ số điểm ảnh cỏ của các vùng bị bỏ, không cần chạy lại mô hình) đúng chính xác ở 68/174 ô PHUN (39%) và lệch không quá 1 vùng ở 114/174 ô (66%). Nó lệch theo cả hai phía: mô hình cần bỏ nhiều vùng hơn ở 56 ô, ít hơn ở 50 ô. Ở các ô mà ước lượng chỉ cần bỏ không quá 2 vùng, ước lượng đúng ở 20/25 ô;
  - mô hình không phải lúc nào cũng "càng bỏ càng ít cỏ": 29 ô PHUN đã hết PHUN rồi lại thành PHUN khi bỏ thêm vùng (thêm 1 ô CHƯA CHẮC bị chuyển thành PHUN khi bỏ vùng), và 4 ô PHUN bỏ hết 100 vùng vẫn chưa thành KHÔNG PHUN;
  - vì vậy lời giải thích chính thức dùng số **đã kiểm chứng bằng mô hình**, mất khoảng 1,2 giây mỗi ô trên GPU T4 của Colab.

Ví dụ ba ô, được chọn theo quy tắc cố định:

![Ví dụ giải thích cho ba ô](../data/weedsgalore/hinh_giai_thich_don_gian.png)

### 4.4. Hạn chế cần lưu ý

- **Tập kiểm tra nhỏ** (216 ô, 8 vị trí). Ở ngưỡng 0,1% chỉ có 22 ô không cần phun, nên mỗi ô phun nhầm đã là 4,5%. Mỗi lần tính ngưỡng chỉ có 9 và 13 ô không cần phun, vừa đủ mức tối thiểu 9 ô để đặt được ngưỡng phun.
- **Một ruộng, một mùa vụ.** Chưa kiểm tra trên ruộng khác.
- **Ở ngưỡng thấp, lợi ích của phun theo ô nhỏ**, vì cỏ có ở gần mọi ô: tối đa 10,2% ở ngưỡng 0,1%.
- **Kích thước ô 50 cm chưa có căn cứ.** Nó nên khớp với vùng mà máy phun điều khiển riêng được. Thử ô 25 cm (cùng ngưỡng 0,1%): diện tích phun giảm 30,1% (tối đa có thể 25,9%), 16/864 ô CHƯA CHẮC, nhưng bỏ sót 8,9% và phun nhầm 9,4%, sát giới hạn 10%. Hai kết quả không so sánh trực tiếp hoàn toàn được: cùng ngưỡng 0,1%, nhưng ô nhỏ chỉ cần ít cỏ hơn là đã "cần phun".
- **Hiệu chỉnh riêng cho từng đợt bay chưa dùng được.** Ở cả ba đợt bay, tỉ lệ ô CHƯA CHẮC đều vượt 30%: 37,5% ngày 25/5, 90,3% ngày 30/5 và 91,7% ngày 6/6. Ở hai ngày sau, gần như mọi ô đều cần phun (70/72 và 72/72 ô), nên không đủ ô "không cần phun" để tính ngưỡng.

## 5. Việc tiếp theo và điểm cần thầy góp ý

- Viết bài báo 12–15 trang trên Overleaf theo chỉ đạo của thầy, có thực nghiệm đối sánh với các nghiên cứu khác.
- Cách trình bày kết quả chính: ở ngưỡng 0,1% với mô hình tốt, cách quyết định của đề tài chưa hơn ngưỡng cố định; ưu điểm thể hiện ở mô hình yếu hơn và ở ngưỡng 1%. Đề tài dự định trình bày đúng như vậy, kèm bảng các mức ngưỡng ở mục 4.2.
- Ba điểm chờ thầy xác nhận: cỏ dại có được tính là dịch hại không; dữ liệu một ruộng, một mùa vụ có đủ không; dữ liệu công bố ở hội nghị (WACV 2025) có được chấp nhận không.
