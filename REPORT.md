# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`="cab" ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`="ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
 Record này nói về **cả bức ảnh** vì các nhãn là những lớp dự đoán cho cùng một ảnh. Trường image_width và image_height mô tả kích thước toàn bộ ảnh. Ngoài ra "taxonomy_name": "ImageNet-1K" cho biết bộ nhãn phân loại được sử dụng. Record không có tọa độ bounding box hay object_id để xác định vị trí và phân biệt từng vật thể riêng lẻ.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
 Danh sách tên lớp (class list) không phải do model tự nghĩ ra, mà được định nghĩa sẵn và đi kèm với checkpoint hoặc cấu hình của mô hình.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
 Vì để đảm bảo kết quả có thể được xác định và đối chiếu chính xác, việc lưu cả ba giúp tránh nhầm lẫn khi các taxonomy khác nhau có thể chứa những lớp trùng tên hoặc sử dụng ID khác nhau. Đồng thời, nó giúp kiểm tra tính chính xác của dự đoán, hỗ trợ chuyển đổi giữa các hệ thống phân loại, thuận tiện cho việc thống kê, so sánh kết quả giữa các mô hình và tái sử dụng dữ liệu trong tương lai mà không phải suy đoán lại tên lớp từ ID.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
 Guideline cần quy định rõ tiêu chí chọn lớp đại diện cho toàn bộ bức ảnh khi ảnh có nhiều vật thể hoặc chủ đề khác nhau: cần hướng dẫn người gán nhãn ưu tiên chủ thể chính, xác định chủ đề tổng thể của ảnh thay vì chọn các vật thể phụ. Nếu có nhiều chủ thể quan trọng ngang nhau, guideline cần quy định cách xử lý thống nhất, chẳng hạn chọn chủ thể ở trung tâm hoặc chủ thể thể hiện rõ nhất mục đích của bức ảnh.

6. Vì sao không được lấy `score` của model làm ground truth?
  Vì score chỉ thể hiện mức độ tin cậy của mô hình đối với một nhãn dự đoán, không chứng minh nhãn đó là chính xác. Score không phản ánh đánh giá thực tế của con người và không đảm bảo ảnh thực sự thuộc lớp được dự đoán.
  Model có thể dự đoán sai nhưng vẫn đưa ra score cao do hạn chế của dữ liệu huấn luyện.


## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`="person" ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
 Với record trên:
  góc trên-trái nằm ở (385.33, 69.24)
  góc dưới-phải nằm ở (498.92, 348.92)
  rộng: 113.58 pixel
  cao: 279.68 pixel
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
  Vd 2 ngưỡng 0.20 và 0.60
  0.20 có 0 predictions
  0.60 có 6 predictions
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
 Kiểm tra mỗi bounding box có thực sự chứa vật thể trong ảnh hay không, xác nhận model có phân loại đúng tên vật thể không, xác nhận hộp có bao quanh đúng vị trí và kích thước vật thể không, đánh dấu các trường hợp model nhận diện nhầm vật thể hoặc vùng không có vật thể, phát hiện các bounding box cùng nhận diện một vật thể, sửa class hoặc bounding box sai, bổ sung nhãn nếu quy trình yêu cầu.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
  Quy tắc: Vẽ một box cho mỗi vật thể, bao quanh toàn bộ phần nhìn thấy, sát đường viền nhất có thể, không cắt vật thể hoặc lấy thừa nền; gán đúng class label.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
  1 vật bị che khuất hoặc cắt mép là: person 0.61
  Em chọn hỏi lab coach vì vật thể bị che khuất hoặc cắt mép có thể cần cách xử lý thống nhất, không nên tự đặt quy tắc khi chưa có hướng dẫn chính thức, tránh gây sai lệch giữa những người gán nhãn.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`="kitchen-001" ; `class_name`="person" ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[
        446.0,
        70.0
      ],
      [
        445.0,
        71.0
      ],
      [
        444.0,
        71.0
      ],
      [
        443.0,
        72.0
      ],
      [
        442.0,
        72.0
      ],

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
 So với bounding box, polygon mô tả đường viền và hình dạng thực tế của vật thể chi tiết hơn. Với vật thể person trong ảnh kitchen, bounding box bao quanh người bằng hình chữ nhật, còn polygon gồm 348 điểm tọa độ mô tả đường viền cơ thể.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
 `instance_id` dùng để định danh duy nhất từng đối tượng được phát hiện trong một ảnh, giúp phân biệt các đối tượng khác nhau, kể cả khi chúng có cùng class
  Nó **không** phải class ID và **không** phải tracking ID: Câu này nghĩa là `instance_id` có nhiệm vụ riêng, không được nhầm lẫn với hai loại ID khác:
  Không phải class ID: Vì nó không đại diện cho loại vật thể (ví dụ person, car, oven).
  Không phải tracking ID: Vì nó không dùng để theo dõi cùng một vật thể qua nhiều ảnh hoặc frame video.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
 1 quy tắc vẽ biên mask: Phải vẽ polygon sát theo đường viền thực tế của vật thể, bao phủ toàn bộ phần nhìn thấy, không bao gồm nền hoặc vật thể khác.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
 **một** chỗ biên mờ / hai vật chạm nhau / bị che là: dining table
 Em chọn escalation vì chọn vùng hai vật chạm nhau hoặc bị che khuất, nơi đường biên không thể xác định rõ. Guideline hiện tại chưa quy định cụ thể cách xử lý trường hợp biên mờ hoặc vật thể bị che, nên cần hỏi Lab Coach để thống nhất cách gán nhãn.
 
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn class đại diện cho toàn ảnh (image-level label). | Ảnh có nhiều chủ thể nên khó chọn lớp chính. JSON có rank, class_id, class_name, score; đây là prediction, không chứng minh nhãn đúng. |Chọn class theo guideline, dựa trên chủ thể chính của ảnh.  | Kiểm tra class có phản ánh đúng nội dung ảnh và quy tắc chọn lớp có nhất quán không. |
| Phát hiện vật thể |Mỗi vật thể có một class và một bounding box (bbox_xyxy).  |  Vật thể nhỏ, bị che hoặc sát mép ảnh khó xác định box. JSON có score, class_id, bbox_xyxy; cần kiểm tra prediction sai class, box lệch hoặc vật thể bị bỏ sót.| Vẽ box bao quanh từng vật thể nhìn thấy, sát biên và gán đúng class. |  Kiểm tra vật thể bị thiếu, prediction thừa, class sai, box quá rộng/hẹp hoặc lệch vị trí.|
| Instance segmentation | Một polygon polygon_xy và một instance_id cho mỗi vật | Ảnh kitchen: dining table 0.58 chồng lên bowl 0.70 và 0.74; person 0.43 bị che; traffic-001 (bus) có 120 điểm polygon nên vẽ tay tốn công | Vẽ biên theo vật, tách riêng từng instance | Biên mask có khớp vật không, các instance chồng nhau có bị lẫn không |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ truy cập đúng dữ liệu cần thiết, đúng mục đích và không chia sẻ ra ngoài phạm vi được phép.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Tôi sẽ dừng xử lý dữ liệu đó và báo ngay cho Lab Coach hoặc người phụ trách dự án (Lab Lead/Supervisor) để được hướng dẫn. Không tự ý tiếp tục gán nhãn, chỉnh sửa, tải xuống hoặc chia sẻ dữ liệu.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [v ] `REPORT.md` (file này đã điền)
- [v] `day1_lab_outputs/classification_predictions.json`
- [v ] `day1_lab_outputs/detection_predictions.json`
- [v ] `day1_lab_outputs/segmentation_predictions.json`
- [v ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [v ] `day1_lab_outputs/visuals/classification_top5.png`
- [v ] `day1_lab_outputs/visuals/detection_predictions.png`
- [v ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [v ] Ô validation cuối notebook báo `PASS`
- [v ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [v ] Đã nộp **link repository** trên LMS
