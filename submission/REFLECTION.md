# Reflection — Lab 21

**Họ tên**: Hoàng Lê Minh — **MSSV**: 2A202601653

**1. Điều gì làm tôi ngạc nhiên nhất?**

Điều bất ngờ nhất là adapter `attn_only` có train loss cao hơn `correct` nhưng target lại nhỉnh hơn
(0.740 so với 0.725). Nếu chỉ nhìn loss, tôi đã xếp hạng sai hai cấu hình. Tôi cũng không ngờ prompt
tối ưu một mình có thể đưa format từ 0 lên 1 và target từ 0 lên 0.5 trên model chỉ 0.8B.

**2. Tôi mất nhiều thời gian nhất ở đâu? Có đúng như dự đoán không?**

Phần tốn thời gian nhất là cài PyTorch CUDA 2.75 GB và chạy bốn lượt train thật, sau đó là QLoRA
inference rất chậm. Tôi ban đầu nghĩ train sẽ là nút thắt duy nhất, nhưng QLoRA mất khoảng 301 giây
để sinh 50 target trong khi adapter bf16 đúng chỉ mất khoảng 46 giây ở lượt NB5.

**3. Trước lab tôi tin điều gì về fine-tuning mà giờ không còn tin?**

Tôi không còn tin rằng loss thấp hơn đồng nghĩa model tốt hơn, hoặc fine-tune thắng prompt ngây thơ
là đủ để deploy. Lượt chạy này tăng target nhưng làm regression giảm 0.5444, nên một “win” trên tác
vụ đích vẫn có thể là một model tệ hơn về mặt sản phẩm.

**4. Tôi dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng AI assistant để đọc rubric, chọn model 0.8B phù hợp GPU 6 GB, dựng môi trường CUDA, theo
dõi từng run, kiểm tra checksum và tổng hợp report từ artefact. Sai sót ban đầu của assistant là
chưa dự đoán PowerShell dùng mã hóa cp1252, khiến NB1 dừng với `UnicodeEncodeError`; sau đó phải
chạy lại với `PYTHONUTF8=1`. Assistant cũng phải kiểm chứng thay vì tin báo FAIL checksum: dữ liệu
không hề đổi, chỉ khác CRLF/LF trên Windows.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên tôi làm là gì?**

Tôi sẽ đóng băng một eval set đại diện và đo base model với cả prompt ngây thơ lẫn prompt tối ưu
trước khi tạo adapter. Đồng thời tôi sẽ kiểm tra bằng mắt phần token được tính loss và chuẩn bị một
lượng replay data nhỏ để cổng regression có cơ hội đạt ngay từ thiết kế dữ liệu.
