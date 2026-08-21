# Lab 21 — Evaluation Report

**Họ tên**: Hoàng Lê Minh

**MSSV**: 2A202601653

**Ngày**: 2026-08-21

**Tier**: `CPU` (dùng để chọn model nhẹ)

**Base model**: `Qwen/Qwen3.5-0.8B`

**GPU thực tế**: NVIDIA GeForce RTX 3060 Laptop GPU 6 GB, CUDA 12.8, bf16 native

> Mọi số liệu trong báo cáo được lấy từ `results/`. Tier `CPU` chỉ là khóa cấu hình để chọn
> model 0.8B; `device_map="auto"` đã nạp và huấn luyện model trên GPU thực tế. Tôi chọn model
> này để hoàn thành phép thử đầy đủ trên máy 6 GB VRAM, không dùng `EVAL_LIMIT`.

---

## 1. Setup

| | |
|---|---|
| Dataset | Corpus mặc định: 250 ticket CSKH tiếng Việt → JSON triage |
| Train / val | 225 / 25 (seed 42) |
| Eval | 50 target / 15 regression, không rút gọn |
| `max_length` | 256 — p95 đo được là 98 token; 256 là mức lũy thừa hai kế tiếp |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 1 epoch / 29 optimizer steps |
| Batch hiệu dụng | 1 × 8 gradient accumulation = 8 |

**Template có giữ khối `<think>` không?** Có. `template_check.json` ghi nhận đủ thẻ mở và
nội dung reasoning, với verdict `reasoning preserved — safe to train on traces`. Corpus mặc định
chỉ có câu trả lời JSON trần, không có reasoning trace thật; vì vậy tôi giữ `assistant-only` và
không diễn giải `valid_trace_rate` như một chỉ số chất lượng reasoning.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.3936 (37/94 token ở mẫu kiểm tra) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn được tính loss chỉ có một dòng JSON và token kết thúc, đúng với câu trả lời trần trong corpus:

```text
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Phần system prompt, ticket của người dùng và khối `<think>` rỗng đều nằm trong
`masked_preview`, không tham gia loss.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms/mẫu) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.0000 | 0.6778 | 0.0000 | 2157.2 |
| (b) base + optimized prompt | 0.5000 | 0.6778 | 1.0000 | 600.9 |
| (c) LoRA fine-tune | 0.7250 | 0.1333 | 1.0000 | 928.5 |

**(b) có thật sự mạnh hơn (a) không?** Có. Target tăng 0.500 và format tăng từ 0 lên 1,
trong khi regression không đổi. Tôi không sửa `OPTIMIZED_PROMPT`; SHA đóng băng là
`719e74d3b6232053`. Do đó baseline (b) là một mốc mạnh được đo trước huấn luyện, không bị làm yếu
để giúp fine-tune trông tốt hơn.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss | target (NB5 §4) | thời gian (s) | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 10,822,656 | 1e-4 | 0.6680 | 0.7250 | 352.0 | 3.07 |
| `attn_only` | q,v | 271 (matched) | 10,822,656 | 1e-4 | 0.7337 | 0.7400 | 275.8 | 3.08 |
| `wrong_lr` | text-linear | 16 | 10,822,656 | 1e-5 | 2.2514 | 0.0000 | 321.3 | 3.08 |
| `qlora` | text-linear | 16 | 10,822,656 | 1e-4 | 0.7074 | 0.6700 | 334.6 | 2.29 |

### 4.1 — Vị trí adapter so với rank

`attn_only` có đúng cùng 10,822,656 tham số trainable với `correct`, vì rank được nâng từ 16 lên
271 để khớp ngân sách. Trên target, nó **thắng nhẹ** `correct` (0.740 so với 0.725), dù train loss
lại xấu hơn (0.7337 so với 0.6680). Như vậy thứ tự theo năng lực tác vụ đảo so với thứ tự theo
loss; trên bài triage hẹp này không có bằng chứng rằng text-linear tốt hơn attention-only. Kết quả
cũng cho thấy rank cao tự nó không phải lời giải thích đầy đủ: rank ở đây chỉ bù ngân sách, còn
vị trí gắn adapter và cấu trúc tác vụ quyết định cách ngân sách đó được sử dụng.

### 4.2 — Learning rate sai

`wrong_lr` chỉ hạ LR mười lần, từ 1e-4 xuống 1e-5, còn mask, placement, rank, dữ liệu và 29 step
đều giữ nguyên. Loss của run này giảm rất chậm: log đầu là 3.014 và train loss tổng hợp cuối vẫn
2.2514, so với 0.6680 của `correct`; target và format đều bằng 0. Nếu chỉ nhìn loss mà không biết
LR và số step, tôi có thể kết luận sai rằng model hoặc dữ liệu không học được. Đối chứng cho thấy
nguyên nhân trực tiếp là LR ở thang full fine-tune không đủ để LoRA học hợp đồng JSON trong ngân
sách ngắn, chứ không phải tăng rank hay sửa eval.

### 4.3 — QLoRA

QLoRA giảm peak VRAM từ 3.07 GB xuống 2.29 GB, tiết kiệm 0.78 GB, tương đương khoảng 25.4%.
Cái giá là target giảm từ 0.725 xuống 0.670, format giảm từ 1.00 xuống 0.98, còn latency tăng từ
928.5 lên 6010.1 ms/mẫu (xấp xỉ 6.47 lần) trên máy Windows/RTX 3060 này. Train time cũng không
nhanh hơn đáng kể (334.6 so với 352.0 giây). Vì model bf16 đã vừa trong 6 GB VRAM, số đo của tôi
ủng hộ khuyến nghị không dùng QLoRA cho cấu hình này: phần bộ nhớ tiết kiệm không bù được giảm
chất lượng và chi phí inference.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`

`target Δ = +0.2250` · `regression Δ = -0.5444` · `valid_trace_rate = 0.00`

Fine-tune học tác vụ đích rõ ràng: target tăng từ 0.500 của prompt tối ưu lên 0.725, đồng thời giữ
format ở 1.000. Tuy nhiên đây không phải một chiến thắng có thể deploy, vì regression giảm từ
0.6778 xuống 0.1333, lớn hơn rất nhiều ngưỡng cho phép 0.02. Cổng FAILED vì quên năng lực chung,
không phải vì target không tiến bộ. Nguyên nhân hợp lý nhất là toàn bộ 225 mẫu train đều là triage
JSON và không có 1–5% replay dữ liệu phổ thông; sau 29 step, model nhỏ 0.8B bị ép mạnh vào một kiểu
đầu ra. `valid_trace_rate=0` không chứng minh reasoning collapse trong lượt này, vì generation đã
tắt thinking và corpus không có trace. Tôi không nới ngưỡng hay sửa eval sau khi thấy kết quả.
Hướng sửa hợp lệ là thêm replay data vào train, huấn luyện lại cùng seed/budget và đo lại trên đúng
tập target/regression đã đóng băng.

---

## 6. Định tính — có cả ca THUA

Các giá trị trong bốn trường được viết theo thứ tự `intent / urgency / product / sentiment`.
Điểm chi tiết và output JSON đầy đủ nằm trong `results/qualitative_comparison.json`.

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---:|---|---|---|---|---|
| 1 | “Bình giữ nhiệt… Chưa thấy tiền. Khi nào tiện…” | hoan_tien / thap / bình giữ nhiệt / tich_cuc | hoan_tien / **cao** / đúng product / tich_cuc (0.75) | đúng cả 4 trường (1.00) | ✅ FT thắng: sửa urgency |
| 2 | “Chuột không dây… Bảo hành bao lâu. Không vội…” | hoi_thong_tin / thap / chuột không dây / tich_cuc | **hoan_tien / trung_binh / chữ ký không dây** / tich_cuc (0.25) | đúng cả 4 trường (1.00) | ✅ FT thắng: sửa intent, urgency, product |
| 3 | “Đèn bàn LED… Không hoạt động. Cần trước ngày mai. Quá tệ.” | san_pham_loi / cao / đèn bàn LED / tieu_cuc | **van_chuyen** / cao / đúng product / **tich_cuc** (0.50) | đúng cả 4 trường (1.00) | ✅ FT thắng: sửa intent và sentiment |
| 4 | “Đèn bàn LED… Hoàn tiền. Quá hạn rồi. Cảm ơn…” | hoan_tien / cao / đèn bàn LED / tich_cuc | đúng cả 4 trường (1.00) | hoan_tien / **trung_binh** / đúng product / tich_cuc (0.75) | ❌ **FT thua**: hạ sai urgency |
| 5 | “Sạc dự phòng… Giao hàng chậm. Không vội…” | van_chuyen / thap / sạc dự phòng / trung_tinh | van_chuyen / **cao** / đúng product / trung_tinh (0.75) | **san_pham_loi** / thap / đúng product / **tieu_cuc** (0.50) | ❌ **FT thua**: sai intent và sentiment |

Trên toàn bộ 50 ticket, fine-tune thắng 34, thua 7 và hòa 9 so với baseline (b). Các ca thua không
phải lỗi JSON hay product copying; chúng tập trung ở phân biệt mức urgency và một số câu có tín
hiệu intent/sentiment mơ hồ. Điều này phù hợp với format=1.0 nhưng target chỉ 0.725: model đã học
hình thức rất chắc, còn ranh giới ngữ nghĩa giữa các nhãn chưa ổn định.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi chưa nên deploy adapter này như một thay thế tổng quát cho base model. Nó chứng
minh LoRA có thể chuyển hành vi vào trọng số: với prompt ngắn, target tăng 0.225 so với base đã
được prompt tốt và mọi output đều đúng schema. Tuy nhiên mức regression giảm 0.5444 là một lỗi
triển khai nghiêm trọng; một model triage không nên đổi lấy gần như toàn bộ khả năng trả lời phổ
thông chỉ để tăng 22.5 điểm phần trăm trên target. Nguyên nhân gần nhất là chất lượng **hỗn hợp**
dữ liệu: corpus có 100% mẫu triage, không có replay phổ thông. Trong các nút kỹ thuật, mask là điều
kiện nền tảng—mask sai sẽ làm mọi kết quả vô nghĩa—còn learning rate là đòn bẩy mạnh nhất quan sát
được trong ngân sách này vì giảm LR mười lần làm target và format về 0. Vị trí adapter không cho
kết luận “all-linear luôn thắng”: đối chứng công bằng cho thấy attention-only thắng nhẹ 0.015 trên
tác vụ hẹp dù loss xấu hơn. QLoRA chỉ hữu ích nếu thật sự thiếu VRAM; ở đây nó tiết kiệm 0.78 GB
nhưng giảm target và làm inference chậm hơn khoảng 6.47 lần. Vì vậy bước tiếp theo hợp lý không
phải tăng rank hoặc nới cổng, mà là thêm replay data, huấn luyện lại và giữ nguyên eval đã đóng
băng để kiểm tra quan hệ nhân quả.

**Ba điều tôi học được**:

1. Tôi phải decode mask trước khi train: ở mẫu proof chỉ 39.36% token được giám sát; nếu dùng
   `everything`, 100% prompt sẽ vào loss và model có thể học lặp lại câu hỏi.
2. Prompt engineering là baseline bắt buộc: riêng prompt tối ưu đã đưa format 0 → 1 và target
   0 → 0.5, nên so fine-tune chỉ với prompt ngây thơ sẽ phóng đại lợi ích.
3. Train loss không phải bảng xếp hạng tác vụ: `attn_only` có loss 0.7337 cao hơn `correct` 0.6680
   nhưng target 0.740 lại cao hơn 0.725; phán quyết phải dựa vào eval đã đóng băng.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn 3% câu hỏi phổ thông từ miền regression vào tập train,
giữ nguyên seed 42, 29 steps và eval checksum; sau đó đo lại target, regression, format, latency để
xem replay có kéo regression lên mà không làm mất mức tăng target hay không.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng
- [ ] B3 reasoning-trace collapse
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
