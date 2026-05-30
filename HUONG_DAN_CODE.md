# Hướng dẫn đọc code dự án AI cảnh báo rủi ro giao hàng trễ

Tài liệu này giải thích dự án theo góc nhìn dễ hiểu cho người mới học kinh tế, quản trị, logistics hoặc phân tích dữ liệu. Mục tiêu là giúp bạn hiểu ứng dụng đang làm gì, dữ liệu đi qua những bước nào, và mỗi phần code có vai trò gì.

## 1. Ứng dụng này giải quyết bài toán gì?

Ứng dụng mô phỏng một hệ thống hỗ trợ quyết định trong logistics và thu mua.

Bài toán nghiệp vụ:

- Doanh nghiệp cần nhập nguyên liệu/hàng hóa đúng hạn.
- Nếu giao hàng trễ, doanh nghiệp có thể bị thiếu hàng, chậm sản xuất hoặc giảm chất lượng phục vụ.
- Ứng dụng nhận thông tin đơn hàng/vận chuyển, sau đó dự đoán rủi ro giao trễ.
- Kết quả giúp người quản lý biết đơn nào cần theo dõi, đơn nào cần cảnh báo.

Kết quả chính gồm:

- `Risk Score`: điểm rủi ro từ 0 đến 100%.
- `Mức rủi ro`: Thấp, Trung bình, Cao.
- `Quyết định cảnh báo`: Không cảnh báo, Cần theo dõi, Cảnh báo, Cảnh báo khẩn.
- `Lý do rủi ro`: vì sao hệ thống đánh giá rủi ro như vậy.
- `Gợi ý hành động`: nên làm gì tiếp theo.

## 2. Cấu trúc thư mục

```text
ai-powered-delivery-delay-risk-warning/
├── api/
│   └── index.py
├── model/
│   ├── feature_columns.pkl
│   └── random_forest_model.pkl
├── templates/
│   └── index.html
├── HUONG_DAN_CODE.md
├── README.md
├── requirements.txt
└── vercel.json
```

Ý nghĩa:

- `api/index.py`: backend Flask, nhận dữ liệu, gọi model, tính cảnh báo, xử lý CSV.
- `templates/index.html`: giao diện web, form nhập tay, upload CSV, bảng kết quả.
- `model/random_forest_model.pkl`: model Random Forest đã train.
- `model/feature_columns.pkl`: danh sách cột sau khi xử lý dữ liệu lúc train model.
- `requirements.txt`: thư viện Python cần cài.
- `vercel.json`: cấu hình để deploy lên Vercel.

## 3. Luồng hoạt động tổng quát

Ứng dụng có 2 chế độ:

1. Dự đoán nhập tay.
2. Dự đoán hàng loạt bằng CSV.

### 3.1. Dự đoán nhập tay

Luồng xử lý:

```text
Người dùng nhập form
-> JavaScript gom dữ liệu thành JSON
-> Gửi POST /predict
-> Flask kiểm tra dữ liệu
-> Gọi model AI hoặc demo scoring
-> Tính Risk Score
-> Áp dụng business rules
-> Trả kết quả về frontend
-> Giao diện hiển thị kết quả
```

### 3.2. Dự đoán bằng CSV

Luồng xử lý:

```text
Người dùng upload file CSV
-> JavaScript gửi file đến POST /predict-batch
-> Flask đọc CSV bằng pandas
-> Mỗi dòng CSV được xử lý như một đơn hàng riêng
-> Dòng nào hợp lệ thì trả kết quả dự đoán
-> Dòng nào lỗi thì trả lỗi riêng cho dòng đó
-> Frontend hiển thị bảng kết quả
-> Người dùng có thể lọc, sắp xếp, xem chi tiết, xuất CSV kết quả
```

## 4. Giải thích file `api/index.py`

Đây là phần backend. Backend giống như "bộ não xử lý" phía sau giao diện.

### 4.1. Khởi tạo Flask app

```python
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
CORS(app)
```

Ý nghĩa:

- `Flask(...)`: tạo web server nhỏ bằng Python.
- `template_folder`: chỉ cho Flask biết file HTML nằm trong thư mục `templates`.
- `CORS(app)`: cho phép frontend gọi API thuận tiện hơn.

### 4.2. Load model

```python
MODEL_PATH = BASE_DIR / "model" / "random_forest_model.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "model" / "feature_columns.pkl"
```

Ứng dụng cần 2 file:

- Model Random Forest.
- Danh sách cột feature lúc train.

Vì khi train model, dữ liệu đã được biến đổi thành nhiều cột số. Khi dự đoán, dữ liệu mới cũng phải có đúng cấu trúc cột như lúc train.

### 4.3. Các cột đầu vào

`TRAIN_INPUT_COLUMNS` là danh sách các thông tin đầu vào mà model cần, ví dụ:

- Vĩ độ, kinh độ.
- Số hàng còn tồn kho.
- Nhiệt độ, độ ẩm.
- Tình trạng giao thông.
- Thời gian chờ hàng.
- Giá trị giao dịch.
- Tần suất mua hàng.
- Mức độ ưu tiên.

Về mặt kinh tế/logistics, đây là các yếu tố có thể ảnh hưởng đến rủi ro giao trễ.

Ví dụ:

- Tồn kho thấp làm rủi ro cao hơn.
- Giao thông ùn tắc làm rủi ro cao hơn.
- Nhu cầu cao làm áp lực cung ứng tăng.
- Hàng ưu tiên cao thì tác động trễ hàng nghiêm trọng hơn.

### 4.4. Kiểm tra số âm

Hàm:

```python
kiem_tra_so_khong_am(data)
```

Mục đích:

- Không cho các trường số nhập âm.
- Ví dụ tồn kho, độ ẩm, nhu cầu, giá trị giao dịch không nên là số âm.

Nếu người dùng nhập số âm, API trả lỗi thay vì dự đoán sai.

### 4.5. Chuẩn hóa dữ liệu cho model

Hàm:

```python
chuan_hoa_du_lieu_cho_model(data)
```

Hàm này làm các việc:

1. Nếu người dùng chọn kho, hệ thống đổi kho thành tọa độ.
2. Nếu có thời gian giao, hệ thống tách ra giờ, tháng, thứ trong tuần.
3. Chuyển dữ liệu thành bảng pandas.
4. One-hot encoding các cột dạng chữ.
5. Sắp xếp lại cột đúng như model đã train.

Ví dụ cột chữ `Tình trạng giao thông` có thể được biến thành các cột số:

```text
Tình trạng giao thông_Ùn tắc
Tình trạng giao thông_Đường vòng
```

Model không hiểu chữ trực tiếp, nên phải đổi chữ thành số.

### 4.6. Dự đoán bằng model thật

Hàm:

```python
du_doan_bang_model_that(data)
```

Ý nghĩa:

- Chuẩn hóa dữ liệu.
- Gọi model Random Forest.
- Lấy xác suất rủi ro giao trễ.
- Đổi xác suất thành phần trăm.

Ví dụ model trả `0.72`, ứng dụng hiển thị `72%`.

### 4.7. Demo scoring

Hàm:

```python
du_doan_demo(data)
```

Đây là phương án dự phòng khi không có file model.

Nó không phải AI thật, mà là công thức mô phỏng:

- Ùn tắc: cộng rủi ro.
- Thời gian chờ vượt ngưỡng: cộng rủi ro.
- Tồn kho thấp: cộng rủi ro.
- Mức sử dụng phương tiện cao: cộng rủi ro.
- Nhu cầu cao: cộng rủi ro.
- Hàng ưu tiên cao: cộng rủi ro.

Hàm này giúp ứng dụng vẫn chạy được khi chưa có model thật.

### 4.8. Xác định mức rủi ro

Hàm:

```python
xac_dinh_muc_rui_ro(risk_score_percent)
```

Logic:

```text
>= 70%  -> Cao
>= 40%  -> Trung bình
< 40%   -> Thấp
```

Đây là cách biến một con số thành nhãn dễ hiểu cho người dùng kinh doanh.

### 4.9. Business rules

Hàm:

```python
ap_dung_business_rules(data, risk_score_percent)
```

Model chỉ đưa ra xác suất. Nhưng trong thực tế kinh doanh, quyết định không chỉ dựa vào xác suất. Doanh nghiệp còn quan tâm:

- Hàng có ưu tiên cao không?
- Tồn kho có dưới mức an toàn không?
- Thời gian chờ có vượt ngưỡng không?
- Hàng có quan trọng không?

Vì vậy app kết hợp AI với business rules.

Logic hiện tại:

```text
Risk >= 70% -> Cảnh báo
Risk 40-69% -> Cần theo dõi
Risk < 40% -> Không cảnh báo
```

Một số trường hợp đặc biệt có thể nâng mức cảnh báo:

- Hàng ưu tiên cao + risk cao + tồn kho thấp -> Cảnh báo khẩn.
- Hàng ưu tiên thấp chỉ cảnh báo khi risk rất cao và thời gian chờ vượt ngưỡng.

### 4.10. Lý do rủi ro và gợi ý hành động

Hàm:

```python
phan_tich_rui_ro_va_goi_y(data, risk_score_percent)
```

Hàm này không thay model, mà giải thích thêm cho người dùng.

Ví dụ:

- Nếu giao thông ùn tắc, lý do là tuyến vận chuyển có thể chậm.
- Nếu tồn kho thấp hơn tồn kho an toàn, lý do là nguy cơ thiếu hàng.
- Nếu mức sử dụng phương tiện cao, gợi ý là tăng phương tiện hoặc đổi nhà vận chuyển.

Đây là phần giúp ứng dụng dễ thuyết phục hơn trong demo, vì người xem không chỉ thấy điểm số mà còn hiểu nguyên nhân.

### 4.11. API `/predict`

API này dùng cho form nhập tay.

Input là JSON.

Output ví dụ:

```json
{
  "du_doan_giao_tre": "Có nguy cơ giao trễ",
  "risk_score": 72,
  "muc_rui_ro": "Cao",
  "quyet_dinh_canh_bao": "Cảnh báo",
  "ly_do_rui_ro": ["Risk Score đang ở mức cao (72%)."],
  "goi_y_hanh_dong": ["Ưu tiên xử lý đơn này trước các đơn có rủi ro thấp hơn."],
  "model_mode": "Random Forest model"
}
```

### 4.12. API `/predict-batch`

API này dùng cho upload CSV.

Nó nhận file CSV, sau đó xử lý từng dòng.

Output gồm:

```json
{
  "total_rows": 1000,
  "success_count": 1000,
  "error_count": 0,
  "results": []
}
```

Mỗi dòng trong `results` có:

- `row_number`: dòng số mấy trong file.
- `status`: thành công hoặc lỗi.
- `input_data`: dữ liệu gốc của dòng đó.
- `risk_score`.
- `muc_rui_ro`.
- `quyet_dinh_canh_bao`.
- `ly_do_rui_ro`.
- `goi_y_hanh_dong`.

Nếu một dòng lỗi, chỉ dòng đó bị lỗi. Các dòng khác vẫn có kết quả.

## 5. Giải thích file `templates/index.html`

Đây là giao diện người dùng.

File này gồm 3 phần lớn:

1. HTML: cấu trúc trang.
2. CSS: giao diện, màu sắc, layout.
3. JavaScript: xử lý click, gọi API, render kết quả.

### 5.1. Sidebar

Giao diện có sidebar bên trái với 2 chức năng:

- Dự đoán nhập tay.
- Dự đoán bằng CSV.

JavaScript dùng hàm:

```javascript
switchView(viewName)
```

để chuyển qua lại giữa 2 màn hình.

### 5.2. Form nhập tay

Form nhập tay gồm các trường như:

- Tên mặt hàng.
- Nhóm mặt hàng.
- Mức độ ưu tiên.
- Thời gian giao.
- Địa điểm giao/kho.
- Giá trị giao dịch.
- Tồn kho.
- Nhiệt độ, độ ẩm.
- Giao thông.
- Dự báo nhu cầu.

Khi bấm nút dự đoán, hàm:

```javascript
duDoan()
```

sẽ:

1. Lấy dữ liệu từ form.
2. Kiểm tra dữ liệu số.
3. Gửi đến API `/predict`.
4. Nhận kết quả.
5. Hiển thị Risk Score, mức rủi ro, quyết định, lý do và gợi ý.

### 5.3. Chặn nhập chữ và số âm

Các hàm liên quan:

```javascript
getNonNegativeNumber(id)
handleNumericKeyDown(event)
handleNumericPaste(event)
```

Mục đích:

- Không cho nhập chữ vào ô số.
- Không cho nhập số âm.
- Nếu nhập sai, hiện lỗi ngay dưới ô nhập.

### 5.4. Upload CSV

Hàm:

```javascript
duDoanHangLoat()
```

Luồng xử lý:

1. Người dùng chọn file CSV.
2. JavaScript tạo `FormData`.
3. Gửi file đến `/predict-batch`.
4. Nhận kết quả từng dòng.
5. Render bảng kết quả.

### 5.5. Bảng kết quả CSV

Bảng hiển thị:

- Dòng.
- Trạng thái.
- Risk Score.
- Mức rủi ro.
- Quyết định.
- Nút Chi tiết.

Phần lý do và gợi ý không hiển thị trực tiếp trong bảng để tránh bảng quá dài. Thay vào đó, người dùng bấm nút `Chi tiết`.

### 5.6. Popup chi tiết

Khi bấm `Chi tiết`, hàm:

```javascript
showBatchDetail(index)
```

sẽ mở popup gồm:

- Thông số đầu vào của dòng đó.
- Kết quả dự đoán.
- Risk Score.
- Mức rủi ro.
- Quyết định.
- Lý do rủi ro.
- Gợi ý hành động.

Popup có thể đóng bằng:

- Nút Đóng.
- Click ra ngoài popup.
- Phím Esc.

### 5.7. Filter và sort CSV

Các hàm chính:

```javascript
applyBatchFilters()
updateBatchSummary(results)
```

Người dùng có thể:

- Lọc theo mức rủi ro: Tất cả, Cao, Trung bình, Thấp, Lỗi.
- Sắp xếp theo Risk Score tăng/giảm.
- Xem thống kê số dòng theo từng mức rủi ro.

Việc lọc và sort làm ở frontend, không cần gọi lại API.

### 5.8. Xuất file kết quả

Hàm:

```javascript
exportBatchResults()
```

Chức năng:

- Xuất kết quả đang hiển thị ra file CSV.
- Nếu đang lọc `Cao`, file xuất chỉ gồm các dòng rủi ro cao đang hiển thị.
- File có encoding UTF-8 BOM để Excel đọc tiếng Việt tốt hơn.

File xuất gồm:

- Dòng.
- Trạng thái.
- Risk Score.
- Mức rủi ro.
- Quyết định.
- Dự đoán giao trễ.
- Lý do rủi ro.
- Gợi ý hành động.
- Lỗi nếu có.
- Các thông số đầu vào gốc.

## 6. Giải thích model Random Forest theo cách dễ hiểu

Random Forest là một mô hình machine learning gồm nhiều cây quyết định.

Bạn có thể hiểu đơn giản:

- Một cây quyết định giống như một chuỗi câu hỏi:
  - Tồn kho có thấp không?
  - Giao thông có ùn tắc không?
  - Thời gian chờ có vượt ngưỡng không?
- Random Forest dùng nhiều cây quyết định cùng lúc.
- Sau đó lấy kết quả tổng hợp từ nhiều cây.

Ưu điểm:

- Dễ dùng cho dữ liệu bảng.
- Chịu được nhiều loại biến.
- Phù hợp với bài toán phân loại rủi ro.

Trong app này, model trả về xác suất giao trễ. Xác suất đó được đổi thành Risk Score.

## 7. Vì sao cần cả AI và business rules?

Nếu chỉ dùng AI:

- Model có thể dự đoán đúng về mặt xác suất.
- Nhưng người quản lý vẫn cần quyết định nghiệp vụ rõ ràng.

Nếu chỉ dùng rule:

- Dễ hiểu.
- Nhưng thiếu khả năng học từ dữ liệu quá khứ.

Kết hợp cả hai sẽ hợp lý hơn:

```text
AI -> tính Risk Score
Business rules -> chuyển Risk Score thành quyết định hành động
```

Ví dụ:

- Risk 72% thường là cảnh báo.
- Nhưng nếu hàng ưu tiên cao và tồn kho thấp, có thể nâng lên cảnh báo khẩn.

## 8. Các cột CSV nên có

File CSV nên có các cột giống form nhập tay:

```text
Tên mặt hàng
Nhóm mặt hàng
Mức độ ưu tiên
Hàng quan trọng
Thời gian giao
Địa điểm giao/kho
Giá trị giao dịch
Tần suất mua hàng
Số hàng còn tồn kho
Tồn kho an toàn
Nhiệt độ
Độ ẩm
Tình trạng giao thông
Thời gian chờ hàng
Ngưỡng chịu trễ giờ
Mức sử dụng phương tiện
Dự báo nhu cầu
```

Nếu thiếu một số cột, app vẫn có thể chạy trong nhiều trường hợp vì backend có giá trị mặc định. Tuy nhiên, để kết quả tốt hơn, nên cung cấp đầy đủ cột.

## 9. Cách chạy local

Cài thư viện:

```bash
pip install -r requirements.txt
```

Chạy app:

```bash
python api/index.py
```

Mở trình duyệt:

```text
http://127.0.0.1:5000
```

## 10. Deploy Vercel

File `vercel.json` cấu hình để Vercel biết app Python nằm ở:

```text
api/index.py
```

Khi push code lên GitHub branch `master`, nếu Vercel đã kết nối repo, Vercel sẽ tự deploy lại.

## 11. Những điểm có thể mở rộng thêm

Một số hướng mở rộng:

- Lưu lịch sử dự đoán vào database.
- Thêm dashboard theo tháng/kho/nhóm mặt hàng.
- Xuất Excel thay vì CSV.
- Gửi email hoặc Telegram khi có cảnh báo cao.
- Thêm đăng nhập và phân quyền.
- Tích hợp API thời tiết/giao thông thực tế.
- Train lại model bằng dữ liệu mới.

## 12. Tóm tắt cho người mới học kinh tế

Hãy hiểu ứng dụng này theo 4 lớp:

```text
Lớp 1: Dữ liệu đầu vào
Thông tin hàng hóa, tồn kho, giao thông, thời tiết, nhu cầu.

Lớp 2: AI/model
Tính xác suất giao trễ.

Lớp 3: Business rules
Biến xác suất thành quyết định quản trị.

Lớp 4: Giao diện
Hiển thị kết quả, lý do, gợi ý và hỗ trợ xử lý hàng loạt.
```

Điểm quan trọng nhất: app không chỉ dự đoán, mà còn biến dự đoán thành thông tin có thể hành động cho người quản lý.
