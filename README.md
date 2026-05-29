# Prototype AI cảnh báo rủi ro giao hàng trễ

Prototype mô phỏng hệ thống AI cảnh báo rủi ro giao hàng trễ trong quy trình thu mua nguyên liệu.

## 1. Cấu trúc

- `api/index.py`: API Flask nhận dữ liệu, gọi model, tạo Risk Score và áp dụng Business Rules.
- `templates/index.html`: giao diện nhập thông tin đơn hàng/vận chuyển.
- `model/random_forest_model.pkl`: model Random Forest đã train.
- `model/feature_columns.pkl`: danh sách cột feature sau khi encode.
- `requirements.txt`: danh sách thư viện cần cài.
- `vercel.json`: cấu hình deploy Flask API lên Vercel.

## 2. Chạy local

```bash
pip install -r requirements.txt
python api/index.py
```

Mở trình duyệt tại:

```text
http://127.0.0.1:5000
```

## 3. Deploy Vercel

Import project lên Vercel, Vercel sẽ đọc `vercel.json` và build Python function từ `api/index.py`.
