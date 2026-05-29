from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from pathlib import Path

try:
    import joblib
except ImportError:
    joblib = None


BASE_DIR = Path(__file__).resolve().parent.parent


app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
CORS(app)


MODEL_PATH = BASE_DIR / "model" / "random_forest_model.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "model" / "feature_columns.pkl"

TRAIN_INPUT_COLUMNS = [
    "Vĩ độ",
    "Kinh độ",
    "Số hàng còn tồn kho",
    "Nhiệt độ",
    "Độ ẩm",
    "Tình trạng giao thông",
    "Thời gian chờ hàng",
    "Giá trị giao dịch",
    "Tần suất mua hàng",
    "Mức sử dụng phương tiện",
    "Dự báo nhu cầu",
    "Giờ giao ",
    "Tháng",
    "Thứ trong tuần",
    "Nhóm mặt hàng",
    "Mức độ ưu tiên",
    "Hàng quan trọng",
    "Ngưỡng chịu trễ giờ",
    "Tồn kho an toàn",
]

CATEGORICAL_COLUMNS = [
    "Tình trạng giao thông",
    "Nhóm mặt hàng",
    "Mức độ ưu tiên",
    "Hàng quan trọng",
]

NON_NEGATIVE_NUMERIC_FIELDS = [
    "Giá trị giao dịch",
    "Tần suất mua hàng",
    "Số hàng còn tồn kho",
    "Tồn kho an toàn",
    "Nhiệt độ",
    "Độ ẩm",
    "Thời gian chờ hàng",
    "Ngưỡng chịu trễ giờ",
    "Mức sử dụng phương tiện",
    "Dự báo nhu cầu",
]

LOCATION_COORDINATES = {
    "Kho TP.HCM": {"Vĩ độ": 10.762622, "Kinh độ": 106.660172},
    "Kho Hà Nội": {"Vĩ độ": 21.028511, "Kinh độ": 105.804817},
    "Kho Đà Nẵng": {"Vĩ độ": 16.047079, "Kinh độ": 108.206230},
    "Kho Cần Thơ": {"Vĩ độ": 10.045162, "Kinh độ": 105.746857},
}

model = None
feature_columns = None


if joblib and MODEL_PATH.exists() and FEATURE_COLUMNS_PATH.exists():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    print("Loaded Random Forest model.")
else:
    print("Model files not found. Running demo scoring mode.")


def xac_dinh_muc_rui_ro(risk_score_percent):
    if risk_score_percent >= 70:
        return "Cao"
    if risk_score_percent >= 40:
        return "Trung bình"
    return "Thấp"


def lay_so(data, field, default=0):
    value = data.get(field, default)

    if value in (None, ""):
        return default

    return float(value)


def kiem_tra_so_khong_am(data):
    for field in NON_NEGATIVE_NUMERIC_FIELDS:
        value = data.get(field)

        if value in (None, ""):
            continue

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{field} phải là số hợp lệ.")

        if numeric_value < 0:
            raise ValueError(f"{field} không được nhập số âm.")


def phan_tich_rui_ro_va_goi_y(data, risk_score_percent):
    ly_do = []
    goi_y = []

    giao_thong = data.get("Tình trạng giao thông", "")
    muc_uu_tien = data.get("Mức độ ưu tiên", "")
    hang_quan_trong = data.get("Hàng quan trọng", "")

    ton_kho = lay_so(data, "Số hàng còn tồn kho")
    ton_kho_an_toan = lay_so(data, "Tồn kho an toàn")
    thoi_gian_cho = lay_so(data, "Thời gian chờ hàng")
    nguong_chiu_tre = lay_so(data, "Ngưỡng chịu trễ giờ")
    muc_su_dung = lay_so(data, "Mức sử dụng phương tiện")
    du_bao_nhu_cau = lay_so(data, "Dự báo nhu cầu")
    nhiet_do = lay_so(data, "Nhiệt độ")
    do_am = lay_so(data, "Độ ẩm")

    if risk_score_percent >= 70:
        ly_do.append(f"Risk Score đang ở mức cao ({risk_score_percent}%).")
        goi_y.append("Ưu tiên xử lý đơn này trước các đơn có rủi ro thấp hơn.")
    elif risk_score_percent >= 40:
        ly_do.append(f"Risk Score ở mức trung bình ({risk_score_percent}%), cần theo dõi thêm.")
        goi_y.append("Theo dõi sát tiến độ giao hàng và cập nhật lại khi điều kiện thay đổi.")
    else:
        ly_do.append(f"Risk Score thấp ({risk_score_percent}%), chưa có dấu hiệu rủi ro lớn.")

    if giao_thong == "Ùn tắc":
        ly_do.append("Tình trạng giao thông đang ùn tắc, có thể làm tăng thời gian vận chuyển.")
        goi_y.append("Cân nhắc đổi tuyến đường hoặc lùi/đẩy thời điểm giao để tránh khung giờ ùn tắc.")
    elif giao_thong == "Đường vòng":
        ly_do.append("Tuyến giao hàng phải đi đường vòng, thời gian di chuyển có thể dài hơn dự kiến.")
        goi_y.append("Kiểm tra lại tuyến giao và thông báo sớm cho bên nhận nếu ETA thay đổi.")

    if nguong_chiu_tre > 0 and thoi_gian_cho > nguong_chiu_tre:
        ly_do.append("Thời gian chờ hiện tại đã vượt ngưỡng chịu trễ.")
        goi_y.append("Liên hệ nhà vận chuyển/nhà cung cấp để xác nhận nguyên nhân chờ và ETA mới.")
    elif nguong_chiu_tre > 0 and thoi_gian_cho >= nguong_chiu_tre * 0.7:
        ly_do.append("Thời gian chờ đã gần chạm ngưỡng chịu trễ.")
        goi_y.append("Chuẩn bị phương án dự phòng nếu thời gian chờ tiếp tục tăng.")

    if ton_kho < ton_kho_an_toan:
        ly_do.append("Tồn kho hiện tại thấp hơn tồn kho an toàn.")
        goi_y.append("Bổ sung tồn kho hoặc ưu tiên nhập hàng thay thế để tránh thiếu nguyên liệu.")

    if muc_su_dung >= 0.8:
        ly_do.append("Mức sử dụng phương tiện đang cao, khả năng điều phối bị hạn chế.")
        goi_y.append("Xem xét tăng phương tiện, đổi đơn vị vận chuyển hoặc chia nhỏ chuyến giao.")

    if du_bao_nhu_cau >= 100:
        ly_do.append("Dự báo nhu cầu cao, áp lực đáp ứng đơn hàng tăng.")
        goi_y.append("Tăng kế hoạch đặt hàng sớm và kiểm tra năng lực cung ứng của nhà cung cấp.")

    if nhiet_do >= 35 or do_am >= 85:
        ly_do.append("Điều kiện thời tiết/môi trường không thuận lợi cho vận chuyển.")
        goi_y.append("Kiểm tra yêu cầu bảo quản và cân nhắc phương án giao có kiểm soát điều kiện.")

    if muc_uu_tien == "Cao":
        ly_do.append("Mặt hàng có mức ưu tiên cao, tác động lớn nếu giao trễ.")
        goi_y.append("Gắn nhãn ưu tiên cao và theo dõi riêng đến khi hoàn tất giao hàng.")
    elif muc_uu_tien == "Trung bình" and risk_score_percent >= 70:
        goi_y.append("Nâng mức theo dõi vì rủi ro hiện tại cao hơn mức ưu tiên ban đầu.")

    if hang_quan_trong == "Có":
        ly_do.append("Đây là hàng quan trọng trong quy trình vận hành.")
        goi_y.append("Thông báo sớm cho bộ phận liên quan để chuẩn bị phương án thay thế nếu cần.")

    if not goi_y:
        goi_y.append("Tiếp tục theo dõi đơn hàng theo quy trình thông thường.")

    return ly_do, goi_y


def ap_dung_business_rules(data, risk_score_percent):
    muc_uu_tien = data.get("Mức độ ưu tiên", "")
    ton_kho = float(data.get("Số hàng còn tồn kho", 0))
    ton_kho_an_toan = float(data.get("Tồn kho an toàn", 0))
    thoi_gian_cho = float(data.get("Thời gian chờ hàng", 0))
    nguong_chiu_tre = float(data.get("Ngưỡng chịu trễ giờ", 0))

    ton_kho_duoi_an_toan = ton_kho < ton_kho_an_toan
    thoi_gian_cho_vuot_nguong = thoi_gian_cho > nguong_chiu_tre

    # Rule 1:
    # Nguyên liệu ưu tiên cao, risk từ 70% và tồn kho thấp => cảnh báo khẩn
    if muc_uu_tien == "Cao" and risk_score_percent >= 70 and ton_kho_duoi_an_toan:
        return "Cảnh báo khẩn"

    # Rule 2:
    # Nguyên liệu ưu tiên cao, risk rất cao => cảnh báo
    if muc_uu_tien == "Cao" and risk_score_percent >= 85:
        return "Cảnh báo"

    # Rule 3:
    # Vật tư ưu tiên trung bình, risk cao và tồn kho thấp => cảnh báo
    if muc_uu_tien == "Trung bình" and risk_score_percent >= 80 and ton_kho_duoi_an_toan:
        return "Cảnh báo"

    # Rule 4:
    # Vật tư ưu tiên trung bình, risk rất cao và thời gian chờ vượt ngưỡng => cảnh báo
    if muc_uu_tien == "Trung bình" and risk_score_percent >= 90 and thoi_gian_cho_vuot_nguong:
        return "Cảnh báo"

    # Rule 5:
    # Hàng ưu tiên thấp chỉ cảnh báo khi risk rất cao và thời gian chờ vượt ngưỡng
    if muc_uu_tien == "Thấp" and risk_score_percent >= 90 and thoi_gian_cho_vuot_nguong:
        return "Cảnh báo ưu tiên thấp"

    return "Không cảnh báo"


def chuan_hoa_du_lieu_cho_model(data):
    record = data.copy()

    dia_diem = record.get("Địa điểm giao/kho")
    if dia_diem:
        toa_do = LOCATION_COORDINATES.get(dia_diem)
        if toa_do is None:
            raise ValueError("Địa điểm giao/kho không hợp lệ.")

        record.update(toa_do)

    thoi_gian_giao = record.get("Thời gian giao")
    if thoi_gian_giao:
        dt = pd.to_datetime(thoi_gian_giao, errors="coerce")
        if pd.isna(dt):
            raise ValueError("Thời gian giao không hợp lệ.")

        record["Giờ giao "] = int(dt.hour)
        record["Tháng"] = int(dt.month)
        record["Thứ trong tuần"] = int(dt.dayofweek)

    input_df = pd.DataFrame([record])
    input_df = input_df.reindex(columns=TRAIN_INPUT_COLUMNS, fill_value=0)

    return pd.get_dummies(
        input_df,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True
    )


def du_doan_bang_model_that(data):
    """
    Hàm này dùng model Random Forest thật đã train.

    Yêu cầu:
    - Có file model/random_forest_model.pkl
    - Có file model/feature_columns.pkl

    feature_columns.pkl là danh sách cột sau khi get_dummies lúc train.
    """

    input_encoded = chuan_hoa_du_lieu_cho_model(data)

    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    risk_score = model.predict_proba(input_encoded)[0][1]

    return round(risk_score * 100, 2)


def du_doan_demo(data):
    """
    Đây là chế độ demo khi chưa có model thật.

    Công thức này không thay thế model AI.
    Nó chỉ mô phỏng luồng:
    frontend -> API -> xử lý -> Risk Score -> Business Rules.
    """

    risk = 25

    giao_thong = data.get("Tình trạng giao thông", "")
    muc_uu_tien = data.get("Mức độ ưu tiên", "")
    hang_quan_trong = data.get("Hàng quan trọng", "")

    ton_kho = float(data.get("Số hàng còn tồn kho", 0))
    ton_kho_an_toan = float(data.get("Tồn kho an toàn", 0))

    thoi_gian_cho = float(data.get("Thời gian chờ hàng", 0))
    nguong_chiu_tre = float(data.get("Ngưỡng chịu trễ giờ", 0))

    muc_su_dung = float(data.get("Mức sử dụng phương tiện", 0))
    du_bao_nhu_cau = float(data.get("Dự báo nhu cầu", 0))

    nhiet_do = float(data.get("Nhiệt độ", 0))
    do_am = float(data.get("Độ ẩm", 0))

    if giao_thong == "Ùn tắc":
        risk += 25
    elif giao_thong == "Đường vòng":
        risk += 18

    if thoi_gian_cho > nguong_chiu_tre:
        risk += 18
    elif thoi_gian_cho >= nguong_chiu_tre * 0.7:
        risk += 8

    if ton_kho < ton_kho_an_toan:
        risk += 15

    if muc_su_dung >= 0.8:
        risk += 10

    if du_bao_nhu_cau >= 100:
        risk += 6

    if nhiet_do >= 35 or do_am >= 85:
        risk += 6

    if muc_uu_tien == "Cao":
        risk += 8
    elif muc_uu_tien == "Trung bình":
        risk += 4

    if hang_quan_trong == "Có":
        risk += 5

    risk = min(max(risk, 0), 100)

    return round(risk, 2)


def du_doan_mot_dong(data):
    kiem_tra_so_khong_am(data)

    if model is not None and feature_columns is not None:
        risk_score = du_doan_bang_model_that(data)
        model_mode = "Random Forest model"
    else:
        risk_score = du_doan_demo(data)
        model_mode = "Demo scoring"

    muc_rui_ro = xac_dinh_muc_rui_ro(risk_score)
    quyet_dinh = ap_dung_business_rules(data, risk_score)
    ly_do_rui_ro, goi_y_hanh_dong = phan_tich_rui_ro_va_goi_y(data, risk_score)

    if risk_score >= 50:
        du_doan = "Có nguy cơ giao trễ"
    else:
        du_doan = "Không có nguy cơ giao trễ"

    return {
        "du_doan_giao_tre": du_doan,
        "risk_score": risk_score,
        "muc_rui_ro": muc_rui_ro,
        "quyet_dinh_canh_bao": quyet_dinh,
        "ly_do_rui_ro": ly_do_rui_ro,
        "goi_y_hanh_dong": goi_y_hanh_dong,
        "model_mode": model_mode
    }


def chuan_hoa_record_csv(record):
    cleaned = {}

    for key, value in record.items():
        if pd.isna(value):
            cleaned[key] = ""
        else:
            cleaned[key] = value

    return cleaned


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({
            "error": "Không nhận được dữ liệu JSON."
        }), 400

    try:
        return jsonify(du_doan_mot_dong(data))

    except ValueError as e:
        return jsonify({
            "error": str(e)
        }), 400
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    uploaded_file = request.files.get("file")

    if not uploaded_file or uploaded_file.filename == "":
        return jsonify({
            "error": "Chưa chọn file CSV."
        }), 400

    if not uploaded_file.filename.lower().endswith(".csv"):
        return jsonify({
            "error": "File upload phải có định dạng .csv."
        }), 400

    try:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            uploaded_file.stream.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin1")

        if df.empty:
            return jsonify({
                "error": "File CSV không có dữ liệu."
            }), 400

        results = []

        for index, record in enumerate(df.to_dict(orient="records"), start=1):
            row_data = chuan_hoa_record_csv(record)

            try:
                prediction = du_doan_mot_dong(row_data)
                results.append({
                    "row_number": index,
                    "status": "success",
                    **prediction
                })
            except Exception as e:
                results.append({
                    "row_number": index,
                    "status": "error",
                    "error": str(e)
                })

        success_count = sum(1 for item in results if item["status"] == "success")

        return jsonify({
            "total_rows": len(results),
            "success_count": success_count,
            "error_count": len(results) - success_count,
            "results": results
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
