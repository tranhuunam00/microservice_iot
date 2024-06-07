from flask import Flask, request, jsonify
import joblib

# Load mô hình đã lưu
R_model = joblib.load('./python_fire/r_model.pkl')

# Khởi tạo Flask app
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ request
    data = request.get_json(force=True)
    X_test = data['X_test']
    print(X_test)
    # Dự đoán
    y_pred_R_model = R_model.predict(X_test)

    # Trả về kết quả dưới dạng JSON
    return jsonify(y_pred_R_model.tolist())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
