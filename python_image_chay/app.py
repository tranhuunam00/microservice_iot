from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
import cv2

# Tải mô hình đã lưu
model = joblib.load('./python_image_chay/model_image_chay.pkl')

# Khởi tạo Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy ảnh từ request
    file = request.files['image']
    
    # Đọc ảnh
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    
    # Resize ảnh
    img = cv2.resize(img, (32, 32))
    
    # Normalize ảnh
    img = img / 255.0
    
    # Flatten ảnh
    img = img.flatten()
    
    # Chuyển đổi thành mảng numpy
    img = np.array([img])
    
    # Dự đoán
    prediction = model.predict(img)
    
    # Trả về kết quả dưới dạng JSON
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)