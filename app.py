from flask import Flask, request, jsonify
import face_recognition
import io
from PIL import Image

app = Flask(__name__)

# Hàm giảm kích thước ảnh
def resize_image(image, max_size=800):
    width, height = image.size
    if width > max_size or height > max_size:
        ratio = max_size / float(max(width, height))
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

@app.route('/compare-faces', methods=['POST'])
def compare_faces():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Thiếu ảnh'}), 400

    img1_file = request.files['image1']
    img2_file = request.files['image2']

    try:
        # Đọc ảnh từ bộ nhớ đệm
        img1 = Image.open(io.BytesIO(img1_file.read()))
        img2 = Image.open(io.BytesIO(img2_file.read()))

        # Giảm kích thước ảnh trước khi xử lý
        img1 = resize_image(img1)
        img2 = resize_image(img2)

        # Chuyển ảnh thành RGB để tăng tốc độ nhận diện
        img1 = img1.convert('RGB')
        img2 = img2.convert('RGB')

        # Lưu ảnh thành mảng byte để face_recognition nhận diện
        img1_bytes = io.BytesIO()
        img2_bytes = io.BytesIO()
        img1.save(img1_bytes, format='JPEG')
        img2.save(img2_bytes, format='JPEG')

        # Đưa mảng byte vào face_recognition
        img1_bytes.seek(0)
        img2_bytes.seek(0)
        img1_data = face_recognition.load_image_file(img1_bytes)
        img2_data = face_recognition.load_image_file(img2_bytes)

        # Lấy encoding của khuôn mặt
        encodings1 = face_recognition.face_encodings(img1_data)
        encodings2 = face_recognition.face_encodings(img2_data)

        if len(encodings1) == 0:
            return jsonify({'error': 'Không tìm thấy khuôn mặt trong ảnh người dùng (image1)'}), 400

        if len(encodings2) == 0:
            return jsonify({'error': 'Không tìm thấy khuôn mặt trong ảnh server (image2)'}), 400


        # Chỉ sử dụng khuôn mặt đầu tiên trong danh sách
        encoding1 = encodings1[0]
        encoding2 = encodings2[0]

        result = bool(face_recognition.compare_faces([encoding1], encoding2)[0])
        return jsonify({'matched': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)