from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import face_recognition
import io
from PIL import Image
import asyncio

app = FastAPI()

# Hàm giảm kích thước ảnh
def resize_image(image, max_size=800):
    width, height = image.size
    if width > max_size or height > max_size:
        ratio = max_size / float(max(width, height))
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

@app.post("/detect-face")
async def detect_face(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()

        # Đọc ảnh từ bộ nhớ đệm
        image = Image.open(io.BytesIO(img_bytes))

        # Giảm kích thước và chuẩn hóa ảnh
        image = image.convert('RGB')

        # Lưu ảnh thành mảng byte để xử lý
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG', quality=95)  # Chất lượng cao hơn, ít nén hơn
        img_bytes.seek(0)

        # Nhận diện khuôn mặt
        img_data = face_recognition.load_image_file(img_bytes)
        face_locations = face_recognition.face_locations(img_data)

        # Kiểm tra số lượng khuôn mặt tìm thấy
        has_face = len(face_locations) > 0

        return JSONResponse(content={'face_detected': has_face})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare-faces")
async def compare_faces(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    try:
        img1_bytes = await image1.read()
        img2_bytes = await image2.read()

        # Đọc ảnh từ bộ nhớ đệm
        img1 = Image.open(io.BytesIO(img1_bytes))
        img2 = Image.open(io.BytesIO(img2_bytes))

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

        # Kiểm tra nếu không tìm thấy khuôn mặt trong ảnh
        if len(encodings1) == 0:
            raise HTTPException(status_code=400, detail="Không tìm thấy khuôn mặt trong ảnh người dùng")

        if len(encodings2) == 0:
            raise HTTPException(status_code=400, detail="Không tìm thấy khuôn mặt trong ảnh server")

        # Chỉ sử dụng khuôn mặt đầu tiên trong danh sách
        encoding1 = encodings1[0]
        encoding2 = encodings2[0]

        # Tính khoảng cách giữa hai khuôn mặt
        distance = face_recognition.face_distance([encoding1], encoding2)[0]
        result = distance < 0.4  # Nếu khoảng cách nhỏ hơn 0.4 thì coi là trùng khớp

        return JSONResponse(content={'matched': bool(result)})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Chạy ứng dụng FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8000)
