from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import face_recognition
import io
from PIL import Image
import asyncio
from concurrent.futures import ThreadPoolExecutor

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

# Tạo một ThreadPoolExecutor cho phép xử lý song song
executor = ThreadPoolExecutor()

# Hàm so sánh khuôn mặt
def compare_faces_sync(img1_bytes, img2_bytes):
    try:
        img1 = Image.open(io.BytesIO(img1_bytes))
        img2 = Image.open(io.BytesIO(img2_bytes))

        img1 = resize_image(img1)
        img2 = resize_image(img2)

        img1 = img1.convert('RGB')
        img2 = img2.convert('RGB')

        img1_bytes = io.BytesIO()
        img2_bytes = io.BytesIO()
        img1.save(img1_bytes, format='JPEG')
        img2.save(img2_bytes, format='JPEG')

        img1_bytes.seek(0)
        img2_bytes.seek(0)
        img1_data = face_recognition.load_image_file(img1_bytes)
        img2_data = face_recognition.load_image_file(img2_bytes)

        encodings1 = face_recognition.face_encodings(img1_data)
        encodings2 = face_recognition.face_encodings(img2_data)

        if len(encodings1) == 0:
            raise HTTPException(status_code=400, detail="Không tìm thấy khuôn mặt trong ảnh người dùng")
        if len(encodings2) == 0:
            raise HTTPException(status_code=400, detail="Không tìm thấy khuôn mặt trong ảnh server")

        encoding1 = encodings1[0]
        encoding2 = encodings2[0]

        distance = face_recognition.face_distance([encoding1], encoding2)[0]
        return distance < 0.4  # Nếu khoảng cách nhỏ hơn 0.4 thì coi là trùng khớp

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare-faces")
async def compare_faces(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    try:
        # Đọc ảnh từ client
        img1_bytes = await image1.read()
        img2_bytes = await image2.read()

        # Xử lý so sánh khuôn mặt trong ThreadPoolExecutor (song song)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, compare_faces_sync, img1_bytes, img2_bytes)

        return JSONResponse(content={'matched': bool(result)})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Chạy ứng dụng FastAPI
    uvicorn.run(app, host="0.0.0.0", port=5000)
