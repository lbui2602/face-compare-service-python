from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import face_recognition
import io
from PIL import Image
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient

app = FastAPI()
client = MongoClient("mongodb://localhost:27017/")
db = client["face"]
collection = db["encodings"]

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

@app.post("/detect-face")
async def detect_face(
    image: UploadFile = File(...),
    label: str = Form(...)
):
    try:
        existing = collection.find_one({"label": label})
        if existing:
            return JSONResponse(
                status_code=400,
                content={"message": f"Label '{label}' đã tồn tại, không thể lưu trùng"}
            )
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert('RGB')

        pil_image = resize_image(pil_image)

        # Chuyển sang bytes để xử lý
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format='JPEG', quality=95)
        img_bytes.seek(0)

        # Nhận diện và encoding
        img_data = face_recognition.load_image_file(img_bytes)
        face_locations = face_recognition.face_locations(img_data)
        has_face = len(face_locations) == 1

        if len(face_locations) == 1 :
            encoding = face_recognition.face_encodings(img_data, known_face_locations=face_locations)[0]
            encoding_list = encoding.tolist()
            collection.insert_one({
                "label": label,
                "encoding": encoding_list
            })

        return {"face_detected": has_face, "length": len(face_locations)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Hàm so sánh khuôn mặt
def compare_encoding_with_label(img_bytes: bytes, encoding_by_label: list):
    try:
        pil_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        pil_image = resize_image(pil_image)

        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        buffer.seek(0)

        img_data = face_recognition.load_image_file(buffer)

        face_locations = face_recognition.face_locations(img_data)
        if len(face_locations) == 0:
            # Không tìm thấy khuôn mặt
            # raise HTTPException(status_code=400, detail="Không tìm thấy khuôn mặt trong ảnh người dùng")
            return {"hi": "Không tìm thấy khuôn mặt."}

        encodings = face_recognition.face_encodings(img_data, known_face_locations=face_locations)

        for encoding in encodings:
            distance = face_recognition.face_distance([encoding_by_label], encoding)[0]
            if distance < 0.4:
                return {"matched": True}

        return {"matched": False}

    except Exception as e:
        return {"error": str(e), "matched": False}


@app.post("/compare-faces")
async def compare_face(
    image: UploadFile = File(...),
    label: str = Form(...)
):
    try:
        # Lấy encoding từ DB theo label
        data = collection.find_one({"label": label})
        if not data:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy label '{label}' trong database")

        encoding_by_label = data["encoding"]

        # Đọc file ảnh
        img_bytes = await image.read()

        # Chạy sync function trong thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, compare_encoding_with_label, img_bytes, encoding_by_label)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Chạy ứng dụng FastAPI
    uvicorn.run(app, host="0.0.0.0", port=5000)
