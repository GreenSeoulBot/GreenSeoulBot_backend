from PIL import Image
from image.predict import read_image
from image.predict import predict_image
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from fastapi import FastAPI, Depends, Path, HTTPException
from pydantic import BaseModel
from db.database import engineconn
from db.models import Test


app = FastAPI()

engine = engineconn()
session = engine.sessionmaker()

class Item(BaseModel):
    district_name : str
    contents : str

@app.get("/green-seoul-bot")
async def root():
    example = session.query(Test).all()
    return example


@app.post("/green-seoul-bot/image")
async def create_upload_file(file: bytes = File(...)):

    image = read_image(file)
    prediction = predict_image(image)

    return prediction

# 데이터베이스 연결 테스트용 엔드포인트
@app.get("/db-test")
async def db_test():
    try:
        example = session.query(Test).first()
        return {"status": "success", "result": example}
    except Exception as e:
        return {"status": "failed", "error": str(e)}