from PIL import Image
from image.predict import read_image
from image.predict import predict_image
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from fastapi import FastAPI, Depends, Path, HTTPException
from pydantic import BaseModel
from db.database import engineconn
from db.models import Policy, Admin, LargeWaste
from sqlalchemy import distinct



app = FastAPI()

engine = engineconn()
session = engine.sessionmaker()

seoul_districts = [
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구", "도봉구",
    "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구",
    "관악구", "서초구", "강남구", "송파구", "강동구"
]

def get_db():
    db = engine.sessionmaker()
    try:
        yield db
    finally:
        db.close()

class Item(BaseModel):
    message : str

@app.get("/green-seoul-bot")
async def root():
    example = session.query(Test).all()
    return example


@app.post("/green-seoul-bot/image")
async def create_upload_file(file: bytes = File(...)):

    image = read_image(file)
    prediction = predict_image(image)
    confidence = [float(item["confidence"].replace("%","").strip()) for item in prediction]

    print(type(confidence))
    if confidence[0]<70:
        return "죄송합니다. 다른 사진을 첨부해주세요."
    
    class_type = [item['class'] for item in prediction][0]
    return class_type

# 버튼 채팅 메시지 생성
@app.get("/green-seoul-bot/chatbot/btn/{district_name}")
async def chat_btn(district_name: str, db: session = Depends(get_db)):
    policy = db.query(Policy).filter(Policy.district_name == district_name).first()

    if policy:
        return policy
    else:
        return {
            "status": "failed",
            "error": "정책 정보를 조회할 수 없습니다."
        }

# def get_large_waste(db: session = Depends(get_db)):
#     try:
#         largeWaste = db.query(distinct(LargeWaste.large_waste)).all()
#         result = [row[0] for row in largeWaste]
#         return result
#     finally:
#         session.close()

@app.post("/green-seoul-bot/chatbot/chat")
async def create_item(item: Item, db: session = Depends(get_db)):
    chat = item.message
    word_arr = chat.split()
    # waste_list = get_large_waste()

    try:
        largeWaste = db.query(distinct(LargeWaste.large_waste)).all()
        waste_list = [row[0] for row in largeWaste]
    finally:
        session.close()

    for word in word_arr:
        if word in waste_list: waste_word = word
        elif word in seoul_districts: district_word = word

    if 'waste_word' in locals() and 'district_word' in locals():
        answer = db.query(LargeWaste).filter(
            LargeWaste.district_name == district_word,
            LargeWaste.large_waste == waste_word
        ).first()
    elif 'district_word' in locals():
        answer = db.query(Policy).filter(Policy.district_name == district_word).first()
    else: answer = "필요한 정보가 부족합니다."

    return answer