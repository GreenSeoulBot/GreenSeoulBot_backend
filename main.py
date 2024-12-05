# from PIL import Image
# from image.predict import read_image
# from image.predict import predict_image
# from fastapi import FastAPI, File, UploadFile, Form
# from io import BytesIO
from fastapi import FastAPI, Depends, Path, HTTPException
# from pydantic import BaseModel
from db.database import engineconn
from db.models import Policy, Admin, LargeWaste
# from sqlalchemy import distinct
# import os
# import pandas as pd
# import numpy as np 
# import torch
# from matplotlib import pyplot as plt
# import cv2
# from typing import Optional
# from ultralytics import YOLO
# from fastapi.responses import StreamingResponse  # 추가된 코드
# import openai
# import os
from fastapi import FastAPI, APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.openapi.models import APIKey
from pydantic import BaseModel
import openai
import os
# from config import logger
import logging
from generateMessage import load_rewardPolicy, laod_fee, create_vectorstore, rewardChain, largewastChain
import json
import torch
import base64
import io
import cv2
import numpy as np
from PIL import Image

OPENAI_API_KEY = os.getenv("API-KEY")
openai.api_key = OPENAI_API_KEY
logger = logging.getLogger("uvicorn.error")

app = FastAPI()

engine = engineconn()
session = engine.sessionmaker()

seoul_districts = [
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구", "도봉구",
    "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구",
    "관악구", "서초구", "강남구", "송파구", "강동구"
]

# Initialize the models
model = torch.hub.load("./models/yolov5", 'yolov5s', source='local')

def get_db():
    db = engine.sessionmaker()
    try:
        yield db
    finally:
        db.close()

class Item(BaseModel):
    message : str
class districtName(BaseModel):
    district_name : str



# 버튼 채팅 메시지 생성
@app.post("/chatbot/policy")
async def chat_btn(item: districtName, db: session = Depends(get_db)):
    district_name = item.district_name
    policy = db.query(Policy).filter(Policy.district_name == district_name).first()

    if policy:
        return policy
    else:
        return {
            "status": "failed",
            "error": "정책 정보를 조회할 수 없습니다."
        }
    


# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

router = APIRouter()

# Models
class ChatModel(BaseModel):
    user_input: str

class PolicyModel(BaseModel):
    district_name: str

class UploadModel(BaseModel):
    district_name: str
    image_file: str = None


def load_district_websites():
    with open('district_websites.json', 'r', encoding='utf-8') as file:
        return json.load(file)

district_websites = load_district_websites()

# 필요 시 사용 
def save_image(file: UploadFile):
    with open(f'./uploads/{file.filename}', 'wb') as f:
        f.write(file.file.read())

def get_district_url(district_name: str):
    try:
        with open('districts.json', 'r', encoding='utf-8') as f:
            district_data = json.load(f)
        for district in district_data['districts']:
            if district['title'] == district_name:
                return district['district_url']
    except FileNotFoundError:
        logger.error("District JSON file not found.")
    return None

async def get_response(user_input: str):
    try:
        documents = load_rewardPolicy()
        vectorstore = create_vectorstore(documents)
        qa_chain = rewardChain(vectorstore)
        answer = qa_chain.invoke({"input": user_input})
        return answer
    except Exception as e:
        logger.error(f"Error fetching response from OpenAI: {e}")
        return {"error": "죄송합니다. 현재 서비스를 제공할 수 없습니다. 나중에 다시 시도해 주세요."}

def fee_info(user_input: str):
    try:
        documents = load_largewastePolicy()
        vectorstore = create_vectorstore(documents)
        qa_chain = largewastChain(vectorstore)
        answer = qa_chain.invoke({"input": user_input})
        return answer.get('answer', '')
    except Exception as e:
        logger.error(f"Error fetching response from OpenAI: {e}")
        return "죄송합니다. 현재 서비스를 제공할 수 없습니다. 나중에 다시 시도해 주세요."



@router.post('/chat', response_model=dict)
async def chat_endpoint(chat: ChatModel):
    try:
        if not chat.user_input:
            raise HTTPException(status_code=400, detail="입력해주세요.")
        bot_response = await get_response(chat.user_input)
        answer = bot_response.get('answer', '')
        return {"message": answer}
    except Exception as e:
        logger.error(f"Error processing user input: {e}")
        raise HTTPException(status_code=500, detail="응답을 생성하는 중 오류가 발생했습니다.")

# @app.post('/policy', response_model=dict)
# async def policy_endpoint(policy: PolicyModel, item: districtName, db: session = Depends(get_db)):
#     try:
#         if not policy.district_name:
#             raise HTTPException(status_code=400, detail="지역구 이름을 입력해주세요.")

#         if not any(item["district_name"] == policy.district_name for item in district_websites):
#             raise HTTPException(status_code=400, detail="해당 지역구의 정보를 없습니다.")

#         bot_response = await get_response(policy.district_name)
#         message = bot_response.get('answer', '')
#         homepage_url = next(item["district_url"] for item in district_websites if item["district_name"] == policy.district_name)
#         return {"message": message, "district_url": homepage_url}
#     except Exception as e:
#         logger.error(f"Error processing policy request: {e}")
#         raise HTTPException(status_code=500, detail="정책 정보를 조회하는 중 오류가 발생했습니다.")


# 버튼 채팅 메시지 생성
@router.post("/policy")
async def chat_btn(item: districtName, db: session = Depends(get_db)):
    district_name = item.district_name
    policy = db.query(Policy).filter(Policy.district_name == district_name).first()
    message = policy.contents
    homepage_url = next(item["district_url"] for item in district_websites if item["district_name"] == policy.district_name)
    print(homepage_url)
    if policy:
        return {"message": message, "district_url": homepage_url}
    else:
        return {
            "status": "failed",
            "error": "정책 정보를 조회할 수 없습니다."
        }



@router.post('/upload', response_model=dict)
async def upload_photo(
    district_name: str = Form(...),
    image_file: str = Form(...)
):
    try:
        if not district_name:
            raise HTTPException(status_code=400, detail="district_name이 제공되지 않았습니다.")
        district_url = get_district_url(district_name)
        if not district_url:
            raise HTTPException(status_code=400, detail=f"'{district_name}'에 해당하는 구를 찾을 수 없습니다.")

        base64_str = image_file.split(",")[1]
        base64_str += '=' * (4 - len(base64_str) % 4)
        byte_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(byte_data))
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # model = torch.hub.load("./yolov5", 'custom', path='./best.pt', source='local')
        temp = model(img)
        df = temp.pandas().xyxy[0]
        recognized_result = df.name[0]
        user_input = f"{district_name}의 {recognized_result} 폐기 방법"
        answer = largeWast_info(user_input)

        return {"district_name": district_name, "message": answer, "district_url": district_url}
    except Exception as e:
        logger.error(f"Error processing image upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="이미지 처리 중 오류가 발생했습니다.")

# Register router
app.include_router(router)
