from PIL import Image
from image.predict import read_image
from image.predict import predict_image
from fastapi import FastAPI, File, UploadFile, Form
from io import BytesIO
from fastapi import FastAPI, Depends, Path, HTTPException
from pydantic import BaseModel
from db.database import engineconn
from db.models import Policy, Admin, LargeWaste
from sqlalchemy import distinct
import os
import pandas as pd
import numpy as np 
import torch
from matplotlib import pyplot as plt
import cv2
from typing import Optional
from ultralytics import YOLO
from fastapi.responses import StreamingResponse  # 추가된 코드
import openai
import os

# from ultralytics.yolo.utils.plotting import Annotator, colors

# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import VectorParams, PointStruct
OPENAI_API_KEY = os.getenv("API-KEY")
openai.api_key = OPENAI_API_KEY

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

@app.get("/chatbot")
async def root():
    example = session.query(Test).all()
    return example


# @app.post("/green-seoul-bot/image")
# async def create_upload_file(file: bytes = File(...)):

#     image = read_image(file)
#     prediction = predict_image(image)
#     confidence = [float(item["confidence"].replace("%","").strip()) for item in prediction]

#     print(type(confidence))
#     if confidence[0]<70:
#         return prediction
    
#     class_type = [item['class'] for item in prediction][0]
    
#     return class_type

def results_to_json(results, model):
    return [
        [
          {
          "class": int(pred[5]),
          "class_name": model.model.names[int(pred[5])],
          "bbox": [int(x) for x in pred[:4].tolist()], # convert bbox results to int from float
          "confidence": float(pred[4]),
          }
        for pred in result
        ]
      for result in results.xyxy
      ]

@app.post("/chatbot/upload")
async def create_upload_file(file: UploadFile = File(...)):
    # Process the uploaded image for object detection
    img = await file.read()
    # PIL.Image로 변환
    img = Image.open(BytesIO(img)) # .resize((640, 640), Image.Resampling.LANCZOS)
    
    # numpy 배열로 변환
    img_array = np.array(img)
    
    # YOLOv5 모델에 입력
    results = model(img_array)

    # bbox 확인
    detections = results.pandas().xyxy[0].to_dict(orient="records")
    # for detection in detections:
    #     xmin = int(detection["xmin"])
    #     ymin = int(detection["ymin"])
    #     xmax = int(detection["xmax"])
    #     ymax = int(detection["ymax"])
    
    # cv2.rectangle(img_array, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    # cv2.putText(img_array, detection["name"], (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # # 수정된 이미지를 반환하기 위해 PIL로 변환
    # result_image = Image.fromarray(img_array)
    
    # # 이미지 바이너리로 변환
    # img_byte_arr = BytesIO()
    # result_image.save(img_byte_arr, format='PNG')
    # img_byte_arr.seek(0)

    # return StreamingResponse(img_byte_arr, media_type="image/png")
    return detections
    
    # 결과를 JSON으로 반환
    # return results.pandas().xyxy[0].to_dict(orient="records")
    
# 버튼 채팅 메시지 생성
@app.get("/chatbot/policy{district_name}")
async def chat_btn(district_name: str, db: session = Depends(get_db)):
    policy = db.query(Policy).filter(Policy.district_name == district_name).first()

    if policy:
        return policy
    else:
        return {
            "status": "failed",
            "error": "정책 정보를 조회할 수 없습니다."
        }
    
async def generate_answer(query):
    model = "gpt-3.5-turbo"
    messages = [{
        "role" : "system",
        "content" : "You are a chatbot called 'green-seoul-bot'. Please use a friendly tone. You provide information about recycling policies for each district in Seoul and the fees for large waste disposal. Answer users' questions as much as possible according to your role, but politely decline if a question arises that you do not know the answer to.",
        }, {
            "role" : "user",
            "content" : query
        }]
    response = openai.ChatCompletion.create(model=model, messages=messages)
    answer = response['choices'][0]['message']['content']
    return answer

@app.post("/chatbot/chat")
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
    else: answer = await generate_answer(chat)

    return answer

