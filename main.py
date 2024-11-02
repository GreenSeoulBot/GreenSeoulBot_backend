from PIL import Image
from image.predict import read_image
from image.predict import transformacao
from fastapi import FastAPI, File, UploadFile
from io import BytesIO


app = FastAPI()


@app.get("/green-seoul-bot")
async def root():
    return {"message": "Hello World"} 


@app.post("/green-seoul-bot/image")
async def create_upload_file(file: bytes = File(...)):

    # read image
    imagem = read_image(file)
    # transform and prediction 
    prediction = transformacao(imagem)

    return prediction