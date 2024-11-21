from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
from io import BytesIO

model = ResNet50(weights='imagenet')

def read_image(file) -> Image.Image:
    pil_image = Image.open(BytesIO(file))
    print('Image loaded successfully')
    return pil_image

def predict_image(file: Image.Image):
    img = file.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  
    x = preprocess_input(x)

    # Make predictions
    preds = model.predict(x)

    result = decode_predictions(preds, top=1)[0]
    
    response = []
    for res in result:
        response.append({
            "class": res[1],
            "confidence": f"{res[2]*100:0.2f} %"
        })

    return response
