from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
from io import BytesIO

# Pre-trained ResNet50 model with ImageNet weights
model = ResNet50(weights='imagenet')

def read_image(file) -> Image.Image:
    # Read the image from bytes and convert to PIL image
    pil_image = Image.open(BytesIO(file))
    print('Image loaded successfully')
    return pil_image

def transformacao(file: Image.Image):
    # Resize and preprocess the image
    img = file.resize((224, 224))
    # img = np.asarray(img)[..., :3]  # Convert to RGB array if needed
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    x = preprocess_input(x)

    # Make predictions
    preds = model.predict(x)

    # Decode predictions to readable format
    result = decode_predictions(preds, top=3)[0]
    
    # Format the response
    response = []
    for res in result:
        response.append({
            "class": res[1],
            "confidence": f"{res[2]*100:0.2f} %"
        })

    return response
