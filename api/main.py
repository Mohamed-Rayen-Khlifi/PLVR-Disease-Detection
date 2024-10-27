from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


model = tf.keras.models.load_model('../saved_models/1.keras')
class_names = ['Early Blight', 'Healthy', 'Late Blight']


app = FastAPI()

origins = [
    "http://localhost",  
    "http://localhost:3000"  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def read_file_as_image(data) -> np.array:
    img = np.array(Image.open(BytesIO(data)))
    return img



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)
    predictions = model.predict(image_batch)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        "class": predicted_class,
        "confidence": float(confidence),
    }



if __name__ == "__main__":
    uvicorn.run(app, host= 'localhost', port = 8000)
