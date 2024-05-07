from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


app = FastAPI()
MODEL = tf.keras.models.load_model('C:/Users/Mokksh/PycharmProjects/Potato_Project/models/Potato.keras')
CLASSES = ['Early_Blight','Late_Blight','Healthy']

@app.get("/ping")
async def ping():
    return "Hello , Sir I am alive!"


def read_file_as_img(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)  # variable : this coln is for type defining data type
):
    image = read_file_as_img(await file.read())
    img_batch = np.expand_dims(image, axis=0)
    prediction = MODEL.predict(img_batch)
    print(prediction)
    predicted_class = CLASSES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])

    return{
        "predicted_class": predicted_class,
        "confidence": float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
