from fastapi import FastAPI, UploadFile, HTTPException
from keras.models import load_model
import cv2
import numpy as np
from starlette.responses import JSONResponse
from starlette.requests import Request

app = FastAPI()

# Load the model on startup
model_path = "animal-classification"
loaded_model = load_model(model_path)

@app.post("/predict/")
async def predict(request: Request):
    form = await request.form()
    uploaded_file = form["file"]
    contents = await uploaded_file.read()

    # Convert bytes to numpy array
    nparr = np.frombuffer(contents, np.uint8)

    # Decode the image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resize the image
    resized_image = cv2.resize(image, (64, 64))

    # Add an extra dimension for batch size and predict
    preprocessed_image = np.expand_dims(resized_image, axis=0)
    predictions = loaded_model.predict(preprocessed_image)

    # Get the predicted class
    predicted_class = np.argmax(predictions[0])

    return JSONResponse(content={"predicted_class": int(predicted_class)})

