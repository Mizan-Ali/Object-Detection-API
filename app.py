from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import uvicorn

from Detector import *
from utils import load_model


detector = load_model()
app = FastAPI()


@app.get("/")
def homepage():
    return {"MESSAGE": "Please go to /docs to test the app"}

@app.post('/api/predict')
async def predict(uploaded_file: UploadFile = File(...)):
    file_location = f"test/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())

    imagePath = detector.predictImage(file_location, 0.5)
    
    return FileResponse(imagePath)
