from fastapi import FastAPI, File, Response, UploadFile
from fastapi.responses import Response, FileResponse
import numpy as np
import os
import cv2

from model import upscale_image
app = FastAPI()


@app.get('/')
def read_root():
    return {'Hello': 'World'}


@app.post('/upscale')
def upscale(suffix: str = 'upscaled', scale: int = 4, face_enhance: bool = False, image: UploadFile = File(...)):

    image_content = image.file.read()

    image_np = np.frombuffer(image_content, np.uint8)

    image_cv2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    output_content = upscale_image(image_cv2, scale, face_enhance)

    output_filename = f'{os.path.splitext(os.path.basename(image.filename))[0]}{suffix}.png'

    _, img_encoded = cv2.imencode(".png", output_content)

    return Response(content=img_encoded.tobytes(), media_type="image/png", headers={
        "Content-Disposition": f"attachment; filename={output_filename}"
    })
