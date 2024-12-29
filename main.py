from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import Optional
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
import warnings 
import pickle

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')  

model_path = "/home/Barath/S7Project/models/model.h5"

def load_model():
    model = tf.keras.models.load_model(model_path)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = load_model()
app = FastAPI()

@app.get("/health")
def health_check():
    return JSONResponse(content={"message":"Still Alive"})
    
@app.post("/predict/")
async def predict(file: UploadFile = File(...),patient_id: int = Form(...), sex: int = Form(...),
    age_approx: int = Form(...),anatom_site_general_challenge: int = Form(...)):
    image_bytes = await file.read()
    try:
        image = Image.open(BytesIO(image_bytes))
        image = image.convert("RGB") 
        image = image.resize((256, 256)) 
        image_np = np.array(image) / 255.0 
        width, height = image.size 
        image_tensor = np.expand_dims(image_np, axis=0) 
        patient_id_tensor = tf.convert_to_tensor([patient_id])  
        sex_tensor = tf.convert_to_tensor([sex])  
        age_approx_tensor = tf.convert_to_tensor([age_approx])  
        anatom_site_tensor = tf.convert_to_tensor([anatom_site_general_challenge])  
        width_tensor = tf.convert_to_tensor([width]) 
        height_tensor = tf.convert_to_tensor([height])  

        patient_id_tensor = tf.reshape(patient_id_tensor, (1, 1)) 
        sex_tensor = tf.reshape(sex_tensor, (1, 1))  
        age_approx_tensor = tf.reshape(age_approx_tensor, (1, 1))  
        anatom_site_tensor = tf.reshape(anatom_site_tensor, (1, 1)) 
        width_tensor = tf.reshape(width_tensor, (1, 1))
        height_tensor = tf.reshape(height_tensor, (1, 1))  

        feature_vector = tf.concat([patient_id_tensor, sex_tensor, age_approx_tensor, anatom_site_tensor, width_tensor, height_tensor], axis=1)

        if feature_vector.shape[1] < 8:
            feature_vector = tf.concat([feature_vector, tf.zeros([1, 8 - feature_vector.shape[1]], dtype=tf.int32)], axis=1)

        feature_vector = tf.reshape(feature_vector, (1, 8))
        try:
            prediction = model.predict([image_tensor,feature_vector])

            output = prediction[0] 
            result = output.tolist()[0]
            return {"prediction": int(result), "patient_id": patient_id}

        except Exception as e:
            return JSONResponse(content={"error": "Error during prediction", "details": str(e)}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": "Error processing image", "details": str(e)}, status_code=400)
