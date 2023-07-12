import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pydantic_numpy.dtype as pnd
from pydantic_numpy import NDArray, NDArrayFp32
from tensorflow.keras.models import load_model
import logging
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request, status, Response
from fastapi.encoders import jsonable_encoder

app = FastAPI(
    title="🫶 Main'Terprète API",
    description="description",
    version="0.1",
    contact={
        "name": "Main'Terprète",
        "url": "https://mainterprete.herokuapp.com",
    }
)

@app.get("/")
async def root():
    return {"Hello World!"}

# load model 
model = load_model("model_20230410_50epochs.h5")

# request structure
class TranslationRequest(BaseModel):
    data : NDArray


# post endpoint
@app.post("/translate")
async def translate(request:TranslationRequest):

    #request's numpy array
    list_data = request.data
    npy_array = np.array(list_data).reshape(1, 50, 1659)
    # deep learning model query
    prediction = np.argmax(model.predict(npy_array), axis=-1)
    word_list = ['AUSSI','AVOIR','LANGUE DES SIGNES','MAIS','NON','OUI','PLUS','QUOI','REGARDER','SOURD']
    # finding the result and preparing it for a return
    result = word_list[int(prediction)].capitalize()
    
    json_compatible_item_data = jsonable_encoder(result)

    # return result
    return JSONResponse(content=json_compatible_item_data)

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port='4000')