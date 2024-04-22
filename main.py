from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from imps import predictions

app = FastAPI()

# Allow requests from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.get("/predict")
def root(dataset: str, model:str):
    response = predictions.predict_start(dataset, model)
    return response
