
from keras_malicious_url_detector.library.bidirectional_lstm import BidirectionalLstmEmbedPredictor
from keras_malicious_url_detector.library.utility.url_data_loader import load_url_data
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware

class URLData(BaseModel): 
    url: HttpUrl


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, etc.)
    allow_headers=["*"],  # Allows all headers
)

model_dir_path = './models'
predictor = BidirectionalLstmEmbedPredictor()
predictor.load_model(model_dir_path)
   
@app.post('/predict')
def predict_phishing(url_data: URLData):
    url = str(url_data.url)
    predict, predicted = predictor.predict(url)

    return {
        "url": url,
        "is_phishing": True if predict else False,
        "phishing_prediction": round(predicted[-1] * 100,2),
        "non_phishing_prediction": round(predicted[0] * 100,2)
    }
