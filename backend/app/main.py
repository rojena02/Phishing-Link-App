
from keras_malicious_url_detector.library.bidirectional_lstm import BidirectionalLstmEmbedPredictor
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware

class URLData(BaseModel): 
    url: HttpUrl

app = FastAPI(title="Phishing Link Detection Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
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
