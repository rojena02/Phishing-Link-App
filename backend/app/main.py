
from keras_malicious_url_detector.library.bidirectional_lstm import BidirectionalLstmEmbedPredictor
from keras_malicious_url_detector.library.cnn_lstm import CnnLstmPredictor
from keras_malicious_url_detector.library.lstm import LstmPredictor
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
bidirectional_lstm_predictor = BidirectionalLstmEmbedPredictor()
cnn_lstm_predictor = CnnLstmPredictor()
lstm_predictor = LstmPredictor()
bidirectional_lstm_predictor.load_model(model_dir_path)
cnn_lstm_predictor.load_model(model_dir_path)
lstm_predictor.load_model(model_dir_path)
   
@app.post('/predict')
def predict_phishing(url_data: URLData):
    url = str(url_data.url)
    bidirection_predict, bidirection_predicted = bidirectional_lstm_predictor.predict(url)
    cnn_predict, cnn_predicted = cnn_lstm_predictor.predict(url)
    lstm_predict, lstm_predicted = lstm_predictor.predict(url)

    model_mapping = [
        {
            "model":"bidirectional_lstm",
            "is_phishing": True if bidirection_predict else False,
            "phishing_prediction": round(bidirection_predicted[-1] * 100,2),
            "non_phishing_prediction": round(bidirection_predicted[0] * 100,2)
        },
        {
            "model":"cnn",
            "is_phishing": True if cnn_predict else False,
            "phishing_prediction": round(cnn_predicted[-1] * 100,2),
            "non_phishing_prediction": round(cnn_predicted[0] * 100,2)
        },
        {
            "model":"lstm",
            "is_phishing": True if lstm_predict else False,
            "phishing_prediction": round(lstm_predicted[-1] * 100,2),
            "non_phishing_prediction": round(lstm_predicted[0] * 100,2)
        }
    ]
    return {
        "url": url,
        "model_mapping": model_mapping
    }
