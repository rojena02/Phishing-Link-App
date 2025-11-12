
from keras_malicious_url_detector.library.bidirectional_lstm import BidirectionalLstmEmbedPredictor
from keras_malicious_url_detector.library.cnn_lstm import CnnLstmPredictor
from keras_malicious_url_detector.library.lstm import LstmPredictor
from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr
from fastapi.middleware.cors import CORSMiddleware
from auth import (
    init_db,
    create_user,
    get_user_by_email,
    get_user_by_username,
    verify_password,
    get_db,
    create_access_token,
    get_current_user,
)

class URLData(BaseModel): 
    url: str

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
    url = url_data.url
    bidirection_predict, percentage_phishing = bidirectional_lstm_predictor.predict(url)
    cnn_predict, cnn_predicted = cnn_lstm_predictor.predict(url)
    lstm_predict, lstm_predicted = lstm_predictor.predict(url)
    print(bidirection_predict, percentage_phishing)
    
    model_mapping = [
        {
            "model":"bidirectional_lstm",
            "is_phishing": True if bidirection_predict else False,
            "phishing_prediction": round(percentage_phishing * 100,2),
            "non_phishing_prediction": round((1 -  percentage_phishing) * 100,2)
        },
        {
            "model":"cnn",
            "is_phishing": True if cnn_predict else False,
            "phishing_prediction": round(cnn_predicted * 100,2),
            "non_phishing_prediction": round((1 - cnn_predicted) * 100,2)
        },
        {
            "model":"lstm",
            "is_phishing": True if lstm_predict else False,
            "phishing_prediction": round(lstm_predicted * 100,2),
            "non_phishing_prediction": round((1-lstm_predicted) * 100,2)
        }
    ]
    return {
        "url": url,
        "model_mapping": model_mapping
    }
class SignupData(BaseModel):
    username: str
    email: EmailStr
    password: str


class LoginData(BaseModel):
    email: EmailStr
    password: str


@app.on_event("startup")
def on_startup():
    # ensure DB/tables exist when the app starts
    init_db()


@app.post("/signup")
def signup(data: SignupData):
    # simple signup that stores a hashed password
    db = next(get_db())
    existing = get_user_by_email(db, data.email)
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User already exists")
    # check username
    existing_username = get_user_by_username(db, data.username)
    if existing_username:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already taken")
    # ensure username uniqueness
    # Note: create_user will raise if username duplicate at DB level
    user = create_user(db, data.email, data.password, data.username)
    # create access token on signup (includes username)
    access_token = create_access_token(data={"sub": user.email, "username": user.username})
    return {"id": user.id, "email": user.email, "username": user.username, "access_token": access_token, "token_type": "bearer"}


@app.post("/login")
def login(data: LoginData):
    db = next(get_db())
    user = get_user_by_email(db, data.email)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found. Please sign up.")
    if not verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect password.")
    # create and return JWT access token
    access_token = create_access_token(data={"sub": user.email, "username": user.username})
    return {"access_token": access_token, "token_type": "bearer", "id": user.id, "email": user.email, "username": user.username}
