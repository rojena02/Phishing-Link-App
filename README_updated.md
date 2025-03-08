cd into /phishing_link_app/backend

setup virtual env
python -m venv venv

activate virtual env
--> windows : venv/Scripts/activate
--> mac/linux: venv/bin/activate

install requirements
pip install -r requirements.txt

to run training

cd into /backend/app

python <<model_name>>\_train.py e.g. python lstm_train.py

once the models are trained, you can run the backend server as

uvicorn main.app
