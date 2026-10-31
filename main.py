import dill
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

with open('model/sber_auto_pipe.pkl', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    session_id: str
    client_id: str | None = None
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: str | None = None
    utm_medium: str | None = None
    utm_campaign: str | None = None
    utm_adcontent: str | None = None
    utm_keyword: str | None = None
    device_category: str | None = None
    device_os: str | None = None
    device_brand: str | None = None
    device_model: str | None = None
    device_screen_resolution: str | None = None
    device_browser: str | None = None
    geo_country: str | None = None
    geo_city: str | None = None


class Prediction(BaseModel):
    session_id: str
    target: int
    probability: float


def make_prediction(df):
    proba = model['model'].predict_proba(df)[:, 1]
    pred = (proba >= model['threshold']).astype(int)
    return [{'session_id': s, 'target': int(t), 'probability': round(float(p), 4)}
            for s, t, p in zip(df['session_id'], pred, proba)]


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame([form.model_dump()])
    return make_prediction(df)[0]


# if there are many visits, it's better to send them in one request: the model scores the whole table at once
@app.post('/predict_batch', response_model=list[Prediction])
def predict_batch(forms: list[Form]):
    df = pd.DataFrame([form.model_dump() for form in forms])
    return make_prediction(df)
