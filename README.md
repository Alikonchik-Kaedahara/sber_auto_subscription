# SberAutoSubscription: predicting the target action

Final project of the ML Engineer course. The model takes the attributes of a visit to the website (utm_*, device_*,
geo_*, date, time and visit number) and predicts whether the user will do a target action: leave a request,
ask for a call and so on. The answer is 1 if they will and 0 if they won't.

## Files

- `01_EDA.ipynb` exploratory data analysis, it also builds `data/target.csv`
- `02_model.ipynb` model comparison, parameter search, feature importance
- `pipeline.py` trains the final model and saves it to `model/sber_auto_pipe.pkl`
- `main.py` FastAPI service
- `check_api.py` checks the service and the response time
- `data_sample/ga_sessions_sample.csv` 1000 visits in the input format
- `requirements.txt` libraries

## Data

Google Analytics data of the SberAutoSubscription (СберАвтоподписка) website from May to December 2021:
1.86 million visits (`ga_sessions.csv`) and 15.7 million events inside them (`ga_hits.csv`, about 4 GB).
The data was provided for an educational project, so the full dataset is not published here.

`data_sample/` has 1000 random visits from `ga_sessions.csv` to show the input format and to try the service.
The session and client IDs in it are replaced with made-up ones (`sample_0001`, `client_0001`).

The trained model is already in `model/`, so the full data is not needed to run the service. To rerun the
notebooks or retrain the model, put `ga_sessions.csv` and `ga_hits.csv` into `data/`.

## Results

| Model | ROC-AUC on test |
|---|---|
| Baseline: conversion by utm_source | 0.612 |
| Logistic regression | 0.675 |
| Gradient boosting (HistGradientBoostingClassifier) | 0.696 |

The model comparison is in `02_model.ipynb`. The test set there is 20% of clients who were not in the training
data (I split by client_id, so visits of one person don't end up in both sets). In `pipeline.py` the model is
trained on 80% of clients and gets ROC-AUC 0.702 on the held-out clients, then the final model is trained on all
the data.

The ad campaign and traffic source (utm_campaign, utm_source, utm_keyword, utm_adcontent), the city and the visit
number affect the prediction the most. The service returns 1 if the probability of a target action is 0.06 or
higher: such visits convert almost 3 times more often than average.

## Running the service

Python 3.12 or newer is needed.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app
```

The service starts on http://127.0.0.1:8000, the docs and a form to try it are at http://127.0.0.1:8000/docs.

Endpoints:

- `GET /status` check that the service is running
- `GET /version` model info (training date, ROC-AUC, threshold)
- `POST /predict` one visit, the answer is 0 or 1
- `POST /predict_batch` a list of visits, an answer for each

Request example (the fields are the same as in ga_sessions, empty ones can be skipped):

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "session_id": "sample_0001",
  "visit_date": "2021-05-19",
  "visit_time": "15:00:00",
  "visit_number": 2,
  "utm_source": "fDLlAcSmythWSCVMvqvL",
  "utm_medium": "(none)",
  "utm_campaign": "LTuZkdKfxRGVceoWkVyg",
  "utm_adcontent": "JNHcPlZPxEMWDnRiyoBf",
  "device_category": "mobile",
  "device_brand": "Apple",
  "device_screen_resolution": "414x896",
  "device_browser": "Chrome",
  "geo_country": "Russia",
  "geo_city": "Moscow"
}'
```

Response:

```json
{"session_id": "sample_0001", "target": 0, "probability": 0.0395}
```

Speed check (the service has to be running, it uses the visits from `data_sample/`):

```bash
python check_api.py
```

My result: 1000 requests with one visit each, mean response time 18.4 ms, max 38 ms. A batch of 1000 visits in
one request takes 33 ms. The 3 second limit is met with a big margin.

The service doesn't keep any state, the model is loaded once at start. So it can run in several processes
(`uvicorn main:app --workers 4`) or on several machines behind a load balancer. If there are many visits, it's
better to send them in batches to `/predict_batch`.

## Retraining the model

```bash
python pipeline.py
```

The script reads `data/ga_sessions.csv` and `data/target.csv`, checks the quality on held-out clients and trains
the model on all the data. It takes less than a minute (40 seconds on my laptop). New visits come from different ad
campaigns, so the model should be retrained regularly, for example once a month.
