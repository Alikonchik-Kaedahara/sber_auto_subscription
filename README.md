# SberAutoSubscription: target action prediction

Final project of the ML Engineer course. The model gets a visit to the site (utm_*, device_*, geo_*, date, time,
visit number) and says if the user will do a target action (leave a request, ask for a call and so on): 1 or 0.

## Files

- `01_EDA.ipynb` - data cleaning and EDA, also makes `data/target.csv`
- `02_model.ipynb` - models, parameters, feature importance
- `pipeline.py` - trains the final model and saves it to `model/sber_auto_pipe.pkl`
- `main.py` - FastAPI service
- `check_api.py` - sends visits from the sample to the service and measures the time
- `data_sample/ga_sessions_sample.csv` - 1000 visits to try the service

## Data

Google Analytics data of the SberAutoSubscription (СберАвтоподписка) website, May to December 2021:
`ga_sessions.csv` (1.86 million visits) and `ga_hits.csv` (15.7 million events, about 4 GB). It was given for an
educational project, so the full data is not in the repo. `data_sample/` has 1000 random visits with the session and
client ids replaced. To rerun the notebooks or `pipeline.py` put the full files into `data/`.

## Result

ROC-AUC on the test set (clients that were not in train):

| model | ROC-AUC |
|---|---|
| baseline (conversion by utm_source) | 0.619 |
| logistic regression | 0.676 |
| HistGradientBoosting | 0.689 |

The most important features are the utm tags (campaign, source, keyword, ad content), the city and the visit number.
The service returns 1 if the probability is 0.06 or more.

## How to run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app
```

Then http://127.0.0.1:8000/docs. Endpoints: `/status`, `/version`, `/predict` (one visit), `/predict_batch` (a list).

Example:

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

`python check_api.py` (with the service running) sends the 1000 sample visits one by one. For me it was about 17 ms per request.

To retrain: `python pipeline.py` (needs the full data, takes about a minute).
