import time

import pandas as pd
import requests

URL = 'http://127.0.0.1:8000'

# 1000 visits from the sample, sent to the service one by one and then in one batch
visits = pd.read_csv('data_sample/ga_sessions_sample.csv', dtype=str)
visits['visit_number'] = visits['visit_number'].astype(int)
visits = visits.astype(object).where(visits.notna(), None)

print(requests.get(URL + '/status').json())
print(requests.get(URL + '/version').json())

times = []
for row in visits.to_dict(orient='records'):
    start = time.time()
    response = requests.post(URL + '/predict', json=row)
    times.append(time.time() - start)
    response.raise_for_status()

print('example response:', response.json())
print(f'requests: {len(times)}, mean time: {sum(times) / len(times) * 1000:.1f} ms, '
      f'max: {max(times) * 1000:.1f} ms')

start = time.time()
response = requests.post(URL + '/predict_batch', json=visits.to_dict(orient='records'))
print(f'batch of {len(visits)} visits: {(time.time() - start) * 1000:.1f} ms, '
      f'predicted ones: {sum(r["target"] for r in response.json())}')
