import time

import pandas as pd
import requests

URL = 'http://127.0.0.1:8000'

# берем несколько визитов из исходных данных и отправляем их в сервис по одному
visits = pd.read_csv('data/ga_sessions.csv', dtype=str, nrows=5000).sample(200, random_state=1)
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

print('пример ответа:', response.json())
print(f'запросов: {len(times)}, среднее время: {sum(times) / len(times) * 1000:.1f} мс, '
      f'максимум: {max(times) * 1000:.1f} мс')

start = time.time()
response = requests.post(URL + '/predict_batch', json=visits.to_dict(orient='records'))
print(f'пакет из {len(visits)} визитов: {(time.time() - start) * 1000:.1f} мс, '
      f'предсказано единиц: {sum(r["target"] for r in response.json())}')
