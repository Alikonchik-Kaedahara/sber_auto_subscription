import datetime

import dill
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

CAT_FEATURES = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
                'device_category', 'device_os', 'device_brand', 'device_browser', 'geo_country', 'geo_city']
NUM_FEATURES = ['visit_number', 'visit_hour', 'visit_weekday', 'screen_width', 'screen_height']

# параметры бустинга и порог подобраны в 02_model.ipynb
PARAMS = {'learning_rate': 0.1, 'max_iter': 300, 'max_leaf_nodes': 15, 'min_samples_leaf': 200, 'l2_regularization': 0.0}
THRESHOLD = 0.06


def prepare(df):
    df = df.copy()
    df = df.replace('(not set)', np.nan)

    # device_os часто пустой, но его можно восстановить по бренду
    df.loc[df['device_os'].isna() & (df['device_brand'] == 'Apple'), 'device_os'] = 'iOS'
    df.loc[df['device_os'].isna() & df['device_brand'].notna() & (df['device_brand'] != 'Apple')
           & df['device_category'].isin(['mobile', 'tablet']), 'device_os'] = 'Android'
    df[CAT_FEATURES] = df[CAT_FEATURES].fillna('other')

    df['visit_number'] = pd.to_numeric(df['visit_number'])
    df['visit_hour'] = pd.to_datetime(df['visit_time'], format='%H:%M:%S').dt.hour
    df['visit_weekday'] = pd.to_datetime(df['visit_date']).dt.day_of_week

    # разрешение экрана в виде строки '360x720' делим на ширину и высоту
    res = df['device_screen_resolution'].astype(str).str.split('x', expand=True).reindex(columns=[0, 1])
    df['screen_width'] = pd.to_numeric(res[0], errors='coerce')
    df['screen_height'] = pd.to_numeric(res[1], errors='coerce')
    bad = ((df['screen_width'] < 100) | (df['screen_height'] < 100)
           | (df['screen_width'] > 8000) | (df['screen_height'] > 8000))
    df.loc[bad, ['screen_width', 'screen_height']] = np.nan
    return df[CAT_FEATURES + NUM_FEATURES]


def make_pipeline():
    # редкие значения (меньше 100 визитов) объединяются в одну категорию,
    # новые значения, которых не было в обучении, модель считает пропуском
    encoder = ColumnTransformer([
        ('cat', OrdinalEncoder(min_frequency=100, max_categories=250,
                               handle_unknown='use_encoded_value', unknown_value=-1), CAT_FEATURES),
        ('num', 'passthrough', NUM_FEATURES),
    ], verbose_feature_names_out=False).set_output(transform='pandas')

    classifier = HistGradientBoostingClassifier(categorical_features=CAT_FEATURES, early_stopping=False,
                                                random_state=42, **PARAMS)

    return Pipeline([
        ('prepare', FunctionTransformer(prepare)),
        ('encoder', encoder),
        ('classifier', classifier),
    ])


def main():
    print('Sber Auto Subscription: target action prediction')

    sessions = pd.read_csv('data/ga_sessions.csv', dtype=str)
    target = pd.read_csv('data/target.csv', dtype={'session_id': str})
    df = sessions.merge(target[['session_id', 'target']], on='session_id')
    X = df.drop(columns='target')
    y = df['target'].values
    print('visits:', len(df), 'conversion:', round(y.mean(), 4))

    # визиты одного клиента попадают только в одну из выборок
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                               .split(X, y, groups=df['client_id']))

    pipe = make_pipeline()
    pipe.fit(X.iloc[train_idx], y[train_idx])
    proba = pipe.predict_proba(X.iloc[test_idx])[:, 1]
    roc_auc = roc_auc_score(y[test_idx], proba)
    f1 = f1_score(y[test_idx], (proba >= THRESHOLD).astype(int))
    print(f'test roc_auc: {roc_auc:.4f}, f1: {f1:.4f}')

    # финальную модель обучаем на всех данных
    pipe = make_pipeline()
    pipe.fit(X, y)

    model_filename = 'model/sber_auto_pipe.pkl'
    with open(model_filename, 'wb') as file:
        dill.dump({
            'model': pipe,
            'threshold': THRESHOLD,
            'metadata': {
                'name': 'Sber Auto Subscription target action prediction',
                'author': 'Alikhan Yessenov',
                'version': 1,
                'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                'type': type(pipe.named_steps['classifier']).__name__,
                'roc_auc': round(roc_auc, 4),
                'threshold': THRESHOLD,
            }
        }, file, recurse=True)
    print('model saved to', model_filename)


if __name__ == '__main__':
    main()
