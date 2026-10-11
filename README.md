# Titanic Survival Prediction


## Сравнение моделей

| Модель          | Validation Accuracy | Leaderboard Score |
|-----------------|---------------------|-------------------|
| DecisionTree    | 0.7933              | 0          |         
| XGBoost         | 0.8189              | 0          |
| CatBoost        | 0.815              | 0             |
| RandomForest    | 0.8045              | 0             |
| KNN             | 0.8091           | 0             |
| LightGBM        | 0.8044           | 0             |

Эксперименты кад DL моделью

| DNN (2 фолд) (BatchNorm1d)   | 0.8137                                                       | 0           |
| DNN (4 фолдов) (BatchNorm1d) | 0.8204                                                       | 0           |
| DNN (6 фолдов) (BatchNorm1d) | 0.8249  Std Accuracy:   0.0175  (Дисперсия повысилась )      | 0           |
| DNN (8 фолдов) (BatchNorm1d) | 0.8249  Std Accuracy:   0.0223  (Дисперсия повысилась )      | 0           |
| DNN (9 фолдов) (BatchNorm1d) | 0.8193  Std Accuracy:   0.0345  (Дисперсия повысилась )      | 0           |

| DNN (2 фолд) (LayerNorm)   | 0.8171  Std Accuracy:   0.0013                                 | 0           |
| DNN (4 фолдов) (LayerNorm) | 0.8115  Std Accuracy:   0.0099                                 | 0           |
| DNN (6 фолдов) (LayerNorm) | 0.8194  Std Accuracy:   0.0313  (Дисперсия повысилась )        | 0           |
| DNN (8 фолдов) (LayerNorm) | 0.8215  Std Accuracy:   0.0308  (Дисперсия повысилась )        | 0           |
| DNN (9 фолдов) (LayerNorm) | 0.8238 Std Accuracy:   0.0357  (Дисперсия повысилась )         | 0           |

BatchNorm показывает себя лучше, будем использовать его.

Далее попробуем  изменить кол - во внутренних слоев при кол во фолдов 5 при кросс валидации:

Кол во нейронов в 1 слое:  Mean val   Кол во эпох   кол во слоев
| DNN (64)   | 0.8238  | 60     | 3
| DNN (128)  | 0.8249  | 60     | 3
| DNN (256)  | 0.8204  | 70     | 3
| DNN (512)  | 0.8238  | 70     | 3
| DNN (1024) | 0.8249  | 100    | 3
| DNN (1024) | 0.8137  | 100    | 3              lr уменьшен с "lr": 1e-3, до "lr": 1e-4,
| DNN (1024) | 0.8193  | 100    | 3          lr увеличен с "lr": 1e-4, до "lr": 5e-4, dropout уменьшен с 0.2 до 0.1
| DNN (1024) | 0.8126  | 100    | 5         не удалось улучшить, даже скорее вереобучилась
| DNN (1024) | 0.8204  | 100    | 5         lr увеличен с "lr": 5e-4, до "lr": 1e-3,

По итогу оптимальным выбором для DL служит 512 нейронов в первом слое.

Эксперименты с функциями активации.
Кол во нейронов 512, эпох 50, lr = 1e-3, dropout 0.15, кол во слоев 3.

| DNN (ReLU)   |  0.8193  | 
| DNN (LeakyReLU)   |  0.8170  | 
| DNN (GELU)   |  0.8238  |  Лучший результат

По итогу GELU показал лучший результат, он и будет использоваться далее.
Эксперименты с оптимизатором

| DNN (AdamW)   | 0.8238   | 
| DNN (Adam)    | 0.8238   | 
| DNN (SGD)   |   0.8171  |  

Как показала практика, Adam и AdamW не отличаются по результату. Далее будет использоваться AdamW

Эксперименты с ансамблями.

Voting:
['Xgboost', 'CatBoost', 'LightGBM', 'KNN']: 0.8301
['randomForest', 'CatBoost', 'LightGBM', 'KNN']: 0.8156
['Xgboost', 'CatBoost', 'LightGBM', 'randomForest'] 0.7989

Stacking:
  Xgboost: 0.7933
  CatBoost: 0.8212
  LightGBM: 0.8045
  randomForest: 0.8101
  Stacking (мета-модель): 0.8045

По итогу всех испытаний, лучший результат показал Voting с результатом 0.83.
Обучу этот ансамбль еще раз и узнаю скор в лидерборде.
Результат на лидерборде 0.7823

## Структура проекта

| Файл | Описание |
|------|----------|
| `DL_pipeline/main.py` | Точка входа. Выбор режима работы |
| `DL_pipeline/config.py` | Конфигурация проекта (OmegaConf) |
| `DL_pipeline/models.py` | Нейросеть (MLP) для классификации |
| `DL_pipeline/data_prepearing.py` | Загрузка, препроцессинг, DataLoader |
| `DL_pipeline/ml_utils.py` | Классические ML-модели. Обучение и предсказания |
| `DL_pipeline/utils.py` | Обучение, K-Fold, инференс нейросети |
| `DL_pipeline/draw.ipynb` | EDA (исследование данных) |
| `DL_pipeline/requirements.txt` | Зависимости проекта |
| `DL_pipeline/submission.csv` | Файл предсказаний для Kaggle |
| `DL_pipeline/data/` | Данные (train.csv, test.csv) |
| `checkpoints/` | Сохранённые модели и препроцессоры |
| `wandb/` | Логи Weights & Biases |

## Параметры конфигурации (`DL_pipeline/config.py`)

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `seed` | Seed для воспроизводимости | `42` |
| `mode` | Режим работы (см. ниже) | `"classic_ml"` |
| `device` | Устройство для обучения | `"cuda"` |
| `in_features` | Количество входных признаков | `9` |
| `training.debug` | Режим отладки (1000 samples) | `False` |
| `training.number_of_debug_samples` | Количество сэмплов в debug | `1000` |
| `training.lr` | Learning rate | `1e-3` |
| `training.epochs` | Количество эпох | `60` |
| `training.batch_size` | Размер батча | `32` |
| `data.kfold.use_kfold` | Использовать K-Fold | `True` |
| `data.kfold.n_splits` | Количество фолдов | `6` |
| `data.kfold.val_size` | Размер валидации | `0.2` |
| `paths.checkpoint_dir` | Папка для чекпоинтов | `"./checkpoints"` |
| `paths.train_csv` | Путь к train.csv | `"./data/train.csv"` |
| `logging.wandb_project_name` | Имя проекта в W&B | `"titanic-classification"` |
| `files.data` | Папка с данными | `"./data/"` |
| `files.train` | Имя файла train | `"train.csv"` |

## Режимы работы (`mode`)

| Mode | Описание |
|------|----------|
| `classic_training` | Обучение нейросети с train/val split |
| `K_fold` | K-Fold обучение нейросети |
| `test_inference_K_fold` | Предсказание на тесте через K-Fold |
| `classic_ml` | Классические ML-модели с GridSearchCV |

## Режим `classic_ml` (файл: `DL_pipeline/ml_utils.py`)

### Доступные модели

| `type` | Модель | GridSearchCV |
|--------|--------|--------------|
| `'LogisticRegression'` | LogisticRegression | Да (C, penalty, solver) |
| `'DecisionTree'` | DecisionTreeClassifier | Нет |
| `'randomForest'` | RandomForestClassifier | Да (n_estimators, max_depth, ...) |
| `'CatBoost'` | CatBoostClassifier | Да (iterations, depth, learning_rate, ...) |
| `'Xgboost'` | XGBClassifier | Да (n_estimators, max_depth, learning_rate, ...) |

### Как изменить модель

В `ml_utils.py` строка 130, функция `ml_train`:

```python
model = get_model(config, type = 'Xgboost', in_features = config.in_features)
```

Замените `'Xgboost'` на нужный `type` из таблицы выше.

### Как изменить гиперпараметры

В функции `get_model` для каждой модели задан `param_grid` — словарь с перебираемыми параметрами для GridSearchCV.
