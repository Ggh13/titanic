# Titanic Survival Prediction

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
