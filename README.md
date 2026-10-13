# Titanic Survival Prediction


## Сравнение моделей

| Модель          | Validation Accuracy | Leaderboard Score |
|-----------------|---------------------|-------------------|
| DecisionTree    | 0.7933              | 0                 |
| XGBoost         | 0.8189              | 0                 |
| CatBoost        | 0.8150              | 0                 |
| RandomForest    | 0.8045              | 0                 |
| KNN             | 0.8091              | 0                 |
| LightGBM        | 0.8044              | 0                 |

Эксперименты с DL моделью

| Модель                            | Validation Accuracy | Std Accuracy | Leaderboard Score |
|-----------------------------------|---------------------|--------------|-------------------|
| DNN (2 фолд) (BatchNorm1d)        | 0.8137              | —            | 0                 |
| DNN (4 фолдов) (BatchNorm1d)      | 0.8204              | —            | 0                 |
| DNN (6 фолдов) (BatchNorm1d)      | 0.8249              | 0.0175       | 0                 |
| DNN (8 фолдов) (BatchNorm1d)      | 0.8249              | 0.0223       | 0                 |
| DNN (9 фолдов) (BatchNorm1d)      | 0.8193              | 0.0345       | 0                 |
| DNN (2 фолд) (LayerNorm)          | 0.8171              | 0.0013       | 0                 |
| DNN (4 фолдов) (LayerNorm)        | 0.8115              | 0.0099       | 0                 |
| DNN (6 фолдов) (LayerNorm)        | 0.8194              | 0.0313       | 0                 |
| DNN (8 фолдов) (LayerNorm)        | 0.8215              | 0.0308       | 0                 |
| DNN (9 фолдов) (LayerNorm)        | 0.8238              | 0.0357       | 0                 |

BatchNorm показывает себя лучше, будем использовать его.

Далее попробуем изменить количество внутренних слоев при количестве фолдов 5 при кросс-валидации:

| Количество нейронов в 1 слое | Validation Accuracy | Mean val | Количество эпох | Количество слоев |
|------------------------------|---------------------|----------|-----------------|------------------|
| DNN (64)                     | 0.8238              | —        | 60              | 3                |
| DNN (128)                    | 0.8249              | —        | 60              | 3                |
| DNN (256)                    | 0.8204              | —        | 70              | 3                |
| DNN (512)                    | 0.8238              | —        | 70              | 3                |
| DNN (1024)                   | 0.8249              | —        | 100             | 3                |
| DNN (1024)                   | 0.8137              | —        | 100             | 3                |
| DNN (1024)                   | 0.8193              | —        | 100             | 3                |
| DNN (1024)                   | 0.8126              | —        | 100             | 5                |
| DNN (1024)                   | 0.8204              | —        | 100             | 5                |

> **Примечания:**
> - DNN (1024), 0.8137 — lr уменьшен с `1e-3` до `1e-4`
> - DNN (1024), 0.8193 — lr увеличен с `1e-4` до `5e-4`, dropout уменьшен с 0.2 до 0.1
> - DNN (1024), 0.8126 — 5 слоев, не удалось улучшить, скорее переобучилась
> - DNN (1024), 0.8204 — 5 слоев, lr увеличен с `5e-4` до `1e-3`

По итогу оптимальным выбором для DL служит 512 нейронов в первом слое.

Эксперименты с функциями активации.
Количество нейронов 512, эпох 50, lr = 1e-3, dropout 0.15, количество слоев 3.

| Модель            | Validation Accuracy | Примечание        |
|-------------------|---------------------|-------------------|
| DNN (ReLU)        | 0.8193              |                   |
| DNN (LeakyReLU)   | 0.8170              |                   |
| DNN (GELU)        | 0.8238              | Лучший результат  |

По итогу GELU показал лучший результат, он и будет использоваться далее.

Эксперименты с оптимизатором

| Модель            | Validation Accuracy |
|-------------------|---------------------|
| DNN (AdamW)       | 0.8238              |
| DNN (Adam)        | 0.8238              |
| DNN (SGD)         | 0.8171              |

Как показала практика, Adam и AdamW не отличаются по результату. Далее будет использоваться AdamW.

Эксперименты с ансамблями.

Voting:

| Комбинация моделей                                  | Validation Accuracy |
|----------------------------------------------------|---------------------|
| XGBoost + CatBoost + LightGBM + KNN                | 0.8301              |
| RandomForest + CatBoost + LightGBM + KNN           | 0.8156              |
| XGBoost + CatBoost + LightGBM + RandomForest       | 0.7989              |

Stacking:

| Модель              | Validation Accuracy |
|---------------------|---------------------|
| XGBoost             | 0.7933              |
| CatBoost            | 0.8212              |
| LightGBM            | 0.8045              |
| RandomForest        | 0.8101              |
| Stacking (мета-модель) | 0.8045              |

По итогу всех испытаний, лучший результат показал Voting с результатом 0.83.
Обучу этот ансамбль еще раз и узнаю скор в лидерборде.

### Результат на лидерборде 0.7823

## Структура проекта

| Файл | Описание |
|------|----------|
| `DL_and_classic_pipeline/main.py` | Точка входа. Выбор режима работы |
| `DL_and_classic_pipeline/config.py` | Конфигурация проекта (OmegaConf) |
| `DL_and_classic_pipeline/models.py` | Нейросеть (MLP) для классификации |
| `DL_and_classic_pipeline/data_prepearing.py` | Загрузка, препроцессинг, DataLoader |
| `DL_and_classic_pipeline/ml_utils.py` | Классические ML-модели. Обучение и предсказания |
| `DL_and_classic_pipeline/utils.py` | Обучение, K-Fold, инференс нейросети |
| `DL_and_classic_pipeline/draw.ipynb` | EDA (исследование данных) |
| `DL_and_classic_pipeline/requirements.txt` | Зависимости проекта |
| `DL_and_classic_pipeline/submission.csv` | Файл предсказаний для Kaggle |
| `DL_and_classic_pipeline/data/` | Данные (train.csv, test.csv) |
| `checkpoints/` | Сохранённые модели и препроцессоры |
| `wandb/` | Логи Weights & Biases |

## Параметры конфигурации (`DL_and_classic_pipeline/config.py`)

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `seed` | Seed для воспроизводимости | `42` |
| `mode` | Режим работы (см. ниже) | `"voting"` |
| `type_classic` | Тип классического ML (для `classic_ml`) | `"voting"` |
| `device` | Устройство для обучения | `"cuda"` |
| `training.debug` | Режим отладки (ограничение сэмплов) | `False` |
| `training.number_of_debug_samples` | Количество сэмплов в debug | `1000` |
| `training.lr` | Learning rate | `1e-3` |
| `training.epochs` | Количество эпох | `50` |
| `training.batch_size` | Размер батча | `32` |
| `model.hidden_dim` | Количество нейронов в первом скрытом слое | `512` |
| `model.num_layers` | Количество скрытых слоев | `3` |
| `model.dropout` | Вероятность dropout | `0.15` |
| `data.kfold.n_splits` | Количество фолдов | `5` |
| `data.kfold.val_size` | Размер валидации | `0.2` |
| `paths.checkpoint_dir` | Папка для чекпоинтов | `"./checkpoints"` |
| `paths.train_csv` | Путь к train.csv | `"./data/train.csv"` |
| `paths.test_csv` | Путь к test.csv | `"./data/test.csv"` |
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
| `voting` | Ансамбль голосованием ( majority voting ) |
| `stacking` | Ансамбль стекингом с мета-моделью |

## Режим `classic_ml` (файл: `DL_and_classic_pipeline/ml_utils.py`)

### Доступные модели

| `type` | Модель | GridSearchCV |
|--------|--------|--------------|
| `'LogisticRegression'` | LogisticRegression | Да (C, penalty, solver) |
| `'DecisionTree'` | DecisionTreeClassifier | Нет |
| `'randomForest'` | RandomForestClassifier | Да (n_estimators, max_depth, ...) |
| `'CatBoost'` | CatBoostClassifier | Да (iterations, depth, learning_rate, ...) |
| `'Xgboost'` | XGBClassifier | Да (n_estimators, max_depth, learning_rate, ...) |
| `'KNN'` | KNeighborsClassifier | Да (n_neighbors, weights, ...) |
| `'LightGBM'` | LGBMClassifier | Да (n_estimators, max_depth, learning_rate, ...) |

### Как изменить модель

В `ml_utils.py` функция `ml_train`:

```python
model = get_model(config, type = 'Xgboost', in_features = config.in_features)
```

Замените `'Xgboost'` на нужный `type` из таблицы выше.

### Как изменить гиперпараметры

В функции `get_model` для каждой модели задан `param_grid` — словарь с перебираемыми параметрами для GridSearchCV.

## Режим `voting` (файл: `DL_and_classic_pipeline/ml_utils.py`)

### Как это работает

1. Обучаются несколько классических ML-моделей (XGBoost, CatBoost, LightGBM, RandomForest, KNN)
2. Каждая модель предсказывает класс (0 или 1)
3. Итоговое предсказание — большинство голосов (majority voting)

### Как выбрать модели для голосования

В функции `voting_train` в `ml_utils.py` задаётся список моделей:

```python
models = ['Xgboost', 'CatBoost', 'LightGBM', 'KNN']
```

Можно комбинировать любые модели из списка доступных. Оптимальные комбинации (по результатам экспериментов):

| Комбинация | Validation Accuracy |
|------------|---------------------|
| XGBoost + CatBoost + LightGBM + KNN | 0.8301 |
| RandomForest + CatBoost + LightGBM + KNN | 0.8156 |
| XGBoost + CatBoost + LightGBM + RandomForest | 0.7989 |

## Режим `stacking` (файл: `DL_and_classic_pipeline/ml_utils.py`)

### Как это работает

1. Каждая базовая модель обучается с K-Fold кросс-валидацией
2. Предсказания out-of-fold (OOF) используются как признаки для мета-модели
3. Мета-модель (LogisticRegression) обучается на OOF-предсказаниях
4. Итоговое предсказание — результат мета-модели

### Как выбрать модели для стекинга

В функции `stacking_train` в `ml_utils.py` задаётся список базовых моделей. По результатам экспериментов:

| Модель | Validation Accuracy (OOF) |
|--------|---------------------------|
| XGBoost | 0.7933 |
| CatBoost | 0.8212 |
| LightGBM | 0.8045 |
| RandomForest | 0.8101 |
| Stacking (мета-модель) | 0.8045 |

## Режимы нейросети (файл: `DL_and_classic_pipeline/utils.py`)

### `classic_training`

Обычное обучение MLP с разделением на train/val. Использует параметры из `config.training` и `config.model`.

### `K_fold`

K-Fold кросс-валидация нейросети. Для каждого фолда создаётся отдельная модель, веса сохраняются в `checkpoints/model_fold_{i}.pth`.

### `test_inference_K_fold`

Загружает все обученные модели K-Fold и усредняет их вероятности для финального предсказания.

### Архитектура нейросети (`models.py`)

```
Input (n_features)
  → Linear(n_features, hidden_dim)
  → BatchNorm1d
  → GELU
  → Dropout
  → Linear(hidden_dim, hidden_dim // 2)
  → BatchNorm1d
  → GELU
  → Dropout
  → Linear(hidden_dim // 2, hidden_dim // 4)
  → BatchNorm1d
  → GELU
  → Dropout
  → Linear(..., 1)
```

Размерность каждого следующего слоя уменьшается вдвое. При `hidden_dim=512` и `num_layers=3`: 512 → 256 → 128 → 1.
