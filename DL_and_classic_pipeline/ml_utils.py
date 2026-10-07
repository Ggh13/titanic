import random
import numpy as np
import os
import wandb

from omegaconf import DictConfig
from omegaconf import OmegaConf

import pandas as pd

import joblib

from data_prepearing import set_seed, get_transforms, data_engineering, get_data_loaders

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')


def ml_train(config):
    print(f"\n{'='*50}")
    print(f"Начало обучения модели: {config.classic_mode}")
    print(f"{'='*50}\n")

    wandb.init(project=config.logging.wandb_project_name)

    if not os.path.exists(config.paths.checkpoint_dir):
        os.makedirs(config.paths.checkpoint_dir)

    # Получаем сырые данные и необученный препроцессор
    X_train_raw, X_val_raw, y_train, y_val, preprocessor = get_data_loaders(config)

    # Создаём базовую модель и param_grid в зависимости от типа
    if config.classic_mode == 'DecisionTree':
        base_model = DecisionTreeClassifier(random_state=42)
        param_grid = {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
    elif config.classic_mode == 'LogisticRegression':
        base_model = LogisticRegression(max_iter=1000)
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['liblinear', 'saga']
        }
    elif config.classic_mode == "randomForest":
        base_model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
        }
    elif config.classic_mode == "CatBoost":
        base_model = CatBoostClassifier(random_state=42, verbose=0)
        param_grid = {
            'iterations': [100, 200, 300],
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'l2_leaf_reg': [1, 3, 5],
        }
    elif config.classic_mode == "Xgboost":
        base_model = XGBClassifier(random_state=42, eval_metric='logloss')
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }
    else:
        raise ValueError(f'Invalid model type: {config.classic_mode}')


    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', base_model)
    ])


    param_grid_prefixed = {f'model__{key}': value for key, value in param_grid.items()}

    model = GridSearchCV(
        estimator=full_pipeline,
        param_grid=param_grid_prefixed,
        cv=5,
        scoring='accuracy',
        n_jobs=1
    )


    model.fit(X_train_raw, y_train)


    y_pred = model.predict(X_val_raw)
    print(f"Accuracy на валидации: {accuracy_score(y_val, y_pred):.4f}")
    best_model = model.best_estimator_

    print("Лучшие параметры:", model.best_params_)
    print("Лучший score на кросс-валидации:", model.best_score_)


    test_path = config.paths.test_csv
    test_df = pd.read_csv(test_path)
    test_df = data_engineering(test_df)

    test_predictions = best_model.predict(test_df)

    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': test_predictions
    })
    submission.to_csv('submission.csv', index=False)
    print(f"Submission сохранён! Всего записей: {len(submission)}")


    joblib.dump(model, os.path.join(config.paths.checkpoint_dir, 'model_classic.pkl'))
    joblib.dump(preprocessor, os.path.join(config.paths.checkpoint_dir, 'preprocessor_classic.pkl'))

    wandb.finish()
    print("Модель и препроцессор сохранены!")
