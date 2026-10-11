import random
import numpy as np
import torch
import os
import wandb


from omegaconf import DictConfig
from omegaconf import OmegaConf

import pandas as pd

import joblib
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold
from models import ClassifyModel

from data_prepearing import set_seed, get_transforms, data_engineering, get_data_loaders

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings('ignore')
def get_model(config, type: str = 'LogisticRegression', in_features: int = 9):
    """
    Создаёт модель с GridSearchCV для классических ML-алгоритмов.

    Args:
        config: конфигурация
        type: тип модели ('DecisionTree', 'LogisticRegression', 'randomForest',
              'CatBoost', 'Xgboost', 'KNN', 'LightGBM')
        in_features: количество входных признаков (не используется для ML-моделей)

    Returns:
        GridSearchCV: обёртка с перебором гиперпараметров

    Raises:
        ValueError: если type не поддерживается
    """
    model = ""
    if type == 'DecisionTree':
        param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        }
        model = GridSearchCV(
            estimator=DecisionTreeClassifier(random_state=42),
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=1
        )

    elif type == 'LogisticRegression':
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['liblinear', 'saga']
        }

        grid_search = GridSearchCV(
            estimator=LogisticRegression(max_iter=1000),
            param_grid=param_grid,
            cv=5,               
            scoring='accuracy', 
            n_jobs=-1           
        )
        model = grid_search
    elif type == "randomForest":
        param_grid = {
            'n_estimators': [50, 100, 200],  # Количество деревьев в лесу
            'max_depth': [None, 5, 10, 15],  # Максимальная глубина деревьев
            'min_samples_split': [
                2,
                5,
                10,
            ],  # Минимальное число объектов для разбиения узла
            'min_samples_leaf': [1, 2, 4],  # Минимальное число объектов в листе
            'max_features': ['sqrt', 'log2'],  # Число признаков для выбора при сплите
        }

        # 2. Инициализируем базовую модель
        rf = RandomForestClassifier(random_state=42)

        # 3. Настраиваем GridSearch
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,  # 5-фолдовая кросс-валидация
            scoring='accuracy',
            n_jobs=1,  # Используем 1 поток во избежание проблем с памятью на Windows
        )
        model = grid_search
    elif type == "CatBoost":
        param_grid = {
            'iterations': [100, 200],  # Аналог n_estimators (количество деревьев)
            'depth': [4, 6],  # Глубина деревьев (обычно 4-6 оптимально)
            'learning_rate': [0.01, 0.05],  # Скорость обучения
            'l2_leaf_reg': [3, 5],  # L2-регуляризация (защита от переобучения)
        }

        # 2. Инициализация (verbose=0 выключает спам логов в консоль)
        cb = CatBoostClassifier(random_state=42, verbose=0)

        # 3. GridSearch
        grid_search_cb = GridSearchCV(
            estimator=cb,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=1,  # Защита от ошибки с памятью на Windows
        )
        model = grid_search_cb
    elif type == "Xgboost":
        param_grid = {
            'n_estimators': [50, 100, 200],  # Количество деревьев
            'max_depth': [3, 5, 7],  # Глубина деревьев
            'learning_rate': [0.01, 0.05, 0.1],  # Скорость обучения
            'subsample': [0.8, 1.0],  # Доля выборки для построения дерева
            'colsample_bytree': [0.8, 1.0],  # Доля признаков для каждого дерева
        }

        # 2. Инициализация
        xgb = XGBClassifier(random_state=42, eval_metric='logloss')

        # 3. GridSearch
        grid_search_xgb = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=1,
        )
        model = grid_search_xgb
    elif type == "KNN":
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan'],
        }
        knn = GridSearchCV(
            estimator=KNeighborsClassifier(),
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=1
        )
        model = knn
    elif type == "LightGBM":
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05],
            'num_leaves': [15, 31],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }
        lgbm = GridSearchCV(
            estimator=LGBMClassifier(random_state=42, verbose=-1),
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=1
        )
        model = lgbm
    else:
        raise ValueError(f'Invalid model type: {type}')
    return model

def ml_train(config):
    """
    Обучает классическую ML-модель с GridSearchCV.
    Сохраняет модель, препроцессор и submission.csv.

    Args:
        config: конфигурация с type_classic, paths.*, training.*
    """
    wandb.init(project=config.logging.wandb_project_name)

    if not os.path.exists(config.paths.checkpoint_dir):
        os.makedirs(config.paths.checkpoint_dir)
    
    X_train, X_val, y_train, y_val, pipeline = get_data_loaders(config)
    torch.cuda.empty_cache()

    model = get_model(config, type = config.type_classic, in_features = -1)
    #model = model.to(config.device)

    X_train = pipeline.fit_transform(X_train)

    X_val = pipeline.transform(X_val)


    model.fit(X_train, y_train)


    

    y_pred=model.predict(X_val)
    print(f"Accuracy score: {accuracy_score(y_val,y_pred)}")
    best_model = model.best_estimator_

    # 1. Вывести только те параметры, которые подбирались через сетку (param_grid)
    print("Лучшие подломанные параметры:", model.best_params_)

    # 2. Вывести абсолютно все параметры объекта лучшей модели
    print("Все параметры модели:", best_model.get_params())

    # 3. (Опционально) Вывести средний результат (Accuracy) лучшей модели на кросс-валидации
    print("Лучший score на кросс-валидации:", model.best_score_)
    if hasattr(model, 'state_dict'):
        torch.save(model.state_dict(), os.path.join(config.paths.checkpoint_dir, 'model_weights.pth'))
    else:
        joblib.dump(model, os.path.join(config.paths.checkpoint_dir, 'model_classic.pkl'))
    
    # 2. Сохраняем пайплайн (скейлер, медианы, one-hot-кодировщик)
    joblib.dump(pipeline, os.path.join(config.paths.checkpoint_dir, 'preprocessor_classic.pkl'))





    test_path = config.paths.test_csv
    test_df = pd.read_csv(test_path)
    test_df = data_engineering(test_df)
    

    X_test = pipeline.transform(test_df)
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()
    

    test_predictions = best_model.predict(X_test)
    

    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': test_predictions
    })
    submission.to_csv('submission.csv', index=False)
    print(f"Submission сохранён! Всего записей: {len(submission)}")


    if hasattr(model, 'state_dict'):
        torch.save(model.state_dict(), os.path.join(config.paths.checkpoint_dir, 'model_weights.pth'))
    else:
        joblib.dump(model, os.path.join(config.paths.checkpoint_dir, 'model_classic.pkl'))
    
    joblib.dump(pipeline, os.path.join(config.paths.checkpoint_dir, 'preprocessor_classic.pkl'))

    wandb.finish()
    print("Модель и препроцессор сохранены!")


def voting_train(config):
    """
    Обучает несколько моделей и делает предсказание большинством голосов.

    Args:
        config: конфигурация с paths.*, training.*
    """
    from scipy.stats import mode

    print(f"\n{'='*50}")
    print("Начало обучения: Voting (ансамбль моделей)")
    print(f"{'='*50}\n")

    wandb.init(project=config.logging.wandb_project_name)

    if not os.path.exists(config.paths.checkpoint_dir):
        os.makedirs(config.paths.checkpoint_dir)

    X_train, X_val, y_train, y_val, pipeline = get_data_loaders(config)

    X_train = pipeline.fit_transform(X_train)
    X_val = pipeline.transform(X_val)

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_val = X_val.toarray()


    model_types = ['Xgboost', 'CatBoost', 'LightGBM', 'randomForest']
    models = {}
    val_predictions = {}

    print(f"Обучаем {len(model_types)} моделей для голосования...\n")

    for model_type in model_types:
        print(f"--- Обучение: {model_type} ---")
        model = get_model(config, type=model_type, in_features=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"{model_type} — Accuracy: {acc:.4f}\n")

        models[model_type] = model
        val_predictions[model_type] = y_pred


    pred_matrix = np.array([val_predictions[m] for m in model_types])  # shape: (n_models, n_samples)
    voting_result, _ = mode(pred_matrix, axis=0)
    voting_result = voting_result.flatten()

    voting_acc = accuracy_score(y_val, voting_result)
    print(f"{'='*50}")
    print(f"Voting Результаты:")
    for m in model_types:
        print(f"  {m}: {accuracy_score(y_val, val_predictions[m]):.4f}")
    print(f"  Voting (большинство): {voting_acc:.4f}")
    print(f"{'='*50}\n")


    test_path = config.paths.test_csv
    test_df = pd.read_csv(test_path)
    test_df = data_engineering(test_df)

    X_test = pipeline.transform(test_df)
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    test_pred_matrix = np.array([models[m].predict(X_test) for m in model_types])
    test_voting, _ = mode(test_pred_matrix, axis=0)
    test_voting = test_voting.flatten()

    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': test_voting
    })
    submission.to_csv('submission.csv', index=False)
    print(f"Submission сохранён! Всего записей: {len(submission)}")

    for model_type, model in models.items():
        joblib.dump(model, os.path.join(config.paths.checkpoint_dir, f'model_{model_type}.pkl'))
    joblib.dump(pipeline, os.path.join(config.paths.checkpoint_dir, 'preprocessor_voting.pkl'))

    wandb.finish()
    print("Модели и препроцессор сохранены!")


def stacking_train(config):
    """
    Обучает базовые модели с K-fold, собирает out-of-fold предсказания,
    обучает мета-модель (LogisticRegression) и делает финальное предсказание.

    Args:
        config: конфигурация с paths.*, training.*
    """
    print(f"\n{'='*50}")
    print("Начало обучения: Stacking (мета-модель)")
    print(f"{'='*50}\n")

    wandb.init(project=config.logging.wandb_project_name)

    if not os.path.exists(config.paths.checkpoint_dir):
        os.makedirs(config.paths.checkpoint_dir)

    X_train, X_val, y_train, y_val, pipeline = get_data_loaders(config)

    X_train = pipeline.fit_transform(X_train)
    X_val = pipeline.transform(X_val)

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_val = X_val.toarray()

    # Базовые модели
    model_types = ['Xgboost', 'CatBoost', 'LightGBM', 'randomForest']
    n_models = len(model_types)
    n_splits = 2

    # Шаг 1: Out-of-fold предсказания
    print(f"Шаг 1: Сбор out-of-fold предсказаний ({n_splits} фолдов)...\n")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)

    oof_preds = np.zeros((len(X_train), n_models))
    test_preds = np.zeros((len(X_val), n_models))

    for i, model_type in enumerate(model_types):
        print(f"--- {model_type} ---")
        fold_preds = np.zeros(len(X_train))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            model = get_model(config, type=model_type, in_features=-1)
            model.fit(X_train[train_idx], y_train[train_idx])
            fold_preds[val_idx] = model.best_estimator_.predict(X_train[val_idx])

        oof_preds[:, i] = fold_preds

        # Обучаем на полном train для предсказания на val
        model = get_model(config, type=model_type, in_features=-1)
        model.fit(X_train, y_train)
        test_preds[:, i] = model.best_estimator_.predict(X_val)

        acc = accuracy_score(y_val, test_preds[:, i])
        print(f"  OOF Accuracy: {accuracy_score(y_train, fold_preds):.4f}")
        print(f"  Val Accuracy: {acc:.4f}\n")

    # Шаг 2: Обучение мета-модели
    print("Шаг 2: Обучение мета-модели (LogisticRegression)...\n")
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(oof_preds, y_train)

    # Оценка на val
    val_meta = meta_model.predict(test_preds)
    stacking_acc = accuracy_score(y_val, val_meta)

    print(f"{'='*50}")
    print(f"Stacking Результаты:")
    for i, m in enumerate(model_types):
        print(f"  {m}: {accuracy_score(y_val, test_preds[:, i]):.4f}")
    print(f"  Stacking (мета-модель): {stacking_acc:.4f}")
    print(f"{'='*50}\n")

    # Шаг 3: Финальное предсказание на test.csv
    test_path = config.paths.test_csv
    test_df = pd.read_csv(test_path)
    test_df = data_engineering(test_df)

    X_test = pipeline.transform(test_df)
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    # Предсказания базовых моделей на test
    test_matrix = np.zeros((len(X_test), n_models))
    for i, model_type in enumerate(model_types):
        model = get_model(config, type=model_type, in_features=-1)
        model.fit(X_train, y_train)
        test_matrix[:, i] = model.best_estimator_.predict(X_test)

    final_predictions = meta_model.predict(test_matrix)

    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': final_predictions
    })
    submission.to_csv('submission.csv', index=False)
    print(f"Submission сохранён! Всего записей: {len(submission)}")

    joblib.dump(meta_model, os.path.join(config.paths.checkpoint_dir, 'meta_model.pkl'))
    joblib.dump(pipeline, os.path.join(config.paths.checkpoint_dir, 'preprocessor_stacking.pkl'))

    wandb.finish()
    print("Мета-модель и препроцессор сохранены!")

