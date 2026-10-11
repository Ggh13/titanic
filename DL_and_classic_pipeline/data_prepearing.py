
import random
import numpy as np
import torch
import os
import wandb


from omegaconf import DictConfig
from omegaconf import OmegaConf

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

import joblib
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold
from models import ClassifyModel

def set_seed(seed: int):
    '''Set a random seed for complete reproducibility.'''

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_transforms(config):
    """
    Создаёт sklearn Pipeline для предобработки данных.

    Args:
        config: конфигурация (не используется внутри, но нужна для единообразия)

    Returns:
        Pipeline: пайплайн с ColumnTransformer (числовые + категориальные признаки)
    """
    numeric_features  = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
    categorical_features = ['Pclass', 'Sex', 'Embarked', 'IsAlone']

    numeric_transformer = Pipeline(
        steps = [
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers = [
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    return full_pipeline


def data_engineering(df):
    """
    Создаёт новые признаки и кодирует категориальные.

    Args:
        df: DataFrame с исходными данными

    Returns:
        DataFrame: с добавленными FamilySize, IsAlone и закодированными Sex, Embarked
    """
    df["FamilySize"] = df["SibSp"] + df["Parch"] 
    df['IsAlone'] = (df['FamilySize'] == 0).astype(int)

    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    df['Embarked'] = df['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})

    return df

def get_data_loaders(config):
    """
    Загружает train.csv, применяет data_engineering и разделяет на train/val.

    Args:
        config: конфигурация с paths.train_csv, data.kfold.val_size, seed

    Returns:
        X_train_raw: сырые признаки train (без Survived)
        X_val_raw: сырые признаки val (без Survived)
        y_train: целевая переменная train
        y_val: целевая переменная val
        pipeline: необученный препроцессор
    """
    pipeline = get_transforms(config)
    df = pd.read_csv(config.paths.train_csv)
    df = data_engineering(df)


    train_df, val_df = train_test_split(
        df, 
        test_size=config.data.kfold.val_size, 
        random_state=config.seed,
        stratify=df['Survived']  
    )

    y_train = train_df['Survived'].values
    y_val = val_df['Survived'].values


    X_train_raw = train_df.drop(columns=['Survived'])
    X_val_raw = val_df.drop(columns=['Survived'])

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

        # (Опционально) Снять ограничение на длину текста в ячейках
    pd.set_option('display.max_colwidth', None)
    
    # Возвращаем сырые данные и необученный pipeline
    # Предобработка будет обучаться внутри GridSearchCV (без утечки)
    return X_train_raw, X_val_raw, y_train, y_val, pipeline