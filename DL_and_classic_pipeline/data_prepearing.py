
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
    df["FamilySize"] = df["SibSp"] + df["Parch"] 
    df['IsAlone'] = (df['FamilySize'] == 0).astype(int)

    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    df['Embarked'] = df['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})

    return df

def get_data_loaders(config):
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
    
    X_train = pipeline.fit_transform(X_train_raw)
    X_val = pipeline.transform(X_val_raw)
    
    feature_names = pipeline.get_feature_names_out()
    print(pd.DataFrame(X_train, columns=feature_names).head())

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_val = X_val.toarray()

    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), 
        torch.tensor(y_val, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)

    return train_loader, val_loader, pipeline