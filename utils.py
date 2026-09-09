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



def train(model, train_loader, val_loader, config):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.05)

    best_val_loss = float('inf')

    for epoch in range(config.training.epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(config.device)
            y_batch = y_batch.to(config.device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(config.device)
                y_batch = y_batch.to(config.device).float().unsqueeze(1)
                
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
                # Метрика Accuracy
                probs = torch.sigmoid(preds)
                acc_preds = (probs > 0.5).float()
                correct += (acc_preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        
        scheduler.step(val_loss)
        
        

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config.paths.checkpoint_dir, 'best_model.pth'))

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]['lr']
        })

def Predict_kfold(config):
    test_path = "./data/test.csv"
    test_df = pd.read_csv(test_path)
    test_df = data_engineering(test_df)
    n_splits = config.data.kfold.n_splits

    all_fold_probs = []

    for fold in range(1, n_splits + 1):
        pipeline_path = os.path.join(config.paths.checkpoint_dir, f"preprocessor_fold_{fold}.pkl")
        model_path = os.path.join(config.paths.checkpoint_dir, f"model_fold_{fold}.pth")

        pipeline = joblib.load(pipeline_path)
        X_test = pipeline.transform(test_df)
        if hasattr(X_test, "toarray"):
            X_test = X_test.toarray()
        
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(config.device)

        model = ClassifyModel(in_features=config.in_features).to(config.device)
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        model.eval()

        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_fold_probs.append(probs)

        mean_probs = np.mean(all_fold_probs, axis=0)
        final_predictions = (mean_probs >= 0.5).astype(int)

        submission = pd.DataFrame({
            'PassengerId': test_df['PassengerId'],
            'Survived': final_predictions
        })
    submission.to_csv('submission.csv', index=False)
    print(f"Готово! Предсказания сохранены в submission.csv (всего записей: {len(submission)})")

def KFoldTraining(config):
    if not os.path.exists(config.paths.checkpoint_dir):
        os.makedirs(config.paths.checkpoint_dir)
    


   
    train_df = pd.read_csv(config.paths.train_csv)
    train_df = data_engineering(train_df)
    y_full = train_df['Survived'].values

    skf = StratifiedKFold(n_splits=config.data.kfold.n_splits, shuffle=True, random_state=config.seed)



    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_full)):
        print(f"\n============ Fold {fold + 1}/{config.data.kfold.n_splits} ============")
        
        wandb.init(
            project=config.logging.wandb_project_name,
            group="kfold-experiment",
            name=f"fold_{fold + 1}",
            reinit=True
        )

        train_fold_df = train_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_idx].reset_index(drop=True)

        

        pipeline = get_transforms(config)
        X_train = pipeline.fit_transform(train_fold_df)
        X_val = pipeline.transform(val_fold_df)

        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()
            X_val = X_val.toarray()

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(train_fold_df['Survived'].values, dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(val_fold_df['Survived'].values, dtype=torch.long))

        train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False)

        in_features = X_train.shape[1]
        model = get_model(config, in_features=in_features, type='classic').to(config.device)

        train(model, train_loader, val_loader, config)

        torch.save(model.state_dict(), os.path.join(config.paths.checkpoint_dir, f'model_fold_{fold + 1}.pth'))
        joblib.dump(pipeline, os.path.join(config.paths.checkpoint_dir, f'preprocessor_fold_{fold + 1}.pkl'))
        
        wandb.finish()

    
def data_engineering(df):
    df["FamilySize"] = df["SibSp"] + df["Parch"] 
    df['IsAlone'] = (df['FamilySize'] == 0).astype(int)

    return df

def get_data_loaders(config):
    pipeline = get_transforms(config)
    train_df = pd.read_csv(config.paths.train_csv)
    train_df = data_engineering(train_df)

    pipeline.fit(train_df)

    if hasattr(X_full, "toarray"):
        X_full = X_full.toarray()
    y_full = train_df['Survived'].values



    X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=config.data.kfold.val_size, random_state=config.seed)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)

    return train_loader, val_loader, pipeline

def get_model(config, type: str = 'classic', in_features: int = 9):
    if type == 'classic':
        return ClassifyModel(in_features=in_features)
    else:
        raise ValueError(f'Invalid model type: {type}')

def ClassicTraining(config):

    wandb.init(project=config.logging.wandb_project_name)

    if not os.path.exists(config.paths.checkpoint_dir):
        os.makedirs(config.paths.checkpoint_dir)
    
    train_loader, val_loader, pipeline = get_data_loaders(config)
    torch.cuda.empty_cache()

    model = get_model(config, type = 'classic', in_features = config.in_features)
    model = model.to(config.device)
    train(model, train_loader, val_loader, config)


    torch.save(model.state_dict(), os.path.join(config.paths.checkpoint_dir, 'model_weights.pth'))
    
    # 2. Сохраняем пайплайн (скейлер, медианы, one-hot-кодировщик)
    joblib.dump(pipeline, os.path.join(config.paths.checkpoint_dir, 'preprocessor.pkl'))

    wandb.finish()
    print("Модель и препроцессор сохранены!")