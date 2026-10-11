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

from data_prepearing import set_seed, get_transforms, data_engineering, get_data_loaders


def train(model, train_loader, val_loader, config):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.05)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    early_stopping_patience = 30
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

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]['lr']
        })

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(config.paths.checkpoint_dir, 'best_model.pth'))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping на эпохе {epoch + 1} (val_loss не улучшалась {early_stopping_patience} эпох)")
                break

    return val_acc

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

        model = ClassifyModel(in_features=X_test.shape[1]).to(config.device)
        
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
    print(f"Готово! Предсказания сохранены в submission.csv (всего записий: {len(submission)})")

def KFoldTraining(config):
    print(f"\n{'='*50}")
    print(f"Начало обучения: KFoldTraining (DL, {config.data.kfold.n_splits} фолдов)")
    print(f"{'='*50}\n")

    if not os.path.exists(config.paths.checkpoint_dir):
        os.makedirs(config.paths.checkpoint_dir)
    


   
    train_df = pd.read_csv(config.paths.train_csv)
    train_df = data_engineering(train_df)
    y_full = train_df['Survived'].values

    skf = StratifiedKFold(n_splits=config.data.kfold.n_splits, shuffle=True, random_state=config.seed)


 
    test_path = "./data/test.csv"
    test_df = pd.read_csv(test_path)
    test_df = data_engineering(test_df)
    all_fold_probs = []
    fold_accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_full)):
        set_seed(config.seed + fold)
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
        print("У модели столько слоев:", model.hidden_dim)
        best_val_acc = train(model, train_loader, val_loader, config)
        fold_accuracies.append(best_val_acc)
        print(f"Fold {fold + 1} — Accuracy: {best_val_acc:.4f}")

        torch.save(model.state_dict(), os.path.join(config.paths.checkpoint_dir, f'model_fold_{fold + 1}.pth'))
        joblib.dump(pipeline, os.path.join(config.paths.checkpoint_dir, f'preprocessor_fold_{fold + 1}.pkl'))
        



        X_test = pipeline.transform(test_df)
        if hasattr(X_test, "toarray"):
            X_test = X_test.toarray()
        
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(config.device)
        
        model.eval()
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_fold_probs.append(probs)

        wandb.finish()

    # Итоговые метрики K-fold
    mean_probs = np.mean(all_fold_probs, axis=0)
    final_predictions = (mean_probs >= 0.5).astype(int)

    print(f"\n{'='*50}")
    print(f"K-fold Cross-Validation Результаты ({config.data.kfold.n_splits} фолдов):")
    print(f"  Mean Accuracy:  {np.mean(fold_accuracies):.4f}")
    print(f"  Std Accuracy:   {np.std(fold_accuracies):.4f}")
    print(f"  Min Accuracy:   {np.min(fold_accuracies):.4f}")
    print(f"  Max Accuracy:   {np.max(fold_accuracies):.4f}")
    print(f"{'='*50}\n")

    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': final_predictions
    })
    submission.to_csv('submission.csv', index=False)
    print(f"Submission сохранён! Всего записей: {len(submission)}")
    
def get_model(config, type: str = 'classic', in_features: int = -1):
    if type == 'classic':
        return ClassifyModel(in_features=in_features)
    else:
        raise ValueError(f'Invalid model type: {type}')

def ClassicTraining(config):
    print(f"\n{'='*50}")
    print("Начало обучения: ClassicTraining (DL, 1 модель)")
    print(f"{'='*50}\n")

    wandb.init(project=config.logging.wandb_project_name)

    if not os.path.exists(config.paths.checkpoint_dir):
        os.makedirs(config.paths.checkpoint_dir)

    # Получаем сырые данные и необученный препроцессор
    X_train_raw, X_val_raw, y_train, y_val, pipeline = get_data_loaders(config)
    torch.cuda.empty_cache()

    # Обучаем препроцессор только на train (без утечки из val)
    X_train = pipeline.fit_transform(X_train_raw)
    X_val = pipeline.transform(X_val_raw)

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

    model = get_model(config, type='classic', in_features=X_train.shape[1])
    model = model.to(config.device)
    train(model, train_loader, val_loader, config)

    torch.save(model.state_dict(), os.path.join(config.paths.checkpoint_dir, 'model_weights.pth'))
    joblib.dump(pipeline, os.path.join(config.paths.checkpoint_dir, 'preprocessor.pkl'))

    # Предсказание на test.csv
    test_path = config.paths.test_csv
    test_df = pd.read_csv(test_path)
    test_df = data_engineering(test_df)

    X_test = pipeline.transform(test_df)
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(config.device)

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        test_predictions = (probs >= 0.5).astype(int)

    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': test_predictions
    })
    submission.to_csv('submission.csv', index=False)
    print(f"Submission сохранён! Всего записей: {len(submission)}")

    wandb.finish()
    print("Модель и препроцессор сохранены!")