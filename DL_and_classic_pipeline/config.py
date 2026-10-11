from omegaconf import OmegaConf


config = OmegaConf.create({
    "seed": 42,
    "mode": "K_fold",
    "type_classic": "CatBoost",
    "device": "cuda",
    "training": {
        "debug": False,
        "number_of_debug_samples": 1000,
        "lr": 1e-3,
        "epochs": 70,
        "batch_size": 32,
    },
    "data": {
        "kfold": {
            "n_splits": 5,        
            "val_size": 0.2,
        }
    },
    "paths": {
        "checkpoint_dir": "./checkpoints",
        "train_csv": "./data/train.csv",
        "test_csv": "./data/test.csv",
    },
    "logging": {
        "wandb_project_name": "titanic-classification"
    },
    "files": {
    "data": "./data/",     
    "train": "train.csv"  
},
})