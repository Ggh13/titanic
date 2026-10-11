from omegaconf import OmegaConf


config = OmegaConf.create({
    "seed": 42,
    "mode": "voting",
    "type_classic": "voting",
    "device": "cuda",
    "training": {
        "debug": False,
        "number_of_debug_samples": 1000,
        "lr": 1e-3,
        "epochs": 50,
        "batch_size": 32,
    },
    "model": {
        "hidden_dim": 512,
        "num_layers": 3,
        "dropout": 0.15,
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