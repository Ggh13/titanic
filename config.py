from omegaconf import OmegaConf

config = {
    "files":{
        "local": True,
        "data": "./data",
        "train": "/train.csv",
        "test": "/test.csv"
    }
}

conf = OmegaConf.create(config)