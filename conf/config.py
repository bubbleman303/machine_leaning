BASE_PATH = __file__.replace("\\", "/").replace("conf/config.py", "{}")
TRAIN_MODEL_PATH = BASE_PATH.format("train_models/{}")
