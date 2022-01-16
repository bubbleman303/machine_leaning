BASE_PATH = __file__.replace("\\", "/").replace("src/conf/config.py", "{}")
TRAIN_MODEL_PATH = BASE_PATH.format("train_models/{}")
XL_PATH = BASE_PATH.format("xls/othello.xlsm")
CACHE_PATH = BASE_PATH.format("cache/cache.pkl")

XL_START_ROW = 55
XL_START_COL = 1
XL_DIS_STONE_COUNT_ROW = 3
XL_DIS_STONE_COUNT_COL = 21
