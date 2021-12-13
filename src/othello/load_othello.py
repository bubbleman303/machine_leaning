import numpy as np
import glob
import random
import os

from src.conf import config


def load_o():
    path_list = config.TRAIN_DATA_DIR.format("othello/values/*.npy")
    path_list = glob.glob(path_list)
    random.shuffle(path_list)
    for k in path_list:
        base = os.path.basename(k).replace("board_status", "label")
        target = config.TRAIN_DATA_DIR.format(f"othello/labels/{base}")
        target = np.load(target)
        target = np.identity(64)[target]
        yield np.load(k), target
