import pickle
import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    pass
from typing import Union

from src.conf import config
from src.machine.src.models.othello_learn_class import OthelloLearn
from src.othello.src.othello_agent.board_calculator import BoardCalculator


class OthelloTrainer:
    def __init__(self, load_leaner=None, batch_size=300):
        self.leaner: Union[OthelloLearn, None] = None
        self.save_learner_name = None
        if load_leaner is not None:
            self.save_learner_name = load_leaner
            f = open(config.TRAIN_MODEL_PATH.format(load_leaner), "rb")
            self.leaner = pickle.load(f)
            f.close()
        self.board_calculator = BoardCalculator()
        self.batch_size = batch_size

    def set_leaner(self, ol: OthelloLearn, save_name: str):
        self.save_learner_name = save_name
        self.leaner = ol

    def save_leaner(self):
        f = open(config.TRAIN_MODEL_PATH.format(self.save_learner_name), "wb")
        pickle.dump(self.leaner, f)
        f.close()

    def predict(self, board, train_flag=True):
        board = np.array(board)
        if board.ndim == 2:
            b = board.reshape((1, 8, 8))
        else:
            b = board.reshape((board.shape[0], 8, 8))
        b = self.board_calculator.get_train_array(b)
        return self.leaner.predict(b, train_flag=train_flag)

    def train(self, board_list, t, epoch=1):
        # 初期化しておく
        self.leaner.loss_list = []
        board_list = np.array(board_list)
        t = np.array(t)
        # 盤面をと評価値を反転させたものもトレーニングデータに入れておく
        board_list, t = self.board_calculator.concatenate_board_and_score(board_list, t)
        for e in range(epoch):
            index = np.random.permutation(np.arange(board_list.shape[0]))
            for i in range(0, board_list.shape[0], self.batch_size):
                data = board_list[index[i:i + self.batch_size]]
                data = self.board_calculator.get_train_array(data)
                target = t[index[i:i + self.batch_size]]
                target = target.reshape((target.size, 1))
                self.leaner.train(data, target)
                if i % 30 == 0:
                    self.save_leaner()
            self.save_leaner()

    def show_loss_graph(self):
        plt.plot(np.arange(len(self.leaner.loss_list)), self.leaner.loss_list)
        plt.show()
