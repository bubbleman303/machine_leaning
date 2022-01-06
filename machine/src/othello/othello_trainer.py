import pickle
import numpy as np
from conf import config
from machine.src.models.othello_learn_class import OthelloLearn
from othello.src.othello_agent.board_calculator import BoardCalculator


class OthelloTrainer:
    def __init__(self, load_leaner=None, batch_size=300):
        self.leaner = None
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

    def predict(self, board):
        b = board.reshape((1, 8, 8))
        b = self.board_calculator.get_train_array(b)
        return self.leaner.predict(b)

    def train(self, board_list, t, epoch=1):
        for e in range(epoch):
            print(epoch)
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
