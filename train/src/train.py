import numpy as np
from othello.src.othello_agent.board import Board
from machine.src.models.othello_learn_class import OthelloLearn
from othello.src.othello_agent.board_calculator import BoardCalculator
from conf import config
import pickle

othello_learn = OthelloLearn(lr=10, depth=5)
board = Board()
board_calculator = BoardCalculator()

othello_learn.set_bond(16, 16, 2, 20, 4, 20, 4)
othello_learn.set_layers(16, 50, depth=3)

x = board_calculator.get_train_array([board.board])

target = np.array([[30]])

for i in range(100):
    othello_learn.train(x, target)

print(othello_learn.predict(x, train_flag=True))

