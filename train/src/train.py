import numpy as np
from othello.src.othello_agent.board import Board
from machine.src.models.othello_learn_class import OthelloLearn
from othello.src.othello_agent.board_calculator import BoardCalculator

othello_learn = OthelloLearn(lr=0.5, depth=5)
board = Board()
board_calculator = BoardCalculator()

x = board_calculator.diagonal_array(board.board)

y = board_calculator.corner_array(board.board)
print(y)
