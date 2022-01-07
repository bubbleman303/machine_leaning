from othello.src.othello_agent.board import Board
from machine.src.othello.othello_trainer import OthelloTrainer
import numpy as np

board = Board()
trainer = OthelloTrainer(load_leaner="othello.pkl")
learner = trainer.leaner
for i in range(52):

    board.random_put()

print(board.board)
print(np.where(board.board == 1)[0].size - np.where(board.board == 2)[0].size)
x = trainer.predict(board.board)
print(x)
