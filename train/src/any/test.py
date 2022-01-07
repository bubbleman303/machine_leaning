from othello.src.othello_agent.board import Board
from othello.src.othello_agent.agent import NewOthelloAgent

from machine.src.othello.othello_trainer import OthelloTrainer
import numpy as np

agent = NewOthelloAgent()
board = Board()
trainer = OthelloTrainer(load_leaner="othello.pkl")
learner = trainer.leaner
for i in range(56):
    board.random_put()

his = agent.read_all(board.board, board.turn, board.pss)
print(board.board)
x = trainer.predict(board.board)
print(x)
score = his[agent.get_flatten_board_str_with_turn(board.board,board.turn)]
print(score)