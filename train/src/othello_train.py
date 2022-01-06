from othello.src.othello_agent.agent import NewOthelloAgent
from othello.src.othello_agent.board import Board
from machine.src.othello.othello_trainer import OthelloTrainer
import time
import numpy as np

trainer = OthelloTrainer(load_leaner="othello.pkl")
agent = NewOthelloAgent()
board = Board()
start = time.time()
for i in range(30):
    print(i)
    board.__init__()
    while np.where(board.board == 0)[0].size > 10:
        board.random_put()

    x, y = agent.read_all_for_train(board.board, board.turn, board.pss)
    print(len(x))
    trainer.train(x, y, epoch=1)
