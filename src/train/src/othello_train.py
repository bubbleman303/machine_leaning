from src.othello_agent.agent import OthelloAgent
from src.othello_agent.board import Board
from src.othello.othello_trainer import OthelloTrainer
import time
import numpy as np

trainer = OthelloTrainer(load_leaner="othello.pkl")
agent = OthelloAgent()
board = Board()
trainer.save_leaner()
# start = time.time()
# for i in range(100):
#     print(i)
#     board.__init__()
#     while np.where(board.board == 0)[0].size > 10:
#         board.random_put()
#
#     x, y = agent.read_all_for_train(board.board, board.turn, board.pss)
#     trainer.train(x, y, epoch=5)
#     if i % 9 == 0:
#         trainer.show_loss_graph()
