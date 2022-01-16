import numpy as np
from src.othello_agent.board import Board

from src.othello.othello_trainer import OthelloTrainer
from src.models.othello_learn_class import OthelloLearn
import time

start = time.time()
ol = OthelloLearn(lr=0.7, depth=3, one_output_hidden_size=30)

ol.set_bond(16, 16, 2, 20, 4, 20, 4)
ol.set_layers(16, 100, depth=3)
board = Board()
trainer = OthelloTrainer()
trainer.set_leaner(ol, "othello3.pkl")
ll = []
for i in range(50):
    board.random_put()
    ll.append(np.copy(board.board))
target = np.random.randint(-64, 64, (len(ll), 1))
ll = np.array(ll)
print(target.shape)
for i in range(300):
    trainer.train(ll, target)
pre = trainer.predict(ll)
t = np.concatenate((pre.reshape(1, -1), target.reshape(1, -1))).T
print(t)
print(time.time() - start)
trainer.show_loss_graph()
