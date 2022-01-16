import numpy as np
from src.othello.src.othello_agent.board import Board

ll = []
board = Board()
for i in range(5):
    board.random_put()
    ll.append(np.copy(board.board))

ll = np.array(ll)
print(ll)
t = np.zeros_like(ll)
t[np.where(ll == 1)] = 2
t[np.where(ll == 2)] = 1
print(t)
s = np.concatenate((ll, t))

print(s.shape)

t = np.random.random((5, 1))
t2 = t * -1
print(t)
print(t2)
t = np.concatenate((t,t2))
print(t.shape)
print(t)