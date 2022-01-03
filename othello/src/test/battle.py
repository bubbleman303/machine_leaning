from src.othello_agent.board import Board
from src.othello_agent.agent import NewOthelloAgent

import time

board = Board()
agent = NewOthelloAgent()
start = time.time()
while True:
    if board.end_check():
        break
    if board.turn == 1:
        board.random_put()
    else:
        board.put_stone(agent.monte_carlo(board.board,board.turn))

    print(board.board)


