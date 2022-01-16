from src.othello_agent.board import Board
from src.othello_agent.agent import OthelloAgent

from src.othello.othello_trainer import OthelloTrainer

agent = OthelloAgent()
board = Board()
trainer = OthelloTrainer(load_leaner="othello2.pkl")
learner = trainer.leaner
for i in range(56):
    board.random_put()

his = agent.read_all(board.board, board.turn, board.pss)
print(board.board)
x = trainer.predict(board.board)
print(x)
score = his[agent.get_flatten_board_str_with_turn(board.board, board.turn)]
print(score)
