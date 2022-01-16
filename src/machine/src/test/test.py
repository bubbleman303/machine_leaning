from src.othello.src.othello_agent.board import Board
from src.machine.src.othello.othello_trainer import OthelloTrainer

board = Board()
othello_trainer = OthelloTrainer()
pred = othello_trainer.predict(board.board)
print(pred)