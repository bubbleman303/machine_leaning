import os
import numpy as np
import xlwings as xw
import sys
import time
import pickle

sys.path.append("C:/Users/Keizaburo Takashiba/Desktop/machine_leaning")
from src.conf import config
from src.common import common_method as cm
from src.othello.src.othello_agent.board import Board
from src.othello.src.othello_agent.agent import OthelloAgent

board = Board()
wb = xw.Book(config.XL_PATH)

wb.sheets["reversi"].activate()


def load_cache() -> OthelloAgent:
    if not os.path.exists(config.CACHE_PATH):
        a = OthelloAgent()
        return a
    f = open(config.CACHE_PATH, "rb")
    a = pickle.load(f)
    f.close()
    return a


def save_cache(a):
    f = open(config.CACHE_PATH, "wb")
    pickle.dump(a, f)
    f.close()


def delete_cache():
    if os.path.exists(config.CACHE_PATH):
        os.remove(config.CACHE_PATH)


def write_board(reset_board=False, hi_light=False):
    highlight = np.zeros_like(board.board)
    if reset_board:
        cm.set_value(xw, config.XL_START_ROW, config.XL_START_COL, np.concatenate((highlight, highlight)))
        return
    pos_array = np.array(board.search_available()).T
    if hi_light and pos_array.size != 0:
        pos_array = (pos_array[0], pos_array[1])
        highlight[pos_array] = 1
    cm.set_value(xw, config.XL_START_ROW, config.XL_START_COL, np.concatenate((board.board, highlight)))
    board.get_stone_count()
    cm.set_value(xw, 12, 9, board.stone_count)


def read_board():
    return np.array(
        cm.get_value(xw, config.XL_START_ROW, config.XL_START_COL, config.XL_START_ROW + 7,
                     config.XL_START_COL + 7)).astype(np.int)


def get_turn(turn_str):
    if turn_str == "黒":
        return 1
    else:
        return 2


def write_stone_count(s=None):
    cm.set_value(xw, config.XL_DIS_STONE_COUNT_ROW, config.XL_DIS_STONE_COUNT_COL, s)


def write_real_stone_count(reset_value=False, agent: OthelloAgent = None):
    start_row, start_col = 6, 20
    if reset_value:
        cm.set_value(xw, start_row, start_col, [[None, None] for _ in range(20)])
        return
    if agent.read_his is None:
        return
    pos_list, score_list = agent.get_max_pos_from_read_his(board.board, board.turn, return_list=True)
    temp = [[None, None] for _ in range(20)]
    temp[:len(pos_list)] = [[f"{pos[0]} {pos[1]}", score] for pos, score in zip(pos_list, score_list)]
    cm.set_value(xw, start_row, start_col, temp)


def reset():
    write_board(reset_board=True)
    cm.set_value(xw, 12, 6, None)
    write_stone_count()
    delete_cache()
    cm.set_value(xw, 12, 9, None)
    write_real_stone_count(reset_value=True)


def start():
    reset()
    agent = OthelloAgent()
    save_cache(agent)
    turn = get_turn(sys.argv[2])
    if turn == 2:
        board.random_put()
    write_board(hi_light=True)
    cm.set_value(xw, 12, 6, None)
    sys.exit()


def random_board():
    reset()
    num = int(sys.argv[2])
    turn = get_turn(sys.argv[3])
    for i in range(num):
        board.random_put()
        write_board()
    if not board.turn == turn:
        board.random_put()
    write_board()
    cm.set_value(xw, 12, 6, None)
    sys.exit()


def put():
    origin = read_board()
    y, x, turn = sys.argv[2:]
    y = int(y)
    x = int(x)
    board.board = np.copy(origin)
    board.turn = get_turn(turn)
    if not board.is_available((y, x)):
        sys.exit()
    board.put_stone((y, x))
    write_board()
    time.sleep(0.3)
    agent = load_cache()
    while True:
        if len(board.search_available()) == 0:
            board.pss += 1
            if board.pss == 2:
                winner = "黒" if board.get_winner() == 1 else "白"
                cm.set_value(xw, 12, 6, winner + "の勝ち")
                break
            board.turn_change()
            continue
        board.pss = 0
        pos = agent.agent_put(board.board, board.turn, board.pss)
        if agent.max_count is not None:
            write_stone_count(f"{agent.winner_str},石差:{agent.max_count}")
        board.put_stone(pos)
        write_board(hi_light=True)
        if len(board.search_available()) != 0:
            board.pss = 0
            write_real_stone_count(agent=agent)
            break
        board.pss += 1
        board.turn_change()
    sys.exit()


meth = sys.argv[1]
if meth == "start":
    start()
if meth == "reset_board":
    reset()
    sys.exit()
if meth == "put":
    put()
if meth == "random":
    random_board()
