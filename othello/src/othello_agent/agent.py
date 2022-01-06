import numpy as np
import copy
import json

from othello.src.othello_agent.board import Board
import random
from othello.src.conf import config


class NewOthelloAgent(Board):
    def __init__(self):
        super().__init__()
        self.all_read = 10
        self.blank_num = None
        self.my_turn = None
        self.corner_rate = 50
        self.minimum_rate = -10
        self.many_rate = 5
        self.nakawari_rate = 2
        self.x_rate = -30
        self.next_rate = -20
        self.one_of_three = 22
        self.two_of_three = 43
        self.monte_num = 10
        self.read_his = None
        self.read_depth_of_com = 3
        self.read_depth_of_agent = 3
        self.read_width = 3
        self.eval_corner_dict = None
        self.eval_edge_black_dict = None
        self.eval_edge_white_dict = None
        self.reverse_dict = None
        self.diagonal_pos_dict = None

        self.load_eval()
        self.alpha = 1
        self.gammma = 1
        self.depth_read_dict = {}

    def load_eval(self):
        with open(config.PARAM_DIR.format("reverse_dict.json"), "rt") as f:
            self.reverse_dict = json.load(f)
        with open(config.PARAM_DIR.format("diag_pos_dict.json"), "rt") as f:
            self.diagonal_pos_dict = json.load(f)

    def save_eval(self):
        with open(config.PARAM_DIR.format("eval_edge_black.json"), "wt") as f:
            json.dump(self.eval_edge_black_dict, f, indent=4)
        with open(config.PARAM_DIR.format("eval_edge_white.json"), "wt") as f:
            json.dump(self.eval_edge_white_dict, f, indent=4)

    def get_opp(self, turn):
        return self.white if turn == self.black else self.black

    @staticmethod
    def get_diagonal_index(pos, right=True):
        if right:
            t = abs(pos[0] - pos[1])
            if pos[0] < pos[1]:
                y = np.arange(8 - t)
                x = np.arange(t, 8)
            else:
                y = np.arange(t, 8)
                x = np.arange(8 - t)
            return y, x
        t = abs(pos[0] - (7 - pos[1]))
        if pos[0] < 7 - pos[1]:
            y = np.arange(8 - t)
            x = np.arange(8 - t)[::-1]
        else:
            y = np.arange(t, 8)
            x = np.arange(t, 8)[::-1]
        return y, x

    def is_available_of_agent(self, pos, board_copy, turn):
        dic = self.reverse_dict[str(turn)]
        s = self.make_string(board_copy[pos[0], :])
        temp = dic["8"]
        if s in temp:
            if str(pos[1]) in temp[s]:
                return True
        s = self.make_string(board_copy[:, pos[1]])
        if s in temp:
            if str(pos[0]) in temp[s]:
                return True

        y, x = self.get_diagonal_index(pos)
        s = self.make_string(board_copy[y, x])
        if len(s) > 2:
            if s in dic[str(len(s))]:
                if str(np.where(y == pos[0])[0][0]) in dic[str(len(s))][s]:
                    return True

        y, x = self.get_diagonal_index(pos, right=False)
        s = self.make_string(board_copy[y, x])
        if len(s) > 2:
            if s in dic[str(len(s))]:
                if str(np.where(y == pos[0])[0][0]) in dic[str(len(s))][s]:
                    return True
        return False

    def search_available_of_agent(self, board_copy, turn):
        if np.where(board_copy == self.blank)[0].size >= 10:
            dic = self.reverse_dict[str(turn)]
            pos_list = []
            eight_dict = dic["8"]
            for i in range(8):
                s = self.make_string(board_copy[i, :])
                if s in eight_dict:
                    pos_list += [[i, int(k)] for k in eight_dict[s]]
                s = self.make_string(board_copy[:, i])
                if s in eight_dict:
                    pos_list += [[int(k), i] for k in eight_dict[s]]
            for i in range(-5, 6):
                s = self.make_string(np.diag(board_copy, i))
                t = str(8 - abs(i))
                if s in dic[t]:
                    pos_list += [self.diagonal_pos_dict["normal"][str(i)][k] for k in dic[t][s]]
            board_copy = np.fliplr(board_copy)
            for i in range(-5, 6):
                s = self.make_string(np.diag(board_copy, i))
                t = str(8 - abs(i))
                if s in dic[t]:
                    pos_list += [self.diagonal_pos_dict["flip"][str(i)][k] for k in dic[t][s]]
            seen = []
            return [k for k in pos_list if k not in seen and not seen.append(k)]
        pos_list = np.where(board_copy == self.blank)
        return [[y, x] for y, x in zip(pos_list[0], pos_list[1]) if
                self.is_available_of_agent([y, x], board_copy, turn)]

    def put_stone_of_agent(self, pos, board, turn):
        board_copy = copy.deepcopy(board)
        dic = self.reverse_dict[str(turn)]
        # 横方向のリバース
        s = self.make_string(board[pos[0], :])
        if s in dic["8"]:
            s = dic["8"][s]
            if str(pos[1]) in s:
                s = s[str(pos[1])]
                board_copy[pos[0], :] = list(map(int, [k for k in s]))

        # 縦方向のリバース
        s = self.make_string(board[:, pos[1]])
        if s in dic["8"]:
            s = dic["8"][s]
            if str(pos[0]) in s:
                s = s[str(pos[0])]
                board_copy[:, pos[1]] = list(map(int, [k for k in s]))

        # 右上方向のリバース
        y, x = self.get_diagonal_index(pos)
        s = self.make_string(board[y, x])
        if len(s) > 2:
            if s in dic[str(len(s))]:
                s = dic[str(len(s))][s]
                ind = str(np.where(y == pos[0])[0][0])
                if ind in s:
                    s = s[ind]
                    board_copy[y, x] = list(map(int, [k for k in s]))
        # 左上方向のリバース
        y, x = self.get_diagonal_index(pos, right=False)
        s = self.make_string(board[y, x])
        if len(s) > 2:
            if s in dic[str(len(s))]:
                s = dic[str(len(s))][s]
                ind = str(np.where(y == pos[0])[0][0])
                if ind in s:
                    s = s[ind]
                    board_copy[y, x] = list(map(int, [k for k in s]))
        return board_copy

    def check_corner(self, pos, board):
        if pos not in [[0, 1], [1, 0], [0, 6], [1, 7], [6, 0], [7, 1], [7, 6], [6, 7], [1, 1], [1, 6], [6, 1], [6, 6],
                       [0, 1]]:
            return False
        if pos in [[0, 1], [1, 0], [1, 1]]:
            if board[0][0] == self.blank:
                return True
        elif pos in [[1, 7], [0, 6], [1, 6]]:
            if board[0][7] == self.blank:
                return True
        elif pos in [[7, 1], [6, 0], [6, 1]]:
            if board[7][0] == self.blank:
                return True
        elif pos in [[6, 6], [7, 6], [6, 7]]:
            if board[7][0] == self.blank:
                return True
        return False

    def monte_carlo(self, board, turn):
        max_win = -11
        best_pos = None
        for pos in self.search_available_of_agent(board, turn):
            win = 0
            if self.check_corner(pos, board):
                win -= 2
            new_board = self.put_stone_of_agent(pos, board, turn)
            now_turn = turn
            for i in range(self.monte_num):
                pss = 0
                this_board = copy.deepcopy(new_board)
                while True:
                    now_turn = self.turn_change_of_agent(now_turn)
                    pos_list = self.search_available_of_agent(this_board, now_turn)
                    if len(pos_list) == 0:
                        pss += 1
                        if pss == 2:
                            break
                        continue
                    pss = 0
                    this_board = self.put_stone_of_agent(random.choice(pos_list), this_board, now_turn)
                if self.get_winner_of_agent(this_board) == turn:
                    win += 1
            if max_win < win:
                max_win = win
                best_pos = pos
        print(f"勝率：{max_win / self.monte_num}")
        return best_pos

    def get_winner_of_agent(self, board):
        black_count = len(np.where(board == self.black)[0])
        white_count = len(np.where(board == self.white)[0])
        winner = 0
        if black_count >= white_count:
            winner = self.black
        elif white_count > black_count:
            winner = self.white
        return winner

    def read_all(self, board_copy, turn, pss):
        board_copy = copy.deepcopy(board_copy)
        score_dict = {}

        def read(new_board, new_turn, new_pss):
            available_list = self.search_available_of_agent(new_board, new_turn)
            if len(available_list) == 0:
                new_pss += 1
                if new_pss == 2:
                    stone_count = self.get_stone_count(new_board)
                    score_dict.update({self.get_flatten_board_str_with_turn(new_board, new_turn): stone_count})
                    return stone_count
                score = read(new_board, self.turn_change_of_agent(new_turn), new_pss)
                score_dict.update({self.get_flatten_board_str_with_turn(new_board, new_turn): score})
                return score
            score_list = []
            for pos in available_list:
                new_board_copy = copy.deepcopy(new_board)
                new_board_copy = self.put_stone_of_agent(pos, new_board_copy, new_turn)
                next_turn = self.turn_change_of_agent(new_turn)
                new_flatten_turn_board = self.get_flatten_board_str_with_turn(new_board_copy, next_turn)
                if new_flatten_turn_board in score_dict:
                    score_list.append(score_dict[new_flatten_turn_board])
                    continue
                score_list.append(read(new_board_copy, next_turn, 0))
            max_stone_count = self.get_max_of_turn(score_list, new_turn)
            score_dict.update({self.get_flatten_board_str_with_turn(new_board, new_turn): max_stone_count})

            return max_stone_count

        read(board_copy, turn, pss)
        return score_dict

    def get_max_pos_from_read_his(self, board, turn):
        ll = []
        pos_list = self.search_available_of_agent(board, turn)
        for pos in pos_list:
            board_copy = copy.deepcopy(board)
            board_copy = self.put_stone_of_agent(pos, board_copy, turn)
            board_copy = self.get_flatten_board_str_with_turn(board_copy, self.turn_change_of_agent(turn))
            ll.append(self.read_his[board_copy])
        max_score = self.get_max_of_turn(ll, turn)
        return pos_list[ll.index(max_score)], max_score

    def get_stone_count(self, board):
        black_count = len(np.where(board == self.black)[0])
        white_count = len(np.where(board == self.white)[0])
        return black_count - white_count

    @staticmethod
    def get_flatten_board_str_with_turn(board, turn):
        return ''.join([str(k) for k in np.append(board, turn).flatten()])

    @staticmethod
    def turn_change_of_agent(turn):
        turn = 1 if turn == 2 else 2
        return turn

    @staticmethod
    def get_max_of_turn(ll, turn):
        n = max(ll) if turn == 1 else min(ll)
        return n

    def random_pos_choice(self, board, turn):
        return random.choice(self.search_available_of_agent(board, turn))

    # モンテカルロ法を用いない新しい評価関数
    def put_able_num(self, board, turn):
        diff = len(self.search_available_of_agent(board, self.black)) - len(
            self.search_available_of_agent(board, self.white))
        return diff if turn == self.black else diff * -1

    @staticmethod
    def make_string(ll):
        return "".join([str(k) for k in ll])

    def get_eval_of_corner(self, ll):
        return self.eval_corner_dict[self.make_string(ll)]

    def eval_corner_shape(self, board, turn):
        score = 0
        board_copy = copy.deepcopy(board)
        for i in range(4):
            score += self.get_eval_of_corner(np.diag(board_copy)[:3])
            score += self.get_eval_of_corner(board_copy[0, :3])
            score += self.get_eval_of_corner(board_copy[:3, 0])
            if i == 3:
                break
            board_copy = np.rot90(board_copy)
        return score if turn == self.black else score * -1

    def evaluate(self, board, turn):
        return self.eval_corner_shape(board, turn) * 1.5 + self.put_able_num(board, turn)

    def read_and_get_pos(self, board, turn):

        def read(board_c, now_turn, n):
            if n == self.read_depth_of_com:
                return self.evaluate(board_c, turn)
            pos_list = self.search_available_of_agent(board_c, now_turn)
            if len(pos_list) == 0:
                now_turn = self.turn_change_of_agent(now_turn)
                return read(board_c, now_turn, n + 1)
            temp_list = []
            for pos in pos_list:
                board_copy = copy.deepcopy(board_c)
                new_board = self.put_stone_of_agent(pos, board_copy, now_turn)
                temp_list.append([new_board, pos, self.evaluate(new_board, turn)])
            if now_turn == turn:
                temp_list.sort(key=lambda x: x[2], reverse=True)
            else:
                temp_list.sort(key=lambda x: x[2])
            if n == 0:
                s = [(read(b[0], self.turn_change_of_agent(now_turn), n + 1), b[1]) for b in
                     temp_list[:min(self.read_width, len(temp_list))]]
                s.sort(key=lambda x: x[0], reverse=True)
                return s[0][1]
            return max(
                [read(b[0], self.turn_change_of_agent(now_turn), n + 1) for b in
                 temp_list[:min(self.read_width, len(temp_list))]])

        pos = read(board, turn, 0)
        if type(pos) != list:
            return None
        return pos

    def new_agent_put(self, board, turn, pss):
        blank_num = len(np.where(board == 0)[0])
        if blank_num <= self.all_read:
            import time
            start = time.time()
            if self.read_his is None:
                self.read_his = self.read_all(board, turn, pss)
            print(len(self.read_his))
            print(time.time() - start)
            pos, max_count = self.get_max_pos_from_read_his(board, turn)
            winner_str = "引き分け"
            if max_count > 0:
                winner_str = "黒の勝ち"
            elif max_count < 0:
                winner_str = "白の勝ち"
            print(f"{winner_str}:{max_count}")
            return pos
        # if random.random() < 0.2:
        #     return self.random_pos_choice(board, turn)
        #
        # return self.read_and_get_pos(board, turn)
        return self.random_pos_choice(board, turn)

    def train(self, board_list: list):
        """
        黒番をメインにした評価関数を作成する
        :param board_list: 局面と手番が格納されたリスト
        """
        b_list = list(reversed(board_list))
        # now_gamma = 1
        winner = self.get_winner_of_agent(b_list[0][0])
        his = [[] for k in range(4)]
        for i, board in enumerate(b_list[:min(35, len(b_list))]):
            board_copy = board[0]
            eval_dict = self.eval_edge_black_dict if board[1] == 1 else self.eval_edge_white_dict
            for j in range(4):
                corner = ''.join([str(k) for k in board_copy[0, :]])
                # 同じ局面では学習させない
                if corner in his[j]:
                    continue
                his[j].append(corner)
                t = 1 if winner == self.black else -1
                eval_dict[corner] += t
                eval_dict[corner[::-1]] += t
                # diff = count - eval_dict[corner]
                # if i <= self.all_read:
                #     eval_dict[corner] += diff * self.alpha
                #     eval_dict[corner[::-1]] += diff * self.alpha
                # else:
                #     now_gamma *= self.gammma
                #     eval_dict[corner] += now_gamma * self.alpha * diff
                #     eval_dict[corner[::-1]] += now_gamma * self.alpha * diff
                if j == 3:
                    break
                board_copy = np.rot90(board_copy)
        self.save_eval()

    def eval_put(self, board, turn, pss):
        blank_num = len(np.where(board == 0)[0])
        if blank_num <= self.all_read:
            if self.read_his is None:
                self.read_his = self.read_all(board, turn, pss)
            pos, max_count = self.get_max_pos_from_read_his(board, turn)
            winner_str = "引き分け"
            if max_count > 0:
                winner_str = "黒の勝ち"
            elif max_count < 0:
                winner_str = "白の勝ち"
            print(f"{winner_str}:{max_count}")
            return pos
        return self.read_eval_and_get_pos(board, turn)

    def read_eval_and_get_pos(self, board, turn):
        def read(board_c, now_turn, n):
            if n == self.read_depth_of_agent:
                return self.eval_edge(board_c, now_turn, turn)
            pos_list = self.search_available_of_agent(board_c, now_turn)
            if len(pos_list) == 0:
                now_turn = self.turn_change_of_agent(now_turn)
                return read(board_c, now_turn, n + 1)
            temp_list = []
            for pos in pos_list:
                board_copy = copy.deepcopy(board_c)
                new_board = self.put_stone_of_agent(pos, board_copy, now_turn)
                temp_list.append([new_board, pos, self.eval_edge(new_board, now_turn, turn)])
            if now_turn == turn:
                temp_list.sort(key=lambda x: x[2], reverse=True)
            else:
                temp_list.sort(key=lambda x: x[2])
            if n == 0:
                s = [(read(b[0], self.turn_change_of_agent(now_turn), n + 1), b[1]) for b in
                     temp_list[:min(self.read_width, len(temp_list))]]
                s.sort(key=lambda x: x[0], reverse=True)
                print(s[0][0])
                return s[0][1]
            return max(
                [read(b[0], self.turn_change_of_agent(now_turn), n + 1) for b in
                 temp_list[:min(self.read_width, len(temp_list))]])

        pos = read(board, turn, 0)
        if type(pos) != list:
            return None
        return pos

    def eval_edge(self, board, turn, my_turn):
        score = 0
        board = copy.deepcopy(board)
        eval_dict = self.eval_edge_black_dict if turn == self.black else self.eval_edge_white_dict
        for j in range(4):
            corner = ''.join(self.make_string(board[0, :]))
            score += eval_dict[corner]
            if j == 3:
                break
            board = np.rot90(board)
        score += self.put_able_num(board, my_turn) * 10
        return score if my_turn == self.black else score * -1

    def test_read(self, board, turn):
        if self.get_flatten_board_str_with_turn(board, turn) in self.depth_read_dict:
            return

        def read(board_c, now_turn, n):
            if n == self.read_depth_of_agent:
                return self.eval_edge(board_c, now_turn, turn)
            pos_list = self.search_available_of_agent(board_c, now_turn)
            if len(pos_list) == 0:
                now_turn = self.turn_change_of_agent(now_turn)
                return read(board_c, now_turn, n + 1)
            temp_list = []
            for pos in pos_list:
                board_copy = copy.deepcopy(board_c)
                new_board = self.put_stone_of_agent(pos, board_copy, now_turn)
                temp_list.append([new_board, pos, self.eval_edge(new_board, now_turn, turn)])
            if now_turn == turn:
                temp_list.sort(key=lambda x: x[2], reverse=True)
            else:
                temp_list.sort(key=lambda x: x[2])
            if n == 0:
                s = [(read(b[0], self.turn_change_of_agent(now_turn), n + 1), b[1]) for b in
                     temp_list[:min(self.read_width, len(temp_list))]]
                s.sort(key=lambda x: x[0], reverse=True)
                print(s[0][0])
                return s[0][1]
            return max(
                [read(b[0], self.turn_change_of_agent(now_turn), n + 1) for b in
                 temp_list[:min(self.read_width, len(temp_list))]])

        return read(board, turn, 0)

    def read_all_for_train(self, board_copy, turn, pss):
        board_copy = copy.deepcopy(board_copy)
        score_dict = {}
        x_list = []
        y_list = []

        def read(new_board, new_turn, new_pss):
            available_list = self.search_available_of_agent(new_board, new_turn)
            if len(available_list) == 0:
                new_pss += 1
                if new_pss == 2:
                    stone_count = self.get_stone_count(new_board)
                    x_list.append(new_board)
                    y_list.append(stone_count)
                    return stone_count
                score = read(new_board, self.turn_change_of_agent(new_turn), new_pss)
                x_list.append(new_board)
                y_list.append(score)
                return score
            score_list = []
            for pos in available_list:
                new_board_copy = copy.deepcopy(new_board)
                new_board_copy = self.put_stone_of_agent(pos, new_board_copy, new_turn)
                next_turn = self.turn_change_of_agent(new_turn)
                new_flatten_turn_board = self.get_flatten_board_str_with_turn(new_board_copy, next_turn)
                if new_flatten_turn_board in score_dict:
                    score_list.append(score_dict[new_flatten_turn_board])
                    continue
                score_list.append(read(new_board_copy, next_turn, 0))
            max_stone_count = self.get_max_of_turn(score_list, new_turn)
            score_dict.update({self.get_flatten_board_str_with_turn(new_board, new_turn): max_stone_count})
            x_list.append(new_board)
            y_list.append(max_stone_count)

            return max_stone_count

        read(board_copy, turn, pss)
        return np.array(x_list), np.array(y_list)
