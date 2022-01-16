import numpy as np


class BoardCalculator:
    def __init__(self):
        self.edge_2x_origin = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        self.corner_origin = np.array(
            [
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]
        )

    @staticmethod
    def get_one_hot(np_arr):
        black_like = np.zeros_like(np_arr)
        black_like[np.where(np_arr == 1)] = 1
        white_like = np.zeros_like(np_arr)
        white_like[np.where(np_arr == 2)] = 1
        return np.array([black_like, white_like]).reshape(-1)

    def diagonal_array(self, board):
        def get_d(b):
            return np.diag(b)

        return np.array([self.get_one_hot(get_d(board)), self.get_one_hot(get_d(np.rot90(board)))])

    def edge_2x_array(self, board):
        def get_e(b):
            return b[np.where(self.edge_2x_origin == 1)]

        b_copy = np.copy(board)
        temp = []
        for i in range(4):
            temp.append(self.get_one_hot(get_e(b_copy)))
            b_copy = np.rot90(b_copy)
        return np.array(temp)

    def corner_array(self, board):
        def get_c(b):
            return b[np.where(self.corner_origin == 1)]

        b_copy = np.copy(board)
        temp = []
        for i in range(4):
            temp.append(self.get_one_hot(get_c(b_copy)))
            b_copy = np.rot90(b_copy)
        return np.array(temp)

    def get_train_array(self, board_list):
        return [np.array([self.diagonal_array(board) for board in board_list]),
                np.array([self.edge_2x_array(board) for board in board_list]),
                np.array([self.corner_array(board) for board in board_list])
                ]

    @staticmethod
    def inverse_board(board_list):
        like = np.zeros_like(board_list)
        like[np.where(board_list == 1)] = 2
        like[np.where(board_list == 2)] = 1
        return like

    @staticmethod
    def inverse_score(target_list):
        return target_list * -1

    def concatenate_board_and_score(self, board_list, target_list):
        inv_board = self.inverse_board(board_list)
        inv_target = self.inverse_score(target_list)
        return np.concatenate((board_list, inv_board)), np.concatenate((target_list, inv_target))
