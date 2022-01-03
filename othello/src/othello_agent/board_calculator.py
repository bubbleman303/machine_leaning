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
