import numpy as np


def softmax(a):
    c = np.max(a, axis=-1, keepdims=True)
    exp_a = np.exp(a - c)
    sum_exp = np.sum(exp_a, axis=-1, keepdims=True)
    return exp_a / sum_exp


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / y.shape[0]


def im2col(x, fil_size, y_h, y_w=None, stride=1, pad=0):
    """
    バッチ×フィルターサイズや出力サイズを基にxを二次元配列に変換する
    :param x: バッチ×チャンネル数×高さ×幅の4次元配列
    :param fil_size: フィルターのサイズ
    :param y_h: 出力画像の高さ
    :param y_w: 出力画像の幅
    :param stride: ストライド数
    :param pad: パディング数
    :return: 二次元に整形された画像
    """
    '''
    バッチサイズ,チャンネル数,入力画像縦幅,入力画像横幅
    フィルターサイズ縦幅横幅
    '''
    x_b, x_c, x_h, x_w = x.shape
    fil_h, fil_w = fil_size, fil_size
    # 出力画像が正方形の場合に幅を省略してもいいように
    if y_w is None:
        y_w = y_h
    x_pad = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad), ], "constant")
    '''
    (フィルター縦幅×フィルター横幅,バッチサイズ,チャンネル数,出力縦幅,出力横幅)
    '''
    col = np.zeros((fil_h * fil_w, x_b, x_c, y_h, y_w))
    idx = -1

    '''
    xの形状:
    (バッチサイズ,チャンネル数,入力画像縦幅,入力画像横幅)
    '''
    for h in range(fil_h):
        h2 = h + y_h * stride
        for w in range(fil_w):
            idx += 1
            w2 = w + y_w * stride
            col[idx, :, :, :, :] = x_pad[:, :, h:h2:stride, w:w2:stride]
    col = col.transpose((2, 0, 1, 3, 4)).reshape(x_c * fil_h * fil_w, x_b * y_h * y_w)
    return col


def col2im(dx_col, x_shape, fil_size, y_size, stride=1, pad=0):
    x_b, x_c, x_h, x_w = x_shape
    fil_h, fil_w = fil_size, fil_size
    y_h, y_w = y_size, y_size
    idx = -1

    dx_col = dx_col.reshape(x_c, fil_h * fil_w, x_b, y_h, y_w).transpose(1, 2, 0, 3, 4)
    dx = np.zeros((x_b, x_c, x_h + 2 * pad + stride - 1, x_w + 2 * pad + stride - 1))

    for h in range(fil_h):
        h2 = h + y_h * stride
        for w in range(fil_w):
            idx += 1
            w2 = w + y_w * stride
            dx[:, :, h:h2:stride, w:w2:stride] += dx_col[idx, :, :, :, :]

    return dx[:, :, pad:x_h + pad, pad:x_w + pad]


def weight_init(i, o):
    return np.random.normal(scale=1 / np.sqrt(i), size=(i, o))
