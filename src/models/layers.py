import numpy as np
import src.models.functions as fs


class MullLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class SigmoidLayer:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, d_out):
        return d_out * self.out * (1 - self.out)


class ReluLayer:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, d_out):
        d_out[self.mask] = 0
        dx = d_out
        return dx


class LeakyReluLayer:
    def __init__(self):
        self.mask = None
        self.alpha = 0.01

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] *= self.alpha
        return out

    def backward(self, d_out):
        d_out[self.mask] *= self.alpha
        dx = d_out
        return dx


class AffineLayer:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x):
        x_copy = x.copy()
        if x_copy.ndim == 1:
            x_copy = x_copy.reshape((1, x_copy.size))
        self.x = x_copy
        out = np.dot(x_copy, self.w) + self.b

        return out

    def backward(self, d_out):
        dx = np.dot(d_out, self.w.T)
        self.dw = np.dot(self.x.T, d_out)
        self.db = np.sum(d_out, axis=0)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = fs.softmax(x)
        self.loss = fs.cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, d_out=1):
        batch_size = self.t.shape[0]
        return (self.y - self.t) / batch_size * d_out


class MeanSquareLoss:
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None

    def forward(self, x, t):
        self.y = x
        self.t = t
        errors = (t - x) ** 2
        sums = np.sum(errors, axis=1)
        self.loss = np.mean(sums)
        return self.loss

    def backward(self, d_out=1):
        return -d_out * (self.t - self.y) / self.y.shape[0]


class Conv:
    def __init__(self, cn, fn, fil, stride=1, pad=0):
        """
        :param cn: 入力チャンネル数
        :param fn: 使うフィルター数
        :param fil: フィルターの大きさ
        :param stride: ストライド
        :param pad: パディングサイズ
        """
        self.cn = cn
        self.fn = fn
        self.fil_h, self.fil_w = fil, fil
        self.stride, self.pad = stride, pad
        # フィルター数×チャンネル数×フィルター高さ×フィルター幅
        self.w = np.random.normal(scale=1 / np.sqrt(cn * fil ** 2), size=(self.fn, self.cn, self.fil_h, self.fil_w))
        # 1×フィルター数
        self.b = np.zeros((1, self.fn))
        self.x_shape = None
        self.x_b = None
        self.x_h = None
        self.x_w = None
        self.y_h = None
        self.y_w = None
        self.x_col = None
        self.w_col = None
        self.y = None
        self.dw = None
        self.db = None

    def load_wb(self, w, b):
        self.w = w
        self.b = b

    def get_params(self):
        return {
            "fil": self.fil_h,
            "cn": self.cn,
            "fn": self.fn,
            "stride": self.stride,
            "pad": self.pad
        }

    def get_output_size(self, x_size, fil_size):
        """
        入力値、フィルターサイズ、ストライドとパディングサイズから出力画像の大きさを得る。
        :param x_size: xの大きさ(幅、高さ問わず）
        :param fil_size: フィルターの大きさ（幅、高さ問わず)
        :return: 出力幅の大きさ
        """
        return (x_size - fil_size + 2 * self.pad) // self.stride + 1

    def forward(self, x: np.ndarray):
        self.x_shape = x.shape
        # xのバッチ数、チャンネル数、高さ、幅
        self.x_b, self.cn, self.x_h, self.x_w = x.shape
        # 出力高さと幅の設定
        self.y_h = self.get_output_size(self.x_h, self.fil_h)
        self.y_w = self.get_output_size(self.x_w, self.fil_w)
        # (xのチャンネル数×フィルターの高さ×フィルター幅,バッチサイズ×出力サイズ×出力サイズ)n
        self.x_col = fs.im2col(x, self.fil_h, self.y_h, self.y_w, self.stride, self.pad)
        '''
        (フィルター数,入力チャンネル数×フィルター高さ×フィルター幅)
        '''
        self.w_col = self.w.reshape(self.fn, -1)

        '''
        np.dot(self.w_col,self.x_col)→
        (フィルター数,入力チャンネル数×フィルター高さ×フィルター幅)×(xのチャンネル数×フィルターの高さ×フィルター幅,バッチサイズ×出力サイズ×出力サイズ)
        ＝(フィルター数,バッチサイズ×出力サイズ×出力サイズ)
        .T→
        (バッチサイズ×出力サイズ×出力サイズ,フィルター数)
        +self.b→
        self.b:1×フィルター数
        つまり、各フィルターに対してブロードキャスト演算をしている。
        '''
        y = np.dot(self.w_col, self.x_col).T + self.b
        '''
        self.y:(バッチサイズ×出力サイズ×出力サイズ,フィルター数)
        y.reshape→
        (バッチサイズ,出力高さ,出力幅,フィルター数)
        .transpose→
        (0,3,1,2)
        (バッチサイズ,フィルター数,出力高さ,出力幅)
        フィルター数を二次元目に持ってきている。欲しい形状が得られた。        
        '''
        self.y = y.reshape(self.x_b, self.y_h, self.y_w, self.fn).transpose(0, 3, 1, 2)
        return self.y

    def backward(self, dy):
        """
        逆伝播
        :param dy:微分値、形状：(バッチサイズ,フィルター数,出力高さ,出力幅)
        """
        '''
        (バッチサイズ,フィルター数,出力高さ,出力幅)
        .transpose→
        (バッチサイズ,出力高さ,出力幅,フィルター数)
        .reshape→
        (バッチサイズ×出力高さ×出力幅,フィルター数)
        '''
        dy = dy.transpose(0, 2, 3, 1).reshape(self.x_b * self.y_h * self.y_w, self.fn)
        '''
        x_col:(xのチャンネル数×フィルターの高さ×フィルター幅,バッチサイズ×出力サイズ×出力サイズ)
        dy:(バッチサイズ×出力高さ×出力幅,フィルター数)
        →(xのチャンネル数×フィルターの高さ×フィルター幅,フィルター数)
        '''
        dw = np.dot(self.x_col, dy)
        '''
        .T→(フィルター数,xのチャンネル数×フィルターの高さ×フィルター幅)
        .reshape→(フィルター数、チャンネル数、フィルターの高さ、フィルター幅)
        '''
        self.dw = dw.T.reshape(self.fn, self.cn, self.fil_h, self.fil_w)
        self.db = np.sum(dy, axis=0)
        '''
        (バッチサイズ×出力高さ×出力幅,フィルター数)×(フィルター数,入力チャンネル数×フィルター高さ×フィルター幅)
        →(バッチサイズ×出力高さ×出力幅,入力チャンネル数×フィルター高さ×フィルター幅)
        '''
        dx_col = np.dot(dy, self.w_col)
        '''
        .T→
        (入力チャンネル数×フィルター高さ×フィルター幅,バッチサイズ×出力高さ×出力幅)
        '''
        return fs.col2im(dx_col.T, self.x_shape, self.fil_h, self.y_h, self.stride, self.pad)


class Pooling:
    def __init__(self, pool):
        self.pool = pool
        self.x_shape = None
        self.x_b = None
        self.cn = None
        self.x_h = None
        self.x_w = None
        self.y_h = None
        self.y_w = None
        self.y = None
        self.max_index = None

    def get_params(self):
        return {
            "pool": self.pool
        }

    def get_output_size(self, x):
        return x // self.pool

    def forward(self, x):
        self.x_shape = x.shape
        self.x_b, self.cn, self.x_h, self.x_w = x.shape
        self.y_h = self.get_output_size(self.x_h)
        self.y_w = self.get_output_size(self.x_w)

        x_col = fs.im2col(x, fil_size=self.pool, y_h=self.y_h, stride=self.pool).T.reshape(-1, self.pool * self.pool)
        y = np.max(x_col, axis=1)
        self.y = y.reshape((self.x_b, self.y_h, self.y_w, self.cn)).transpose(0, 3, 1, 2)
        self.max_index = np.argmax(x_col, axis=1)
        return self.y

    def backward(self, dy):
        dy = dy.transpose(0, 2, 3, 1)
        dx = np.zeros((self.pool ** 2, dy.size))
        dx[self.max_index.reshape(-1), np.arange(dy.size)] = dy.reshape(-1)
        dx = dx.reshape((self.pool, self.pool, self.x_b, self.y_h, self.y_w, self.cn))
        dx = dx.transpose(5, 0, 1, 2, 3, 4)
        dx = dx.reshape(self.cn * self.pool ** 2, self.x_b * self.y_h * self.y_w)
        return fs.col2im(dx, self.x_shape, self.pool, self.y_h, stride=self.pool)


class AffineForConvLayer:
    def __init__(self, w=None, b=None):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None
        self.original_x_shape = None

    def load_wb(self, w, b):
        self.w = w
        self.b = b

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(x, self.w) + self.b

        return out

    def backward(self, d_out):
        dx = np.dot(d_out, self.w.T)
        self.dw = np.dot(self.x.T, d_out)
        self.db = np.sum(d_out, axis=0)
        return dx.reshape(*self.original_x_shape)


class Dropout:

    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, d_out):
        return d_out * self.mask


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """

    def __init__(self, gamma=1, beta=0, momentum=0.9):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # Conv層の場合は4次元、全結合層の場合は2次元

        # テスト時に使用する平均と分散
        self.running_mean = None
        self.running_var = None

        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.d_gamma = None
        self.d_beta = None

    def get_params(self):
        return {
            "gamma": self.gamma,
            "beta": self.beta,
            "momentum": self.momentum,
        }

    def load_mean_var(self, running_mean, running_var):
        self.running_mean = running_mean
        self.running_var = running_var

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            # 縦方向の平均値
            '''
            出力は(バッチ数,チャンネル×出力縦×出力横)になっているので
            チャンネル数×出力縦×出力横に対して平均値をとっているという事になる。
            '''
            mu = x.mean(axis=0)
            # 平均を0にする
            xc = x - mu
            # 分散
            var = np.mean(xc ** 2, axis=0)
            # 標準偏差
            std = np.sqrt(var + 10e-7)
            # 正規化
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

        else:
            xc = x - self.running_mean
            xn = xc / (np.sqrt(self.running_var + 10e-7))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, d_out):
        if d_out.ndim != 2:
            N, C, H, W = d_out.shape
            d_out = d_out.reshape(N, -1)

        dx = self.__backward(d_out)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        d_beta = dout.sum(axis=0)
        d_gamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        d_std = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        d_var = 0.5 * d_std / self.std
        dxc += (2.0 / self.batch_size) * self.xc * d_var
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.d_gamma = d_gamma
        self.d_beta = d_beta

        return dx
