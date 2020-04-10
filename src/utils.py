import cv2
import numpy as np
import scipy.ndimage
import seaborn as sns
from keras.utils import np_utils
from matplotlib import pyplot as plt

import math


def preprocess(_data, _label):
    _processed_data = _data.astype('float32') / 255.0
    _onehot_label = np_utils.to_categorical(_label)

    return _processed_data, _onehot_label


def showImage(*args):
    for _index, _arg in enumerate(args):
        cv2.imshow("img {}".format(_index), _arg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def showImages(**kwargs):
    for _key in kwargs:
        cv2.imshow("{}".format(_key), kwargs[_key])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def showSingleColor(r, b, g):
    height, width = 300, 300
    img = np.zeros((height, width, 3), np.uint8)
    for h in range(height):
        for w in range(width):
            img[h, w] = (b, g, r)
    name = "(b, g, r) = (%d, %d, %d)" % (b, g, r)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def info(_info):
    def decorator(_func):
        def parameters(*args, **kwargs):
            print("[info] {}".format(_info))
            exec_func = _func(*args, **kwargs)
            return exec_func
        return parameters
    return decorator


def splitChannel(_img):
    if _img.ndim == 2:
        return _img
    else:
        _bgr = [_img]

        for i in range(3):
            _temp = _img.copy()
            _temp[:, :, (i + 1) % 3] = 0
            _temp[:, :, (i + 2) % 3] = 0

            _bgr.append(_temp)

        return _bgr


def biBubic(_x):
    _x = abs(_x)
    if _x <= 1:
        return 1 - 2 * (_x**2) + (_x**3)
    elif _x < 2:
        return 4 - 8 * _x + 5 * (_x**2) - (_x**3)
    else:
        return 0


@info("biBubicInterpolation 內容似乎有瑕疵，需校正，請改用 biBubicInterpolation2(_img, _scale, _prefilter=True)")
def biBubicInterpolation(_img, _height_scale, _width_scale):
    # print("這個雙三次插值 (Bicubic interpolation)的內容似乎有瑕疵，需校正")
    if _img.ndim == 2:
        _height, _width = _img.shape
    else:
        _height, _width, _ = _img.shape

    _dst_height = int(_height * _height_scale)
    _dst_width = int(_width * _width_scale)
    _dst = np.zeros((_dst_height, _dst_width, 3), dtype=np.uint8)

    for _h in range(_dst_height):
        for _w in range(_dst_width):
            _x = _h * (_height / _dst_height)
            _y = _w * (_width / _dst_width)

            _x_index = math.floor(_x)
            _y_index = math.floor(_y)

            _u = _x - _x_index
            _v = _y - _y_index

            _temp = 0
            for _h_prime in [-1, 0, 1]:
                for _w_prime in [-1, 0, 1]:
                    if (_x_index + _h_prime < 0 or _y_index + _w_prime < 0 or
                            _x_index + _h_prime >= _height or _y_index + _w_prime >= _width):
                        continue
                    _temp += (_img[_x_index + _h_prime, _y_index + _w_prime] *
                              biBubic(_h_prime - _u) *
                              biBubic(_w_prime - _v))

            _dst[_h, _w] = np.clip(_temp, 0, 255)

    return _dst


def biBubicInterpolation2(_img, _scale, _prefilter=True):
    if _img.ndim == 2:
        _dst = scipy.ndimage.interpolation.zoom(_img, _scale, prefilter=_prefilter)

    else:
        b, g, r = cv2.split(_img)
        b = scipy.ndimage.interpolation.zoom(b, _scale, prefilter=_prefilter)
        g = scipy.ndimage.interpolation.zoom(g, _scale, prefilter=_prefilter)
        r = scipy.ndimage.interpolation.zoom(r, _scale, prefilter=_prefilter)

        _dst = cv2.merge([b, g, r])

    return _dst


# 最大公因數
def gcd(_a, _b):
    # https://www.geeksforgeeks.org/gcd-in-python/
    while _b > 0:
        _a, _b = _b, _a % _b

    return _a


# 最小公倍數
def lcm(_a, _b):
    # http://drweb.nksh.tp.edu.tw/student/lesson/G005/
    return _a * _b // gcd(_a, _b)


def plotImage(image, _size_inches=2):
    fig = plt.gcf()
    fig.set_size_inches(_size_inches, _size_inches)
    plt.imshow(image, cmap='binary')
    plt.show()


def colorfulDataFrame(df, cmap=plt.cm.Blues):
    _df = df.copy()
#    for col in range(len(df)):
#        _sum = df.iloc[:, col].sum()
#        df.iloc[:, col] /= _sum
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    #         繪圖數據   填充色       方塊的間隔     顯示數值
    sns.heatmap(_df, cmap=cmap, linewidths=0.1, annot=True)
    plt.show()


# region 降低解析度
def resizeTest(_width_scale, _height_scale):
    # https://jennaweng0621.pixnet.net/blog/post/403862273-%5Bpython-%2B-
    # opencv%5D-%E8%AA%BF%E6%95%B4%E5%BD%B1%E5%83%8F%E5%A4%A7%E5%B0%8F%28resize%29
    _img = cv2.imread("../../OpenEyes/data/splice4.png")
    _rows, _cols, _ = _img.shape

    # rows:1440, cols:1080
    print("rows:{}, cols:{}".format(_rows, _cols))

    _resize_rows = int(_rows * _height_scale)
    _resize_cols = int(_cols * _width_scale)
    INTER_NEAREST = cv2.resize(_img, (_resize_cols, _resize_rows), interpolation=cv2.INTER_NEAREST)
    INTER_LINEAR = cv2.resize(_img, (_resize_cols, _resize_rows), interpolation=cv2.INTER_LINEAR)
    INTER_AREA = cv2.resize(_img, (_resize_cols, _resize_rows), interpolation=cv2.INTER_AREA)
    INTER_CUBIC = cv2.resize(_img, (_resize_cols, _resize_rows), interpolation=cv2.INTER_CUBIC)
    INTER_LANCZOS4 = cv2.resize(_img, (_resize_cols, _resize_rows), interpolation=cv2.INTER_LANCZOS4)

    print("INTER_LANCZOS4.shape:{}".format(INTER_LANCZOS4.shape))

    showImage(_img, INTER_NEAREST, INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER_LANCZOS4)


def seamCarving1():
    # https://scikit-image.org/docs/0.14.x/auto_examples/transform/plot_seam_carving.html
    # 載入輸入圖像，轉化為灰度圖
    _img = cv2.imread("../../OpenEyes/data/sk_image1.png", cv2.IMREAD_COLOR)
    _gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)

    # 計算Sobel梯度場表示（能量圖）
    _mag = filters.sobel(_gray.astype("float"))

    resized = transform.resize(_img, (_img.shape[0], _img.shape[1] - 200), mode='reflect')

    showImage(_img, _mag, resized)


def pyrDown(_width_scale=0.5, _height_scale=0.5):
    # 高斯金字塔
    # https://blog.csdn.net/on2way/article/details/46867939
    _img = cv2.imread("../../OpenEyes/data/sk_image1.png", cv2.IMREAD_GRAYSCALE)
    _rows, _cols = _img.shape
    print("rows:{}, cols:{}".format(_rows, _cols))
    _new_rows = int(_rows * _height_scale)
    _new_cols = int(_cols * _width_scale)
    _down_img = cv2.pyrDown(_img, dstsize=(_new_cols, _new_rows))
    print("_down_img.sahpe:{}".format(_down_img.shape))

    _up_img = cv2.pyrUp(_down_img, dstsize=(_cols, _rows))
    print("_up_img.sahpe:{}".format(_up_img.shape))

    showImage(_img, _down_img, _up_img)


def pyrDown2():
    # https://blog.csdn.net/on2way/article/details/46867939
    # 拉普拉斯金字塔的圖像看起來就像是邊界圖，經常被用在圖像壓縮中。
    _img = cv2.imread("../../OpenEyes/data/pyrDown1.png", cv2.IMREAD_GRAYSCALE)

    _down_img = cv2.pyrDown(_img)  # 高斯金字塔
    print("_down_img.sahpe:{}".format(_down_img.shape))

    _down_down_img = cv2.pyrDown(_down_img)
    print("_down_down_img.sahpe:{}".format(_down_down_img.shape))
    _up_down_img = cv2.pyrUp(_down_down_img)
    print("_up_down_img.sahpe:{}".format(_up_down_img.shape))
    _laplace = _down_img - _up_down_img
    print("_laplace.sahpe:{}".format(_laplace.shape))

    showImage(_img, _down_img, _laplace)
# endregion


if __name__ == "__main__":
    showSingleColor(20, 40, 80)
    showSingleColor(40, 80, 160)
    showSingleColor(60, 120, 240)
