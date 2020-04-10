import math

import cv2
import numpy as np
from skimage import draw
from skimage import transform
from skimage import filters

from src.utils import showImage


# region https://www.cnblogs.com/super-jjboom/p/9993431.html
def nnInterpolation(_img, _height_scale, _width_scale):
    _height, _width, _ = _img.shape
    _dst_height = _height * _height_scale
    _dst_width = _width * _width_scale
    _dst = np.zeros((_dst_height, _dst_width, 3), dtype=np.uint8)

    for h in range(_dst_height):
        for w in range(_dst_width):
            _x = round((h + 1) * (_height / _dst_height))
            _y = round((w + 1) * (_width / _dst_width))

            _dst[h, w] = _img[_x - 1, _y - 1]

    return _dst


def biLinearInterpolation(_img, _height_scale, _width_scale):
    _height, _width, _ = _img.shape
    _pad = np.pad(_img, ((0, 1), (0, 1), (0, 0)), 'constant')

    _dst_height = _height * _height_scale
    _dst_width = _width * _width_scale
    _dst = np.zeros((_dst_height, _dst_width, 3), dtype=np.uint8)

    for _h in range(_dst_height):
        for _w in range(_dst_width):
            _x = _h * (_height / _dst_height)
            _y = _w * (_width / _dst_width)

            _x_index = math.floor(_x)
            _y_index = math.floor(_y)

            _u = _x - _x_index
            _v = _y - _y_index

            _dst[_h, _w] = ((1-_u) * (1-_v) * _pad[_x_index, _y_index] +
                            _u * (1-_v) * _pad[_x_index + 1, _y_index] +
                            (1-_u) * _v * _pad[_x_index, _y_index + 1] +
                            _u * _v * _pad[_x_index + 1, _y_index + 1])

    return _dst


def biBubic(_x):
    _x = abs(_x)
    if _x <= 1:
        return 1 - 2 * (_x**2) + (_x**3)
    elif _x < 2:
        return 4 - 8 * _x + 5 * (_x**2) - (_x**3)
    else:
        return 0


def biBubicInterpolation(_img, _height_scale, _width_scale):
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


def srTest1(_output=False):
    image = cv2.imread("../../OpenEyes/data/seam_carving1.jpg")

    image1 = nnInterpolation(image, 2, 2)
    image1 = np.uint8(image1)

    image2 = biLinearInterpolation(image, 2, 2)
    image2 = np.uint8(image2)

    image3 = biBubicInterpolation(image, 2, 2)
    image3 = np.uint8(image3)

    if _output:
        cv2.imwrite("../../OpenEyes/data/nnInterpolation.png", image1)
        cv2.imwrite("../../OpenEyes/data/biLinearInterpolation.png", image2)
        cv2.imwrite("../../OpenEyes/data/biBubicInterpolation.png", image3)

    showImage(image, image1, image2, image3)


def biBubicInterpolationTest():
    image = cv2.imread("../../OpenEyes/data/seam_carving1.jpg")
    small = biBubicInterpolation(image, 0.5, 0.5)

    showImage(image, small)
# endregion


if __name__ == "__main__":
    # resizeTest(0.5, 0.5)
    # seamCarving1()
    # pyrDown(_width_scale=0.5, _height_scale=0.5)
    # pyrDown2()
    # srTest1(True)
    biBubicInterpolationTest()
