import os
import glob
import h5py
import random

from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import scipy.ndimage
import cv2
import tensorflow as tf

from utils import (
    info,
    showImage,
    showImages,
    biBubicInterpolation,
    biBubicInterpolation2
)


def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)


def inputSetup_(config):
    inputSetup(config.is_train, config.image_size, config.label_size, config.scale, config.stride)


# region for inputSetup
# inputSetup:將訓練或測試資料產生並保存到 checkpoint 資料夾下的 XXX.h5
# 訓練數據會有很多筆，但測試數據因為要合併為一張圖片，所以只包含一張圖片所產生的圖片子集合。
# config:is_train, image_size, label_size, scale, stride
def inputSetup(_is_train, _image_size, _label_size, _scale, _stride):
    if _is_train:
        _data = prepare_data(dataset="Train")
        sub_input_sequence, sub_label_sequence, (nx, ny) = subData2(_data,
                                                                    _scale,
                                                                    _image_size,
                                                                    _label_size,
                                                                    _stride)
    else:
        _data = prepare_data(dataset="Test")
        sub_input_sequence, sub_label_sequence, (nx, ny) = subData2([_data[1]],
                                                                    _scale,
                                                                    _image_size,
                                                                    _label_size,
                                                                    _stride)

    """
    len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
    (sub_input_sequence[0]).shape : (33, 33, 1)
    """
    # Make list to numpy array. With this transform
    arr_data = np.asarray(sub_input_sequence)  # [?, 33, 33, 1]
    arr_label = np.asarray(sub_label_sequence)  # [?, 21, 21, 1]

    make_data(_is_train, arr_data, arr_label)

    return arr_data, arr_label, (nx, ny)


def prepare_data(dataset, _sub_dataset="Set5"):
    """
    Args:
      dataset: choose train dataset or test dataset
      _sub_dataset:test dataset 有 Set5 和 Set14 兩個數據來源

      For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
    """

    if dataset == "Train":
        data_dir = os.path.join(os.getcwd(), "SRCNN", dataset)
    else:
        data_dir = os.path.join(os.getcwd(), "SRCNN", dataset, _sub_dataset)

    # print("data_dir:", data_dir)
    _files = glob.glob(os.path.join(data_dir, "*.bmp"))
    return _files


@info("請改用 subData2")
def subData(_data, _scale, _image_size, _label_size, _stride):
    sub_input_sequence = []
    sub_label_sequence = []

    # padding 目的?
    padding = abs(_image_size - _label_size) / 2  # 6
    nx = ny = 0

    for i in range(len(_data)):
        # 大小相同但解析度下降的圖片 與 原圖
        _input, _label = preprocess(_data[i], _scale)

        if len(_input.shape) == 3:
            h, w, _ = _input.shape
        else:
            h, w = _input.shape

        # 將 _input 與 _label 拆分成多個子集合
        for x in range(0, h - _image_size + 1, _stride):
            nx += 1
            ny = 0
            for y in range(0, w - _image_size + 1, _stride):
                ny += 1
                # 預設 sub_input[i].shape:(33, 33)
                sub_input = (_input[x: x + _image_size,
                             y: y + _image_size])

                # 預設 sub_label[i].shape:(21, 21)
                sub_label = (_label[x + int(padding):x + int(padding) + _label_size,
                             y + int(padding):y + int(padding) + _label_size])

                # Make channel value
                sub_input = sub_input.reshape([_image_size, _image_size, 1])
                sub_label = sub_label.reshape([_label_size, _label_size, 1])

                # 將圖片的子集合加入陣列中存取
                sub_input_sequence.append(sub_input)  # sub_input_sequence[i].shape = (33, 33, 1)
                sub_label_sequence.append(sub_label)  # sub_label_sequence[i].shape = (21, 21, 1)

    return sub_input_sequence, sub_label_sequence, (nx, ny)


def subData2(_data, _scale, _image_size, _label_size, _stride):
    sub_input_sequence = []
    sub_label_sequence = []

    # padding = abs(_image_size - _label_size) / 2  # 6
    nx = ny = 0

    for i in range(len(_data)):
        # 大小相同但解析度下降的圖片 與 原圖
        _input, _label = preprocess(_data[i], scale=_scale)

        if len(_input.shape) == 3:
            h, w, _ = _input.shape
            channel = 3
        else:
            h, w = _input.shape
            channel = 1

        # 將 _input 與 _label 拆分成多個子集合
        for x in range(0, h - _image_size + 1 + 1, _stride):
            nx += 1
            ny = 0
            for y in range(0, w - _image_size + 1 + 1, _stride):
                ny += 1
                # 預設 sub_input[i].shape:(33, 33, ch)
                sub_input = _input[x: x + _image_size, y: y + _image_size]

                # 預設 sub_label[i].shape:(33, 33, ch)
                sub_label = _label[x: x + _label_size, y: y + _label_size]

                # Make channel value
                is_data_valid = True
                try:
                    sub_input = sub_input.reshape([_image_size, _image_size, channel])
                    sub_label = sub_label.reshape([_label_size, _label_size, channel])
                except ValueError:
                    is_data_valid = False

                # Make channel value
                # sub_input = sub_input.reshape([_image_size, _image_size, channel])
                # sub_label = sub_label.reshape([_label_size, _label_size, channel])

                # if (nx + ny) == 2:
                #     print("x0:{}, x1:{}, y0:{}, y1:{}".format(x, x + _image_size, y, y + _image_size))
                #     print("sub_input:{}, sub_label:{}".format(sub_input.shape, sub_label.shape))

                # 將圖片的子集合加入陣列中存取
                if is_data_valid:
                    sub_input_sequence.append(sub_input)  # sub_input_sequence[i].shape = (33, 33, ch)
                    sub_label_sequence.append(sub_label)  # sub_label_sequence[i].shape = (21, 21, ch)

    return sub_input_sequence, sub_label_sequence, (nx, ny)


def preprocess(path, _is_gray=False, scale=3):
    """
    Preprocess single image file
      (1) Read original image as YCbCr format (and grayscale as default)
      (2) Normalize
      (3) Apply image file with bicubic interpolation

    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)

    Args:
      path: file path of desired file
      _is_gray:是否為灰階圖片，否則為彩圖
      scale: paremeter of resize
    """
    # 讀取原始圖片
    _img = imRead(path, _is_gray)
    # modcrop:裁減原始圖片，使之為 scale 的倍數，方便後面做縮放
    _label = modcrop(_img, scale)

    # 將原圖縮小 scale 倍
    # scipy.ndimage.interpolation.zoom(label_, (1. / scale), prefilter=False)
    _small_img = biBubicInterpolation2(_label, (1. / scale), (1. / scale))

    # 再將原圖放大 scale 倍，一縮一放的過程中，產生大小相同，但解析度下降的圖片
    _input = biBubicInterpolation2(_small_img, scale, scale)

    # 正規化 0 ~ 1
    _input = _input / 255.
    _label = _label / 255.

    # 解析度下降的圖片 與 原圖
    return _input, _label


def imread(path, is_grayscale=True):
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    """
    if is_grayscale:
        # flatten=True: 形成單層的灰階通道
        return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
    else:
        return scipy.misc.imread(path, mode='YCbCr').astype(np.float)


def imRead(_path, _is_gray=True):
    if _is_gray:
        _img = cv2.imread(_path, cv2.IMREAD_GRAYSCALE)
    else:
        _img = cv2.imread(_path, cv2.IMREAD_COLOR)

    return _img


def modcrop(_img, scale=3):
    """
    To scale down and up the original image, first thing to do is to have no remainder while scaling operation.

    We need to find modulo of height (and width) and scale factor.
    Then, subtract the modulo from height (and width) of original image size.
    There would be no remainder even after scaling operation.
    """
    image = _img.copy()
    # print("image.shape:{}, scale:{}".format(_img.shape, scale))

    # np.mod(h, scale):返回 h / scale 的餘數，扣掉後便可被 scale 整除
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0: h, 0: w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0: h, 0: w]

    return image


def make_data(_is_train, data, label):
    """
    Make input data as h5 file format
    Depending on 'is_train' (flag value), savepath would be changed.
    """
    save_path = os.path.join(os.getcwd(), "SRCNN", 'checkpoint')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if _is_train:
        save_path = os.path.join(save_path, 'train.h5')
    else:
        save_path = os.path.join(save_path, 'test.h5')

    # data 和 label 預設類型是 numpy array ，但若建立時內部陣列維度不相等，內部數據將被轉為 dtype=object
    # 導致 h5py 無法儲存: TypeError: Object dtype dtype('O') has no native HDF5 equivalent
    with h5py.File(save_path, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)
# endregion


def read_data(path):
    """
    Read h5 format data file
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values

    Args:
      path: file path of desired file
    """
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label


def readData(_dataset):
    """
    Read h5 format data file
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values

    Args:
      _dataset: train data or test data
    """
    if _dataset == "Train":
        _path = os.path.join(os.getcwd(), "SRCNN", "checkpoint", "train.h5")
    else:
        _path = os.path.join(os.getcwd(), "SRCNN", "checkpoint", "test.h5")

    with h5py.File(_path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label


def merge(images, size):
    # images.shape: (2488, 21, 21, 1)
    print("Start merge...")

    h, w = images.shape[1], images.shape[2]
    print("h: {}, w:{}".format(h, w))
    [nx, ny] = size
    print("nx: {}, ny:{}".format(nx, ny))
    img = np.zeros((h * nx, w * ny, 1))
    print("img: {}".format(img.shape))

    for idx, image in enumerate(images):
        i = idx % ny
        j = idx // ny
        print("idx % ny = {}, idx // ny = {}".format(i, j))
        print("img[{}: {}, {}: {}, :]".format(j * h, j * h + h, i * w, i * w + w))
        print("image.shape:", image.shape)
        try:
            img[j * h: j * h + h, i * w: i * w + w, :] = image
        except ValueError:
            print("ValueError img[{}: {}, {}: {}, :]".format(j * h, j * h + h, i * w, i * w + w))
            break

        print("img.shape:", img.shape)

    return img


def mergeImages(_images, _stride, _n_size):
    _patch_height, _patch_width, _channels = _images[0].shape
    _n_height, _n_width = _n_size

    _height = _patch_height + (_n_height - 1) * _stride
    _width = _patch_width + (_n_width - 1) * _stride

    print("height:{}, width:{}".format(_height, _width))
    dst = np.zeros((_height, _width, _channels))

    _idx = -1
    for h in range(0, _height - _patch_height + 2, _stride):
        for w in range(0, _width - _patch_width + 2, _stride):
            _idx += 1
            dst[h: h + _patch_height, w: w + _patch_width, :] = _images[_idx] * 255

    dst = np.clip(dst, 0, 255)
    return np.uint8(dst)


def imsave(image, path):
    scipy.misc.imsave(path, image)


if __name__ == "__main__":
    epoch = 100
    batch_size = 128
    image_size = 32
    label_size = 32
    learning_rate = 1e-4
    c_dim = 3
    scale = 3
    stride = 16
    checkpoint_dir = "checkpoint"
    sample_dir = "result"
    is_train = True

    # files = prepare_data("Train")
    files = prepare_data("Test")
    # files = prepare_data("Test", "Set14")

    img = imRead(files[1], False)
    showImage(img)
    # gray = imRead(files[0], True)
    # print("img.shape:{}, gray.shape:{}".format(img.shape, gray.shape))
    # img_mod3 = modcrop(img)
    # gray_mod3 = modcrop(gray)
    # print("img.shape:{}, gray.shape:{}".format(img_mod3.shape, gray_mod3.shape))
    # img_mod5 = modcrop(img, 5)
    # gray_mod5 = modcrop(gray, 5)
    # print("img.shape:{}, gray.shape:{}".format(img_mod5.shape, gray_mod5.shape))

    # input_, label_ = preprocess(files[0])
    # temp_input = np.uint8(input_.copy() * 255.)
    # temp_label = np.uint8(label_.copy() * 255.)
    # showImage(img, temp_input, temp_label)

    # sub_input_sequence, sub_label_sequence, (nx, ny) = subData(files, scale, image_size, label_size, stride)
    sub_input_sequence, sub_label_sequence, (nx, ny) = subData2([files[0]],
                                                                scale,
                                                                image_size,
                                                                label_size,
                                                                stride)

    # arr_data, arr_label, (nx, ny) = inputSetup(is_train, image_size, label_size, scale, stride)
    # data, label = readData("Train")

