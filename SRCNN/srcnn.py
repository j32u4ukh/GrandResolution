import glob
import os

import cv2
import h5py
import numpy as np
import scipy.misc
import scipy.ndimage
from keras import backend as K

from submodule.Xu3.utils import getFuncName
from utils import (
    info,
    biBubicInterpolation,
    biBubicInterpolation2
)

from utils import showImages


# region setupInput
# 載入數據
def readData(idx, image_size, scale, stride, is_gray=False):
    """

    :param idx: 根據索引值來判斷要使用訓練集(-1)還是測試集(0, 1, 2, ...)
    :param image_size: 圖片大小，輸入輸出大小相同，僅解析度不同
    :param scale: 放大比例，也用於產生輸入數據
    :param stride: 拆分步長
    :param is_gray: 是否為灰階圖片
    :return: input_data, label_data, (nx, ny) 最後兩項為測試數據獨有，訓練數據沒有這兩項
    """
    # 若為訓練集
    if idx == -1:
        if is_gray:
            path = os.path.join(os.getcwd(), "SRCNN", "checkpoint", "train_gray.h5")
        else:
            path = os.path.join(os.getcwd(), "SRCNN", "checkpoint", "train.h5")

        # 檢視是否有 train.h5 ，若有則載入
        if os.path.isfile(path):
            print(f"[{getFuncName()}] train.h5 is existed, now loading...")
            with h5py.File(path, 'r') as hf:
                data = np.array(hf.get('data'))
                label = np.array(hf.get('label'))
                return data, label

    # 沒有事先生成的數據，生成並返回
    input_data, label_data, (nx, ny) = setupInput(idx, image_size, scale, stride, is_gray)

    if idx == -1:
        return input_data, label_data
    else:
        return input_data, label_data, (nx, ny)


# inputSetup:將訓練或測試資料產生並保存到 checkpoint 資料夾下的 XXX.h5
# 訓練時，會將一張圖片拆分成很多張小張的圖片，每次只訓練一小塊，但總共會訓練很多塊
# 而測試時，為了將拆分之後的圖片合併為一張原始大小的圖片，所以只能包含一張圖片所產生的圖片子集合。
# 此種做法可適用各種大小的圖片，因為拆分後的數量不是固定的，訓練時是一小塊一小塊訓練，因此拆後數量
# 不固定也沒關係，而測試時只須給它原始長寬等必要資訊，就能再合併回去，但無法保證接合處的銜接性。
def setupInput(idx, image_size, scale, stride, is_gray=False):
    """

    :param idx: 根據索引值來判斷要使用訓練集(-1)還是測試集(0, 1, 2, ...)
    :param image_size: 圖片大小，輸入輸出大小相同，僅解析度不同
    :param scale: 放大比例，也用於產生輸入數據
    :param stride: 拆分步長
    :param is_gray: 是否為灰階圖片
    :return:
    """
    if idx == -1:
        dataset = "Train"
    else:
        dataset = "Test"

    # 若為測試集，才會使用到 idx，訓練集則不會用到，給不給 idx 都沒關係
    data = prepareData(dataset=dataset, idx=idx)
    input_data, label_data, (nx, ny) = subData(data, scale, image_size, stride, is_gray=is_gray)

    """
    測試數據應該不需事先保存，畢竟一次才一張圖片，而且需要分割後的長寬個數資訊(nx, ny)，
    不同圖片的數值不同，沒有事先儲存的價值。
    """
    if idx == -1:
        makeData(input_data, label_data, is_gray=is_gray)

    return input_data, label_data, (nx, ny)


def prepareData(dataset="Train", idx=0):
    data_dir = os.path.join(os.getcwd(), "data", "SRCNN", dataset)
    # print("data_dir:", data_dir)

    # 返回符合條件的檔案，不包含子資料夾內的檔案
    files = glob.glob(os.path.join(data_dir, "*.bmp"))

    if dataset == "Train":
        return files
    else:
        # dataset Test
        if idx < 0 or len(files) <= idx:
            idx = 0

        # type -> list
        print("prepareData | file name:", files[idx])
        return files[idx: idx + 1]


@info("subData -> subData2 -> subData(沿用之前的名稱，但內容不同)")
def subData(data, scale, image_size, stride, is_gray=False):
    sub_input_sequence = []
    sub_label_sequence = []
    length = len(data)
    nx = ny = 0

    for i in range(length):
        # 大小相同但解析度下降的圖片 與 原圖
        input_data, label_data = differentResolution(data[i], is_gray=is_gray, scale=scale)

        if len(input_data.shape) == 3:
            h, w, _ = input_data.shape
            channel = 3
        else:
            h, w = input_data.shape
            channel = 1

        # 將 input_data 與 label_data 拆分成多個子集合
        for x in range(0, h - image_size + 1, stride):
            nx += 1
            ny = 0
            for y in range(0, w - image_size + 1, stride):
                ny += 1
                sub_input = input_data[x: x + image_size, y: y + image_size]
                sub_label = label_data[x: x + image_size, y: y + image_size]

                # Make channel value
                sub_input = sub_input.reshape([image_size, image_size, channel])
                sub_label = sub_label.reshape([image_size, image_size, channel])

                # 將圖片的子集合加入陣列中存取
                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)

    input_data = np.asarray(sub_input_sequence)  # [?, 33, 33, 1]
    label_data = np.asarray(sub_label_sequence)  # [?, 21, 21, 1]
    return input_data, label_data, (nx, ny)


def differentResolution(path, is_gray=False, scale=3):
    """
    將圖片縮小再放大，以獲得大小相同但解析度下降的圖片

    1. Read original image
    2. Normalize
    3. Apply image file with bicubic interpolation

    :param path: file path of desired file
    :param is_gray: 是否為灰階圖片，否則為彩圖
    :param scale: paremeter of resize
    :return:
    """
    # 讀取原始圖片
    img = imRead(path, is_gray)

    print("differentResolution | 原始圖片 shape:", img.shape)

    # modcrop:裁減原始圖片，使之為 scale 的倍數，方便後面做縮放
    label_data = modCrop(img, scale)

    print("differentResolution | 裁減圖片 shape:", label_data.shape)

    # 將原圖縮小 scale 倍
    small_img = biBubicInterpolation(label_data, (1. / scale))

    # 再將原圖放大 scale 倍，一縮一放的過程中，產生大小相同，但解析度下降的圖片
    input_data = biBubicInterpolation(small_img, scale)

    # 正規化 0 ~ 1
    input_data = input_data / 255.
    label_data = label_data / 255.

    # 解析度下降的圖片(low-resolution) 與 原圖(high-resolution)
    return input_data, label_data


def imRead(path, is_gray=True):
    if is_gray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)

    return img


def modCrop(img, scale=3):
    """ 裁減原始圖片，使之為 scale 的倍數，方便後面做縮放
    To scale down and up the original image, first thing to do is to have no remainder while scaling operation.

    We need to find modulo of height (and width) and scale factor.
    Then, subtract the modulo from height (and width) of original image size.
    There would be no remainder even after scaling operation.
    """
    image = img.copy()
    h, w = image.shape[:2]

    # np.mod(x, scale): 返回 x / scale 的餘數，扣掉後 x 便可被 scale 整除
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)

    # 擷取可被整除的子圖像，現在的圖片大小是可以被 scale 所整除的數值了
    image = image[0: h, 0: w, :]

    return image


def makeData(data, label, is_gray=False):
    save_path = os.path.join(os.getcwd(), "SRCNN", 'checkpoint')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if is_gray:
        save_path = os.path.join(save_path, 'train_gray.h5')
    else:
        save_path = os.path.join(save_path, 'train.h5')

    # data 和 label 預設類型是 numpy array ，但若建立時內部陣列維度不相等，內部數據將被轉為 dtype=object
    # 導致 h5py 無法儲存: TypeError: Object dtype dtype('O') has no native HDF5 equivalent
    with h5py.File(save_path, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)

    print(f"[{getFuncName()}] Save completed: {save_path}")
# endregion


# 用於將測試數據拼接回原圖大小
def mergeImages(images, stride, n_size):
    # shape: C, H, W
    channel, patch_height, patch_width = images[0].shape
    print("mergeImages | images[0].shape:", images[0].shape)

    n_height, n_width = n_size
    print(f"mergeImages | n_height: {n_height}, n_width: {n_width}")

    height = patch_height + (n_height - 1) * stride
    width = patch_width + (n_width - 1) * stride
    print(f"height:{height}, width:{width}")

    dst = np.zeros((channel, height, width))

    idx = -1
    for h in range(0, height - patch_height + 2, stride):
        for w in range(0, width - patch_width + 2, stride):
            idx += 1
            # 由於 patch_height, patch_width 與 stride 不相等，因此會產生重疊的部分，
            # 這裡直接將後面的區塊蓋在前面的區塊之上
            dst[:, h: h + patch_height, w: w + patch_width] = images[idx] * 255

    dst = np.clip(dst, 0, 255)
    return np.uint8(dst)


# ==================================================
# ==================================================
def inputSetup_(config):
    inputSetup(config.is_training, config.image_size, config.label_size, config.scale, config.stride)


# region for inputSetup
# inputSetup:將訓練或測試資料產生並保存到 checkpoint 資料夾下的 XXX.h5
# 訓練時，會將一張圖片拆分成很多張小張的圖片，每次只訓練一小塊，但總共會訓練很多塊
# 而測試時，為了將拆分之後的圖片合併為一張原始大小的圖片，所以只能包含一張圖片所產生的圖片子集合。
# 此種做法可適用各種大小的圖片，因為拆分後的數量不是固定的，訓練時是一小塊一小塊訓練，因此拆後數量
# 不固定也沒關係，而測試時只須給它原始長寬等必要資訊，就能再合併回去，但無法保證接合處的銜接性。
# config: is_training, image_size, label_size, scale, stride
def inputSetup(is_training, image_size, label_size, scale, stride):
    if is_training:
        data = prepare_data(dataset="Train")
        sub_input_sequence, sub_label_sequence, (nx, ny) = subData2(data,
                                                                    scale,
                                                                    image_size,
                                                                    label_size,
                                                                    stride)
    else:
        data = prepare_data(dataset="Test")
        sub_input_sequence, sub_label_sequence, (nx, ny) = subData2([data[1]],
                                                                    scale,
                                                                    image_size,
                                                                    label_size,
                                                                    stride)

    """
    len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
    (sub_input_sequence[0]).shape : (33, 33, 1)
    """
    # Make list to numpy array. With this transform
    # 因與原始程式所用卷積層不同，導致數據形狀不同
    arr_data = np.asarray(sub_input_sequence)  # [?, 33, 33, 1]
    arr_label = np.asarray(sub_label_sequence)  # [?, 21, 21, 1]

    make_data(is_training, arr_data, arr_label)

    return arr_data, arr_label, (nx, ny)


def prepare_data(dataset, _sub_dataset="Set5"):
    """
    Args:
      dataset: choose train dataset or test dataset
      _sub_dataset:test dataset 有 Set5 和 Set14 兩個數據來源

      For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
    """

    # TODO: 若要再次使用，需要調整檔案路徑
    if dataset == "Train":
        data_dir = os.path.join(os.getcwd(), "SRCNN", dataset)
    else:
        data_dir = os.path.join(os.getcwd(), "SRCNN", dataset, _sub_dataset)

    # print("data_dir:", data_dir)
    _files = glob.glob(os.path.join(data_dir, "*.bmp"))
    return _files


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

        # 將 _input 與 _label 拆分成多個子集合，由於 _image_size 與 _stride 不相等，
        # 因此所擷取的小區塊，應該是會重疊的
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
                    # 由於我的卷積層使用了 padding，因此 _image_size == _label_size，
                    # 與原始程式所用數值會因此而產生差異
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


def modcrop(_img, scale=3):
    """ 裁減原始圖片，使之為 scale 的倍數，方便後面做縮放
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
    else:
        h, w = image.shape

    # 扣除無法整除的部分
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)

    # 擷取可被整除的子圖像，現在的圖片大小是可以被 scale 所整除的數值了
    image = image[0: h, 0: w, :]

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


def imsave(image, path):
    scipy.misc.imsave(path, image)


# 其值不能很好地反映人眼主觀感受
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


if __name__ == "__main__":
    # train_files = prepareData()
    # test_file = prepareData("Test", 5)
    # epoch = 100
    # batch_size = 128
    image_size = 32
    # label_size = 32
    # learning_rate = 1e-4
    # c_dim = 3
    scale = 3
    stride = 16
    # checkpoint_dir = "checkpoint"
    # sample_dir = "result"
    # is_train = True
    #
    # # files = prepare_data("Train")
    # files = prepare_data("Test")
    # files = prepareData()

    # img = imRead(files[0], False)
    # showImage(img)
    # img_mod3 = modCrop(img, scale=3)
    # print("img_mod3.shape:", img_mod3.shape)
    # img_mod5 = modCrop(img, scale=5)
    # print("img_mod5.shape:", img_mod5.shape)
    # showImages(img=img, img_mod3=img_mod3, img_mod5=img_mod5)

    # input_data, label_data = differentResolution(files[0], scale=3)

    # input_, label_ = preprocess(files[0])
    # temp_input = np.uint8(input_data.copy() * 255.)
    # temp_label = np.uint8(label_data.copy() * 255.)
    # showImages(img=img, temp_input=temp_input, temp_label=temp_label)

    # subData(data, scale, image_size, stride, is_gray=False)
    # sub_input_sequence, sub_label_sequence, (nx, ny) = subData(files, scale, image_size, stride, is_gray=False)
    # input_data, label_data, _ = setupInput(idx=-1, image_size=image_size, scale=scale, stride=stride)
    input_data, label_data, (nx, ny) = readData(idx=1, image_size=image_size, scale=scale, stride=stride)
    result = mergeImages(input_data, stride, (nx, ny))
    origin = mergeImages(label_data, stride, (nx, ny))

    showImages(origin=origin, result=result)
