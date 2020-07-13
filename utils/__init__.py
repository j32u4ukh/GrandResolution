import cv2
import numpy as np
import scipy.ndimage
import seaborn as sns
from keras.utils import np_utils
from matplotlib import pyplot as plt


def preprocess(data, label):
    processed_data = data.astype('float32') / 255.0
    onehot_label = np_utils.to_categorical(label)

    return processed_data, onehot_label


def showSingleColor(r, b, g, size=None):
    if size is None:
        height, width = 300, 300
    else:
        (height, width) = size

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


def showImage(*args):
    for index, arg in enumerate(args):
        cv2.imshow("img {}".format(index), arg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def showImages(**kwargs):
    for _key in kwargs:
        cv2.imshow("{}".format(_key), kwargs[_key])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def biBubicInterpolation(img, scale):
    h, w = img.shape[:2]

    w = int(w * scale)
    h = int(h * scale)
    dst = cv2.resize(img.copy(), (w, h), interpolation=cv2.INTER_CUBIC)

    return dst


def biBubicInterpolation2(img, scale, prefilter=False):
    if img.ndim == 2:
        dst = scipy.ndimage.interpolation.zoom(img, scale, prefilter=prefilter)

    else:
        b, g, r = cv2.split(img)
        b = scipy.ndimage.interpolation.zoom(b, scale, prefilter=prefilter)
        g = scipy.ndimage.interpolation.zoom(g, scale, prefilter=prefilter)
        r = scipy.ndimage.interpolation.zoom(r, scale, prefilter=prefilter)

        dst = cv2.merge([b, g, r])

    return dst


def showTrainHistory(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()


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


if __name__ == "__main__":
    # showSingleColor(20, 40, 80)
    # showSingleColor(40, 80, 160)
    # showSingleColor(60, 120, 240)
    path = "data/SRCNN/Train/baboon.bmp"
    img = cv2.imread(path)
    print("img.shape:", img.shape)
    scale = 10

    small1 = biBubicInterpolation(img.copy(), 1 / scale)
    print("small1.shape:", small1.shape)
    restore1 = biBubicInterpolation(small1, scale)
    print("restore1.shape:", restore1.shape)

    small2 = biBubicInterpolation2(img.copy(), 1 / scale)
    print("small2.shape:", small2.shape)
    restore2 = biBubicInterpolation2(small2, scale)
    print("restore2.shape:", restore2.shape)

    showImages(img=img, restore1=restore1, restore2=restore2)
