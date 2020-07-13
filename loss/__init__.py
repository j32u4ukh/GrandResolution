import cv2
import numpy as np
import tensorflow as tf
import torch
from tensorflow.math import (
    greater,
    add,
    subtract,
    multiply,
    divide,
    square,
    pow as tf_pow,
    reduce_mean as tf_mean,
    reduce_std as tf_std
)

from utils import (
    showImage
)
from utils.math import (
    log,
    multiOperation
)


class PyTorchLoss:
    # shape: [N, C, H, W]
    @staticmethod
    def ssim(x, y, is_normalized=False):
        k1 = 0.01
        k2 = 0.03
        L = 1.0 if is_normalized else 255.0

        c1 = np.power(k1 * L, 2.0)
        c2 = np.power(k2 * L, 2.0)
        c3 = c2 / 2.0

        ux = x.mean()
        uy = y.mean()

        std_x = x.std()
        std_y = y.std()

        xy = (x - ux) * (y - uy)
        std_xy = xy.mean()

        l_xy = (2.0 * ux * uy + c1) / (ux ** 2.0 + uy ** 2.0 + c1)
        c_xy = (2.0 * std_x * std_y + c2) / (std_x ** 2.0 + std_y ** 2.0 + c2)
        s_xy = (std_xy + c3) / (std_x * std_y + c3)

        ssim = l_xy * c_xy * s_xy
        ssim = torch.clamp(ssim, -1.0, 1.0)

        return ssim

    @staticmethod
    def ssim3(x, y, is_normalized=True):
        xr, xg, xb = torch.split(x, 1, dim=2)
        yr, yg, yb = torch.split(y, 1, dim=2)

        r = PyTorchLoss.ssim(xr, yr, is_normalized)
        g = PyTorchLoss.ssim(xg, yg, is_normalized)
        b = PyTorchLoss.ssim(xb, yb, is_normalized)

        result = (r + g + b) / 3.0

        return result

    @staticmethod
    def ssim4(x, y, is_normalized=True):
        ssim4_loss = 0
        n_image = 0
        for x_image, y_image in zip(x, y):
            ssim4_loss += PyTorchLoss.ssim3(x_image, y_image, is_normalized)
            n_image += 1

        ssim4_loss /= n_image

        return ssim4_loss


# psnr: 其值不能很好地反映人眼主觀感受
def psnr(y_label, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    _result = subtract(y_label, y_pred)
    _result = square(_result)
    _result = tf_mean(_result)
    _result = multiply(-10., log(_result, 10.))
    return _result


def ssim(x, y, is_normalized=False):
    k1 = 0.01
    k2 = 0.03
    L = 1.0 if is_normalized else 255.0
    c1 = np.power(k1 * L, 2)
    c2 = np.power(k2 * L, 2)
    c3 = c2 / 2

    ux = x.mean()
    uy = y.mean()

    std_x = x.std()
    std_y = y.std()

    xy = (x - ux) * (y - uy)
    std_xy = xy.mean()

    l_xy = (2 * ux * uy + c1) / (np.power(ux, 2) + np.power(uy, 2) + c1)
    c_xy = (2 * std_x * std_y + c2) / (np.power(std_x, 2) + np.power(std_y, 2) + c2)
    s_xy = (std_xy + c3) / (std_x * std_y + c3)

    _ssim = l_xy * c_xy * s_xy
    _ssim = np.clip(_ssim, -1.0, 1.0)

    return _ssim


def tf_ssim(x, y, is_normalized=False):
    """
    k1 = 0.01
    k2 = 0.03
    L = 1.0 if is_normalized else 255.0
    c1 = np.power(k1 * L, 2)
    c2 = np.power(k2 * L, 2)
    c3 = c2 / 2
    """
    k1 = 0.01
    k2 = 0.03
    L = 1.0 if is_normalized else 255.0
    c1 = tf_pow(multiply(k1, L), 2.0)
    c2 = tf_pow(multiply(k2, L), 2.0)
    c3 = divide(c2, 2.0)

    # if type(x) is np.ndarray:
    #      x = tf.convert_to_tensor(x, dtype=tf.float32)
    # if type(y) is np.ndarray:
    #      y = tf.convert_to_tensor(y, dtype=tf.float32)

    """
    ux = x.mean()
    uy = y.mean()
    """
    ux = tf_mean(x)
    uy = tf_mean(y)

    """
    std_x = x.std()
    std_y = y.std()
    """
    std_x = tf_std(x)
    std_y = tf_std(y)

    """
    xy = (x - ux) * (y - uy)
    std_xy = xy.mean()
    """
    xy = multiply(subtract(x, ux), subtract(y, uy))
    std_xy = tf_mean(xy)

    """
    l_xy = (2 * ux * uy + c1) / (np.power(ux, 2) + np.power(uy, 2) + c1)
    """
    l_son = add(multiOperation(multiply, 2.0, ux, uy), c1)
    l_mom = multiOperation(add, tf_pow(ux, 2.0), tf_pow(uy, 2.0), c1)
    l_xy = divide(l_son, l_mom)

    """
    c_xy = (2 * std_x * std_y + c2) / (np.power(std_x, 2) + np.power(std_y, 2) + c2)
    """
    c_son = add(multiOperation(multiply, 2.0, std_x, std_y), c2)
    c_mom = multiOperation(add, tf_pow(std_x, 2.0), tf_pow(std_y, 2.0), c2)
    c_xy = divide(c_son, c_mom)

    """
    s_xy = (std_xy + c3) / (std_x * std_y + c3)
    """
    s_son = add(std_xy, c3)
    s_mom = add(multiply(std_x, std_y), c3)
    s_xy = divide(s_son, s_mom)

    one = tf.constant(1.0)
    _ssim = multiOperation(multiply, l_xy, c_xy, s_xy)
    _result = tf.cond(greater(_ssim, one), lambda: one, lambda: _ssim)

    return _result


def ssim3(x, y, is_normalized=True):
    x1, x2, x3 = np.split(x, 3, axis=2)
    y1, y2, y3 = np.split(y, 3, axis=2)

    s1 = ssim(x1, y1, is_normalized)
    s2 = ssim(x2, y2, is_normalized)
    s3 = ssim(x3, y3, is_normalized)

    result = (s1 + s2 + s3) / 3.0

    return result


def tf_ssim3(x, y, is_normalized=True):
    [x1, x2, x3] = tf.split(x, 3, axis=2)
    [y1, y2, y3] = tf.split(y, 3, axis=2)

    s1 = tf_ssim(x1, y1, is_normalized)
    s2 = tf_ssim(x2, y2, is_normalized)
    s3 = tf_ssim(x3, y3, is_normalized)

    three = tf.constant(3.0)
    result = divide(multiOperation(add, s1, s2, s3), three)

    return result


def tf_ssim3_(xy):
    x, y = tf.split(xy, 2, axis=3)
    x = tf.squeeze(x)
    print("[tf_ssim3_] x.shape:", x.shape)
    y = tf.squeeze(y)
    return tf_ssim3(x, y, is_normalized=False)


def tf_ssim3_norm(xy):
    x, y = tf.split(xy, 2, axis=3)
    x = tf.squeeze(x)
    print("[tf_ssim3_norm] x.shape:", x.shape)
    y = tf.squeeze(y)
    return tf_ssim3(x, y, is_normalized=True)


def ssim4(x, y, is_normalized=False):
    each_loss = []

    for _x, _y in zip(x, y):
        each_loss.append(ssim3(_x, _y, is_normalized))

    each_loss = np.array(each_loss)

    total_loss = each_loss.mean()

    return total_loss, each_loss


def tf_ssim4(x, y, is_normalized=False):
    print("x.shape: ", x.shape)
    print("y.shape: ", y.shape)
    stack = tf.stack([x, y], axis=4)
    print("stack.shape: ", stack.shape)

    # tf.map_fn: 接受參數為一個的函數，如 tf_ssim3_norm 等
    # 第二個參數為前一個函式的參數，若要傳遞多個參數，則須將他們合併成一個(元組不知是否可以)
    if is_normalized:
        each_loss = tf.map_fn(tf_ssim3_norm, stack)
    else:
        each_loss = tf.map_fn(tf_ssim3_, stack)

    total_loss = tf_mean(each_loss)

    return total_loss, each_loss


if __name__ == "__main__":
    def ssimTest(img_x, img_y):
        x = tf.placeholder(dtype=tf.float32,
                           shape=[None, None, None, 3])
        y = tf.placeholder(dtype=tf.float32,
                           shape=[None, None, None, 3])
        compute_ssim = tf_ssim4(x, y, True)

        # ssim(self.labels, self.pred, is_normalized=True)
        # labels[i].shape = (None, label_size, label_size, c_dim)
        # pred: (?, 32, 32, self.c_dim)
        tf_config = tf.ConfigProto(log_device_placement=True)
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.1
        with tf.Session(config=tf_config) as sess:
            # ssim_value = sess.run(compute_ssim,
            #                       feed_dict={x: img_x,
            #                                  y: img_y})
            # print("result:", ssim_value)

            total_loss, each_loss = sess.run(compute_ssim,
                                             feed_dict={x: img_x,
                                                        y: img_y})

            print("total_loss:", total_loss)  # total_loss: 0.9589584
            print("each_loss:", each_loss)    # each_loss: [0.9515831  0.9512937  0.9634177  0.96953887]

    # ================================================================================
    img1 = cv2.imread("data/splice1.png")
    img2 = cv2.imread("data/splice2.png")
    img3 = cv2.imread("data/splice3.png")
    img4 = cv2.imread("data/splice4.png")
    images = [img1, img2, img3, img4]
    showImage(img1, img2, img3, img4)

    dsts = []
    for i in range(len(images)):
        img = images[i] / 255.0
        # INTER_CUBIC = cv2.resize(_img, (_resize_cols, _resize_rows), interpolation=cv2.INTER_CUBIC)
        images[i] = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        dst = cv2.GaussianBlur(images[i].copy(), (13, 13), 0)
        dsts.append(dst)

    images = np.array(images)
    dsts = np.array(dsts)

    ssimTest(images, dsts)

    # ssim loss: 0.9724881069549962
    loss1 = ssim(images[0], dsts[0], is_normalized=True)
    loss2 = ssim(images[1], dsts[1], is_normalized=True)
    loss3 = ssim(images[2], dsts[2], is_normalized=True)
    loss4 = ssim(images[3], dsts[3], is_normalized=True)
    print("ssim loss1:", loss1)  # ssim loss1: 0.9530896078865423
    print("ssim loss2:", loss2)  # ssim loss2: 0.951598205259678
    print("ssim loss3:", loss3)  # ssim loss3: 0.9694566425946143
    print("ssim loss4:", loss4)  # ssim loss4: 0.9750622058576149
    # ssim total loss: 0.9623016653996124
    print("ssim total loss:", np.mean([loss1, loss2, loss3, loss4]))

    pt_images = torch.from_numpy(images)
    pt_dsts = torch.from_numpy(dsts)

    pt_image1 = pt_images[0]
    result = torch.split(pt_image1, 1, dim=2)

    pt_loss1 = PyTorchLoss.ssim(pt_images[0], pt_dsts[0], is_normalized=True)
    print("pt_loss1:", pt_loss1)  # pt_loss1: tensor(0.9531, dtype=torch.float64)
    pt_loss2 = PyTorchLoss.ssim(pt_images[1], pt_dsts[1], is_normalized=True)
    print("pt_loss2:", pt_loss2)  # pt_loss2: tensor(0.9516, dtype=torch.float64)
    pt_loss3 = PyTorchLoss.ssim(pt_images[2], pt_dsts[2], is_normalized=True)
    print("pt_loss3:", pt_loss3)  # pt_loss3: tensor(0.9695, dtype=torch.float64)
    pt_loss4 = PyTorchLoss.ssim(pt_images[3], pt_dsts[3], is_normalized=True)
    print("pt_loss4:", pt_loss4)  # pt_loss4: tensor(0.9751, dtype=torch.float64)
    # pt_total_loss: tensor(0.9623, dtype=torch.float64)
    pt_total_loss = torch.mean(torch.tensor([pt_loss1, pt_loss2, pt_loss3, pt_loss4]))
    pt_loss = PyTorchLoss.ssim(torch.from_numpy(images), torch.from_numpy(dsts), is_normalized=True)
    print("pt_loss:", pt_loss)  # pt_loss: tensor(0.9531, dtype=torch.float64)

    """
    上面會將圖片的三個通道"共同"計算 ssim 值
    下面則將圖片的三個通道"分別"計算 ssim 值
    """

    pt_ssim3_loss = 0
    for img, dst in zip(pt_images, pt_dsts):
        pt_ssim3_loss += PyTorchLoss.ssim3(img, dst, is_normalized=True)
    pt_ssim3_loss /= pt_images.shape[0]
    # pt_ssim3_loss: 0.9589
    print("pt_ssim3_loss:", pt_ssim3_loss)

    ssim3_loss = 0
    for img, dst in zip(images, dsts):
        ssim3_loss += ssim3(img, dst, is_normalized=True)
    ssim3_loss /= len(images)

    # ssim3_loss: 0.9589583362703996
    print("ssim3_loss:", ssim3_loss)

    # ssim4: (0.9589583362703996, array([0.95158301, 0.95129378, 0.96341769, 0.96953887]))
    print("ssim4:", ssim4(images, dsts, is_normalized=True))

    # PyTorch.ssim4: tensor(0.9589, dtype=torch.float64)
    pt_dsts = pt_dsts.requires_grad_(True)
    ssim4_loss = PyTorchLoss.ssim4(pt_images, pt_dsts, is_normalized=True)
    print("PyTorch.ssim4:", ssim4_loss)
