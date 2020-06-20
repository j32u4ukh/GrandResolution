import os
import time
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
from skimage import metrics

import torch
import torch.nn as nn
from torch.utils import data

from GrandResolution.loss import (
    tf_ssim4,
    ssim3
)
from GrandResolution.utils import (
    showImage
)
from SRCNN.srcnn import (
    inputSetup,
    readData,
    mergeImages
)


class TfSrcnn:
    def __init__(self,
                 epoch=100,
                 image_size=32,
                 label_size=32,
                 learning_rate=1e-4,
                 batch_size=128,
                 c_dim=3,
                 scale=3,
                 stride=16,
                 checkpoint_dir=None,
                 sample_dir=None):
        # 嘗試將 SRCNN 的建立移至 tf.Session() 之外，事後定義 sess
        # self.sess = sess
        self.sess = None

        self.epoch = epoch
        self.is_grayscale = (c_dim == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.c_dim = c_dim
        self.scale = scale
        self.stride = stride

        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir

        # region build_model
        self.images = None
        self.labels = None
        self.weights = None
        self.biases = None
        self.pred = None
        self.total_loss = None
        self.each_loss = None
        self.loss = None
        self.saver = None
        self.importer = None
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.build_model()
        # endregion

        self.train_op = None
        # self.training_history = []

    def setSess(self, _sess):
        self.sess = _sess

    def build_model(self):
        # 先對低分辨率圖像(LR)進行雙三次插值(Bicubic Interpolation)處理，得到和高分辨率圖像"一樣大小的圖像"作為輸入圖像
        # 即 image_size == label_size，其實應該用同一個變數就好，更可清楚理解兩者為同一數值
        # images[i].shape = (None, image_size, image_size, c_dim)
        self.images = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.image_size, self.image_size, self.c_dim],
                                     name='images')

        # labels[i].shape = (None, label_size, label_size, c_dim)
        self.labels = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.label_size, self.label_size, self.c_dim],
                                     name='labels')

        # images: (?, 32, 32, 3), labels: (?, 32, 32, 3)
        # print("images: {}, labels:{}".format(self.images.shape, self.labels.shape))

        self.weights = {
            # 參數 1, 2: fliter 大小；參數 3: input channels；參數 4: output channels
            # w1: (9, 9, self.c_dim, 64), w2: (1, 1, 64, 32), w3: (5, 5, 32, self.c_dim)
            'w1': tf.Variable(tf.truncated_normal([9, 9, self.c_dim, 64], stddev=1e-1), name='w1'),
            'w2': tf.Variable(tf.truncated_normal([1, 1, 64, 32], stddev=1e-1), name='w2'),
            'w3': tf.Variable(tf.truncated_normal([5, 5, 32, self.c_dim], stddev=1e-1), name='w3')
        }

        # for key in self.weights.keys():
        #     print("{}: {}".format(key, self.weights[key].shape))

        self.biases = {
            # b1: (64,), b2: (32,), b3: (self.c_dim,)
            'b1': tf.Variable(tf.truncated_normal([64], stddev=1e-3), name='b1'),
            'b2': tf.Variable(tf.truncated_normal([32], stddev=1e-3), name='b2'),
            'b3': tf.Variable(tf.truncated_normal([self.c_dim], stddev=1e-3), name='b3')
        }

        # for key in self.biases.keys():
        #     print("{}: {}".format(key, self.biases[key].shape))

        # pred: (?, 32, 32, self.c_dim)
        self.pred = self.model()
        # print("pred: {}".format(self.pred.shape))

        # Loss function
        # self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        # self.loss = PSNRLoss(self.labels, self.pred)
        # self.loss = metrics.structural_similarity(self.labels, self.pred, data_range=255)
        one = tf.constant(1.0)
        self.total_loss, self.each_loss = tf_ssim4(self.labels, self.pred, is_normalized=True)

        # 雖然命名 total_loss 但應該是 ssim 的得分，因此 1 - total_loss 才是模型誤差
        self.loss = tf.subtract(one, self.total_loss)

        # 建立 saver 物件
        """如果你希望每2小時保存一次模型，並且只保存最近的5個模型文件：
        tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)
        注意：tensorflow默認只會保存最近的5個模型文件，如果你希望保存更多，可以通過max_to_keep來指定
        
        如果我們不對tf.train.Saver指定任何參數，默認會保存所有變量。
        如果你不想保存所有變量，而只保存一部分變量，可以通過指定variables/collections。
        在創建tf.train.Saver實例時，通過將需要保存的變量構造list或者dictionary，傳入到Saver中。
        例：
        w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
        w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
        saver = tf.train.Saver([w1,w2])"""
        self.saver = tf.train.Saver()

    def model(self):
        # original padding='VALID'
        # 參數: input, filter, strides, padding
        # conv1: (?, 32, 32, 64)
        self.conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1, 1, 1, 1], padding='SAME') +
                                self.biases['b1'])
        # conv2: (?, 32, 32, 32)
        self.conv2 = tf.nn.relu(tf.nn.conv2d(self.conv1, self.weights['w2'], strides=[1, 1, 1, 1], padding='SAME') +
                                self.biases['b2'])
        # conv3: (?, 32, 32, c_dim)
        self.conv3 = (tf.nn.conv2d(self.conv2, self.weights['w3'], strides=[1, 1, 1, 1], padding='SAME') +
                      self.biases['b3'])

        # print("conv1: {}".format(conv1.shape))
        # print("conv2: {}".format(conv2.shape))
        # print("conv3: {}".format(conv3.shape))

        return self.conv3

    def train(self, is_data_prepared=True):
        # input_setup:將訓練或測試資料產生並保存到 checkpoint 資料夾下的 XXX.h5
        # input_setup(config)
        # TypeError: cannot unpack non-iterable NoneType object
        if not is_data_prepared:
            arr_data, arr_label, (nx, ny) = inputSetup(is_training=True,
                                                       image_size=self.image_size,
                                                       label_size=self.label_size,
                                                       scale=self.scale,
                                                       stride=self.stride)
        # data_dir = os.path.join('./{}'.format(self.checkpoint_dir), "train.h5")

        # 將圖片子集合讀取進來 readData(idx, image_size, scale, stride)
        train_data, train_label = readData(-1, self.image_size, self.scale, self.stride)

        # Stochastic gradient descent with the standard backpropagation
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        # self.lr = tf.train.AdamOptimizer(self.learning_rate)

        tf.initialize_all_variables().run()

        # 重新訓練: counter = 0, 接續訓練: counter = 上一次訓練儲存模型編號
        counter = 34500
        start_time = time.time()

        # 若有之前的訓練模型，則載入
        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        print("Training...")

        last_loss = 1000
        indexs = [i for i in range(len(train_data))]
        for ep in range(self.epoch):
            # Run by batch images
            batch_idxs = len(train_data) // self.batch_size
            # 每個 epoch 打亂一次 indexs
            np.random.shuffle(indexs)
            for idx in range(0, batch_idxs):
                batch_indexs = indexs[idx * self.batch_size: (idx + 1) * self.batch_size]
                # batch_images = train_data[idx * self.batch_size: (idx + 1) * self.batch_size]
                # batch_labels = train_label[idx * self.batch_size: (idx + 1) * self.batch_size]
                batch_images = train_data[batch_indexs]
                batch_labels = train_label[batch_indexs]

                counter += 1
                _, err = self.sess.run([self.train_op, self.loss],
                                       feed_dict={self.images: batch_images,
                                                  self.labels: batch_labels})
                # print("err:", err)

                # self.training_history.append(err)

                if counter % 500 == 0:
                    print("sess size:", self.sess.__sizeof__())

                    # 判斷 loss ，錯誤率若增加就沒有必要保存模型了
                    if err < last_loss:
                        self.save(counter)

                    last_loss = err
                    # self.predict()

                if counter % 10 == 0:
                    print("Epoch: {:0>2d}, step: {:0>2d}, time: {:.4f}, loss: {:.8f}".format((ep + 1),
                                                                                             counter,
                                                                                             time.time() - start_time,
                                                                                             err))

                if counter % 50 == 0:
                    pass

                #     training_images, training_labels, training_pred = self.sess.run([self.images,
                #                                                                      self.labels,
                #                                                                      self.pred],
                #                                                                     feed_dict={
                #                                                                         self.images: batch_images,
                #                                                                         self.labels: batch_labels})
                #
                #     conv1, conv2, conv3 = self.sess.run([self.conv1, self.conv2, self.conv3],
                #                                         feed_dict={self.images: batch_images,
                #                                                    self.labels: batch_labels})
                #     print("conv1")
                #     print(conv1[0, :, :, 0])
                #     print("conv2")
                #     print(conv2[0, :, :, 0])
                #     print("conv3")
                #     print(conv3[0, :, :])
                #
                #     temp_data = np.uint8(np.clip(training_images[0].squeeze() * 255, 0, 255))
                #     temp_label = np.uint8(np.clip(training_labels[0].squeeze() * 255, 0, 255))
                #     temp_pred = np.uint8(np.clip(training_pred[0].squeeze() * 255, 0, 255))
                #     showImages(data=temp_data, label=temp_label, pred=temp_pred)
                #     print("="*30)

        # self.predict()
        # temp_data = np.uint8(np.clip(training_images[0].squeeze() * 255, 0, 255))
        # temp_label = np.uint8(np.clip(training_labels[0].squeeze() * 255, 0, 255))
        # temp_pred = np.uint8(np.clip(training_pred[0].squeeze() * 255, 0, 255))
        # showImages(data=temp_data, label=temp_label, pred=temp_pred)
        # print("=" * 30)

    # predict 不需要初始化步驟 init= tf.initialize_all_variables()
    def predict(self):
        # TODO: is_data_prepared 若測試數據不存在才執行
        # input_setup:將訓練或測試資料產生並保存到 checkpoint 資料夾下的 XXX.h5
        arr_data, arr_label, (nx, ny) = inputSetup(is_training=False,
                                                   image_size=self.image_size,
                                                   label_size=self.label_size,
                                                   scale=self.scale,
                                                   stride=self.stride)
        # print("nx: {}, ny: {}".format(nx, ny))

        # 將圖片子集合讀取進來
        # data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")
        _data, _label = readData("Test")
        # print("data: {}, label:{}".format(_data.shape, _label.shape))

        # 若有之前的訓練模型，則載入
        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        print("Testing...")
        # tf.global_variables_initializer().run()

        # result.shape: (2488, 21, 21, 1)
        result = self.pred.eval({self.images: _data,
                                 self.labels: _label})

        # print("result.shape:", result.shape)
        # print(result[0])

        loss = self.loss.eval({self.images: _data,
                               self.labels: _label})
        print("model loss:", loss)
        print("total loss:", self.total_loss.eval({self.images: _data,
                                                   self.labels: _label}))
        print("each loss:", self.each_loss.eval({self.images: _data,
                                                 self.labels: _label}))

        result = result * 255
        result = np.uint8(result)
        result = np.clip(result, 0, 255)
        # print(result[0])

        result = result * 255
        result = np.uint8(result)
        result = np.clip(result, 0, 255)

        # 將預測圖片子集合，彙整成一張圖片
        merged_result = mergeImages(result, self.stride, [nx, ny])
        # print("merged result.shape:", merged_result.shape)

        # numpy squeeze:將陣列 shape 中為1的維度，例如>> (1,1,10) → (10,)
        squeeze_result = merged_result.squeeze()
        # print("squeezed result.shape:", squeeze_result.shape)

        origin = mergeImages(_data, self.stride, [nx, ny])
        multichannel = squeeze_result.ndim == 3
        if not multichannel:
            origin = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)

        # print("origin.shape:", origin.shape)

        error = metrics.structural_similarity(origin,
                                              squeeze_result,
                                              data_range=origin.max()-origin.min(),
                                              multichannel=multichannel)
        print("metrics error:", 1 - error)

        error = ssim3(origin, squeeze_result, is_normalized=False)
        print("ssim3 error:", 1 - error)

        _file_name = datetime.now().strftime("%Y%m%d%H%M%S")
        image_path = os.path.join(os.getcwd(), "SRCNN", self.sample_dir, "{}.png".format(_file_name))
        # print("image_path:", image_path)

        # cv2.imwrite(image_path, squeeze_result)
        showImage(squeeze_result)

        return _data, result, merged_result, squeeze_result

    def save(self, step):
        # https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/583546/
        model_name = "SRCNN.model"
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(os.getcwd(), "SRCNN", self.checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # 儲存模型
        # self.saver = tf.train.Saver()
        """Tensorflow的模型儲存時有幾點需要注意：
        1、利用tf.train.write_graph()預設情況下只匯出了網路的定義（沒有權重weight）。
        2、利用tf.train.Saver().save()匯出的檔案graph_def與權重是分離的。
        
        我們知道，graph_def檔案中沒有包含網路中的Variable值（通常情況儲存了權重），但是卻包含了constant值，
        所以如果我們能把Variable轉換為constant，即可達到使用一個檔案同時儲存網路架構與權重的目標"""
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self):
        # https://www.itread01.com/content/1544892147.html
        """
        saver=tf.train.import_meta_graph('./checkpoint_dir/MyModel-1000.meta')
        step1 構造網絡圖
        一個比較笨的方法是，手敲代碼，實現跟模型一模一樣的圖結構。其實，我們既然已經保存了圖，那就沒必要在去手寫一次圖結構代碼。
        上面一行代碼，就把圖加載進來了。

        step2 加載參數
        僅僅有圖並沒有用，更重要的是，我們需要前面訓練好的模型參數（即weights、biases等），變量值需要依賴於Session，
        因此在加載參數時，先要構造好Session。
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel-1000.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))
        :return:
        """
        print(" [*] Reading checkpoints...")
        model_dir = "{}_{}".format("srcnn", self.label_size)
        checkpoint_dir = os.path.join(os.getcwd(), "SRCNN", self.checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # ckpt_name = SRCNN.model-2550000
            # ckpt.model_checkpoint_path = ../OpenEyes/SRCNN/checkpoint/srcnn_21/SRCNN.model-2550000
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print("ckpt_name:", ckpt_name)

            # ../OpenEyes/SRCNN/checkpoint/srcnn_21/SRCNN.model-2550000
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def load2(self):
        # https://www.itread01.com/content/1544892147.html
        # https://blog.csdn.net/huachao1001/article/details/78501928
        """
        saver=tf.train.import_meta_graph('./checkpoint_dir/MyModel-1000.meta')
        step1 構造網絡圖
        一個比較笨的方法是，手敲代碼，實現跟模型一模一樣的圖結構。其實，我們既然已經保存了圖，那就沒必要在去手寫一次圖結構代碼。
        上面一行代碼，就把圖加載進來了。

        step2 加載參數
        僅僅有圖並沒有用，更重要的是，我們需要前面訓練好的模型參數（即weights、biases等），變量值需要依賴於Session，
        因此在加載參數時，先要構造好Session。
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel-1000.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))
        :return:
        """
        importer = tf.train.import_meta_graph('SRCNN/checkpoint/srcnn_32/SRCNN.model-13000.meta')
        # print("new_saver:", new_saver)

        _latest_checkpoint = tf.train.latest_checkpoint('SRCNN/checkpoint/srcnn_32/')
        # print("latest_checkpoint:", _latest_checkpoint)

        importer.restore(self.sess, _latest_checkpoint)

        # w1 是 tensor 的名稱， :0 似乎是 w1 這個 output 的 index
        # print(self.sess.run('w1:0'))

    def test(self):
        # https://blog.csdn.net/changeforeve/article/details/80268522
        # self.load2()
        model_dir = "{}_{}".format("srcnn", self.label_size)
        checkpoint_dir = os.path.join(os.getcwd(), "SRCNN", self.checkpoint_dir, model_dir)
        print("checkpoint_dir:", checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            print(self.sess.run("w1:0"))

            graph = tf.get_default_graph()
            print(graph)
        else:
            print("test failed.")

        return ckpt


class PtSrcnn(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class SrcnnDataLoader(data.Dataset):
    def __init__(self, idx, image_size, scale, stride, is_gray=False):
        self.image_size = image_size
        self.scale = scale
        self.stride = stride
        if is_gray:
            self.c_dim = 1
        else:
            self.c_dim = 3

        if idx == -1:
            self.input_data, self.label_data = readData(idx, self.image_size, self.scale, self.stride)
        else:
            self.input_data, self.label_data, (self.nx, self.ny) = readData(idx,
                                                                            self.image_size,
                                                                            self.scale,
                                                                            self.stride)

    def __getitem__(self, index):
        input_data = torch.from_numpy(self.input_data[index]).float()
        label_data = torch.from_numpy(self.label_data[index]).float()
        return input_data, label_data

    def __len__(self):
        return len(self.input_data)


if __name__ == "__main__":
    # https://github.com/tegg89/SRCNN-Tensorflow
    epoch = 25
    batch_size = 64
    image_size = 32
    label_size = 32
    learning_rate = 1e-5
    c_dim = 3
    scale = 3
    stride = 16
    checkpoint_dir = "checkpoint"
    sample_dir = "result"
    # is_train = True
    history = None

    tf.reset_default_graph()
    tf_config = tf.ConfigProto(log_device_placement=True)

    # 限制 GPU 使用量
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8

    srcnn = TfSrcnn(epoch=epoch,
                    image_size=image_size,
                    label_size=label_size,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    c_dim=c_dim,
                    scale=scale,
                    stride=stride,
                    checkpoint_dir=checkpoint_dir,
                    sample_dir=sample_dir)

    with tf.Session(config=tf_config) as sess:
        srcnn.setSess(sess)

        # print(srcnn.load())
        # srcnn.load2()
        # srcnn.train()

        # print(srcnn.load())
        data, result, merged_result, squeeze_result = srcnn.predict()
        # print("sess size:", sess.__sizeof__())
        # print("graph size:", sess.graph.__sizeof__())

    # x = [i for i in range(len(srcnn.training_history))]
    # plt.plot(x, srcnn.training_history)
    # plt.show()
