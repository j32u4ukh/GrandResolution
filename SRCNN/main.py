""" https://github.com/tegg89/SRCNN-Tensorflow
實作論文 Image Super-Resolution Using Deep Convolutional Networks
先透過 雙三次插值 (Bicubic interpolation) 令小圖放大至目標大小，
在透過卷積網路提高圖像品質。

目前以 SSIM 來衡量圖像品質，目前品質分數約在 0.837 ，完全一致為 1，完全不一致為 -1。
接下來可透過超參數調整、網路架構調整 與 放大誤差函數級距，進一步提高圖像品質。


流程
1. 插值處理：先對低分辨率圖像(LR)進行雙三次插值(Bicubic Interpolation)處理，得到和高分辨率圖像(SR)一樣大小的圖像作為輸入圖像(YY)。
2. 特征提取(Patch Extration and Representation)：對插值後圖像(YY)進行一次卷積操作(CNN)，
   目的是提取圖像特征，filters=[f1, f1, c, n1]。
3. 非線性映射(Non-linear Mapping)：對(2)提取的特征進行一次卷積操作，目的是對特征進行非線性變換，filters=[f2, f2, n1, n2]。
4. 重構(Reconstruction)：對2的非線性映射進行一次卷積操作，目的是進行圖像超分辨率重建，filters=[f3, f3, n2, c]。

# c 為圖片的通道數。
"""

import argparse
from SRCNN.model import SRCNN


def main():
    # 主要提供命令列使用，但還是更希望可以在"命令列"與"直接使用程式碼"之間無痛轉換
    parser = argparse.ArgumentParser()

    # parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    """
    image_size = 32
    scale = 3
    stride = 16
    lr = 0.01
    batch_size = 64
    model_save_step = 10
    resume_iters = 150
    is_gray = False
    idx=33
    """
    parser.add_argument("-size", type=int, default=32, help="切割用於訓練的圖片大小")
    parser.add_argument("-scale", type=int, default=3, help="放大倍數")
    parser.add_argument("-stride", type=int, default=32, help="卷積濾波器步長")
    parser.add_argument("-lr", type=float, default=0.01, help="學習率")
    parser.add_argument("-batch", type=int, default=64, help="每一次訓練的批次大小")
    parser.add_argument("-save", type=int, default=10, help="每訓練幾次儲存一次模型")
    parser.add_argument("-iters", type=int, default=0, help="累積訓練次數，用於載入之前儲存的模型以接續訓練")
    parser.add_argument("-gray", type=bool, default=False, help="是否為灰階圖片")
    parser.add_argument("-workers", type=int, default=0, help="使用核心數，超過 0 必須使用 Run 來執行，且須在 __main__ 當中"
                                                              "，但讀取數據較有效率")
    parser.add_argument("-idx", type=int, default=-1, help="測試圖片索引值，若為 -1 表示使用訓練集")
    parser.add_argument("-store", type=bool, default=False, help="是否儲存圖片")
    config = parser.parse_args()

    srcnn = SRCNN(image_size=config.size,
                  scale=config.scale,
                  stride=config.stride,
                  lr=config.lr,
                  batch_size=config.batch,
                  model_save_step=config.save,
                  resume_iters=config.iters,
                  is_gray=config.gray,
                  n_workers=config.workers)

    if config.idx == -1:
        srcnn.train(n_iter=config.iters)
    else:
        srcnn.predict(idx=config.idx, save_image=config.store)


if __name__ == "__main__":
    main()
