""" https://github.com/tegg89/SRCNN-Tensorflow
OpenEyes/SRCNN 資料夾，實作論文 Image Super-Resolution Using Deep Convolutional Networks
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

