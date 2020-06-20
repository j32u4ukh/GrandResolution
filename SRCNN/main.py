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
import numpy as np
import torch.nn as nn
import torch
from GrandResolution.SRCNN.srcnn import (
    inputSetup,
    readData,
    mergeImages
)
from GrandResolution.utils import (
    showImage
)
from Xu3.network import printNetwork


class SRCNN:
    def __init__(self, image_size, scale, stride, lr, is_gray=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        self.scale = scale
        self.stride = stride
        if is_gray:
            self.c_dim = 1
        else:
            self.c_dim = 3

        self.data_loader = None
        self.ps = PtSrcnn(c_dim)
        self.ps.to(self.device)

        # torch.optim.Adam
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.optimizer = torch.optim.Adam(self.ps.parameters(), self.lr, (self.beta1, self.beta2))

    def train(self):
        self.data_loader = SrcnnDataLoader(-1, self.image_size, self.scale, self.stride, self.is_gray)

    # TODO: 模型可儲存，若已有儲存過的檔案，則將模型載入
    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))

        # 根據訓練次數，形成數據檔案名稱
        path_g = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        path_d = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))

        # 匯入之前訓練到一半的模型
        self.G.load_state_dict(torch.load(path_g, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(path_d, map_location=lambda storage, loc: storage))


class PtSrcnn(nn.Module):
    def __init__(self, c_dim):
        super().__init__()
        self.c_dim = c_dim

        # 'SAME' padding = (kernel_size - 1) / 2
        self.conv1 = nn.Conv2d(self.c_dim, 64, kernel_size=9, stride=1, padding=4, bias=True)
        self.relu1 = nn.ReLU()
        self.layer1 = None
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu2 = nn.ReLU()
        self.layer2 = None
        self.conv3 = nn.Conv2d(32, self.c_dim, kernel_size=5, stride=1, padding=2, bias=True)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        self.layer1 = self.relu1(self.conv1(x))
        self.layer2 = self.relu2(self.conv2(self.layer1))
        output = self.relu3(self.conv3(self.layer2))
        return output


if __name__ == "__main__":
    matrix = np.random.rand(1, 3, 30, 30)
    print("matrix:", matrix.shape)
    ps = PtSrcnn(3)
    mat = torch.from_numpy(matrix)
    mat = mat.float()
    y = ps(mat)
    print("y:", y.shape)
    # printNetwork(ps, "PtSrcnn")
