import torch
from torch.utils import data as tdata
from convcnp.dataset import Synthetic1D


class MixtureGPDataset(tdata.Dataset):
    def __init__(self, kernels=['eq', 'matern', 'periodic'], train=True, **kwargs):
        """
        Args:
            kernels (list): A list of kernel names to sample from.
            train (bool): Training or testing mode.
            **kwargs: Other arguments to pass to the underlying Synthetic1D dataset.
        """
        super().__init__()
        self.train = train
        self.kernels = kernels
        self.num_kernels = len(kernels)
        self.num_total_max = 100
        # 为每种核创建一个独立的Synthetic1D实例
        # 它们共享大部分参数，但有不同的核
        self.datasets = {
            kernel: Synthetic1D(kernel=kernel, train=train, **kwargs)
            for kernel in self.kernels
        }

        # 训练时，我们希望有很多样本
        # 测试时，我们可能只需要每个核生成一些固定的样本
        if self.train:
            self.length = 256 * self.num_kernels  # 可以设置一个较大的值
        else:
            self.length = 100  # 例如，测试时总共生成100个样本

    def __len__(self):
        return self.length

    def sample(self, num_context, num_target):
        """
        这个方法现在是DataLoader的入口点。
        它接收由"猴子补丁"生成的、对整个批次统一的num_context和num_target。
        """
        # 1. 随机选择一个核
        kernel_idx = torch.randint(0, self.num_kernels, size=()).item()
        selected_kernel = self.kernels[kernel_idx]

        # 2. 从对应核的数据集实例中，使用给定的参数采样一个函数
        # 直接调用其 .sample() 方法
        context_x, context_y, target_x, target_y = self.datasets[selected_kernel].sample(num_context, num_target)

        # 3. 返回数据以及用于标签的核索引
        return context_x, context_y, target_x, target_y #, torch.tensor(kernel_idx)
    def __getitem__(self, index):
        # 1. 随机选择一个核
        num_context = torch.randint(3, self.datasets[self.kernels[0]].num_total_max, size=())
        num_target = torch.randint(3, self.datasets[self.kernels[0]].num_total_max, size=())
        return self.sample(num_context, num_target)
