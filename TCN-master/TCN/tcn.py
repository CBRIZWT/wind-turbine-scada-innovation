import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


def _make_activation(name: str) -> nn.Module:
    """构造内部激活函数 (Phase D 最优超参数: GELU 替代 ReLU)."""
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"未知激活函数: {name}")


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN 基础块.

    Phase D 最优超参数扩展 (2026-05-25):
    - inner_activation='gelu' 替代原 ReLU,允许负值激活,避免非负偏置注入
    - remove_final_relu=True 取消残差加和后的 ReLU,保持输出可正可负
    旧调用 (kernel_size, stride, dilation, padding, dropout) 行为完全等价原版。
    """

    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2,
        inner_activation: str = "relu",
        remove_final_relu: bool = False,
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.act1 = _make_activation(inner_activation)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.act2 = _make_activation(inner_activation)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.act1, self.dropout1,
                                 self.conv2, self.chomp2, self.act2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.remove_final_relu = remove_final_relu
        self.final_act = None if remove_final_relu else nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        merged = out + res
        if self.final_act is None:
            return merged                       # Phase D 最优: 残差可正可负
        return self.final_act(merged)           # 原版兼容: ReLU


class TemporalConvNet(nn.Module):
    """TCN 主体网络. 向后兼容 + Phase D 最优配置扩展.

    新增可选参数:
    - inner_activation='gelu' (默认 'relu' 保持原版行为)
    - remove_final_relu=True (默认 False 保持原版行为)
    """

    def __init__(
        self,
        num_inputs,
        num_channels,
        kernel_size=2,
        dropout=0.2,
        inner_activation: str = "relu",
        remove_final_relu: bool = False,
    ):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size,
                dropout=dropout,
                inner_activation=inner_activation,
                remove_final_relu=remove_final_relu,
            )]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
