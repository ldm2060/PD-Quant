import torch
from torch import nn
from .quant_layer import UniformAffineQuantizer, round_ste


class AdaRoundQuantizer(nn.Module):
    """
    自适应舍入量化器，用于通过重构中间输出来优化舍入策略
    基于 Up or Down? Adaptive Rounding for Post-Training Quantization: https://arxiv.org/abs/2004.10568

    :param uaq: UniformAffineQuantizer，用于初始化该量化器的量化参数
    :param round_mode: 控制该量化器的前向传播方式
    :param weight_tensor: 初始化 alpha
    """

    def __init__(self, uaq: UniformAffineQuantizer, weight_tensor: torch.Tensor, round_mode='learned_round_sigmoid'):
        super(AdaRoundQuantizer, self).__init__()
        # 从 UniformAffineQuantizer 复制所有属性
        self.n_bits = uaq.n_bits
        self.sym = uaq.sym
        self.delta = uaq.delta
        self.zero_point = uaq.zero_point
        self.n_levels = uaq.n_levels

        self.round_mode = round_mode
        self.alpha = None
        self.soft_targets = False

        # sigmoid 函数的参数
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3
        self.init_alpha(x=weight_tensor.clone())

    def forward(self, x):
        if self.round_mode == 'nearest':
            x_int = torch.round(x / self.delta)  # 最近舍入
        elif self.round_mode == 'nearest_ste':
            x_int = round_ste(x / self.delta)  # 最近舍入（STE）
        elif self.round_mode == 'stochastic':
            x_floor = torch.floor(x / self.delta)
            rest = (x / self.delta) - x_floor  # 舍入的余数
            x_int = x_floor + torch.bernoulli(rest)  # 随机舍入
            print('Draw stochastic sample')  # 绘制随机样本
        elif self.round_mode == 'learned_hard_sigmoid':
            x_floor = torch.floor(x / self.delta)
            if self.soft_targets:
                x_int = x_floor + self.get_soft_targets()  # 使用软目标
            else:
                x_int = x_floor + (self.alpha >= 0).float()  # 使用硬目标
        else:
            raise ValueError('Wrong rounding mode')  # 舍入模式错误

        x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)  # 量化
        x_float_q = (x_quant - self.zero_point) * self.delta  # 反量化

        return x_float_q

    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)  # 获取软目标

    def init_alpha(self, x: torch.Tensor):
        x_floor = torch.floor(x / self.delta)
        if self.round_mode == 'learned_hard_sigmoid':
            print('Init alpha to be FP32')  # 初始化 alpha 为 FP32
            rest = (x / self.delta) - x_floor  # 舍入的余数 [0, 1)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
            self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError

    @torch.jit.export
        def extra_repr(self):
            """
            返回一个字符串，表示对象的额外信息。

            Returns:
                str: 表示对象位数的字符串
            """
            return 'bit={}'.format(self.n_bits)
