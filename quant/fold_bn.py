import torch
import torch.nn as nn
import torch.nn.init as init


class StraightThrough(nn.Module):
    def __int__(self):
        super().__init__()

    def forward(self, input):
        return input


def _fold_bn(conv_module, bn_module):
    """
    将批归一化层（BN）折叠到卷积层（Conv）的函数

    Args:
        conv_module (nn.Module): 卷积层模块
        bn_module (nn.Module): 批归一化层模块

    Returns:
        torch.Tensor: 折叠后的权重
        torch.Tensor: 折叠后的偏置
    """
    w = conv_module.weight.data
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    safe_std = torch.sqrt(y_var + bn_module.eps)
    w_view = (conv_module.out_channels, 1, 1, 1)
    if bn_module.affine:
        # 计算折叠后的权重和偏置
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta

    return weight, bias


def fold_bn_into_conv(conv_module, bn_module):
    # 将折叠后的权重和偏置应用到卷积层
    w, b = _fold_bn(conv_module, bn_module)
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b)
    else:
        conv_module.bias.data = b
    conv_module.weight.data = w
    # 设置BN层的运行统计信息
    bn_module.running_mean = bn_module.bias.data
    bn_module.running_var = bn_module.weight.data ** 2


def reset_bn(module: nn.BatchNorm2d):
    # 重置BN层的参数
    if module.track_running_stats:
        module.running_mean.zero_()
        module.running_var.fill_(1-module.eps)
        # 不重置追踪的批次数
        # self.num_batches_tracked.zero_()
    if module.affine:
        init.ones_(module.weight)
        init.zeros_(module.bias)


def is_bn(m):
    # 判断是否为BN层
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_absorbing(m):
    # 判断是否为吸收层（卷积层或线性层）
    return (isinstance(m, nn.Conv2d)) or isinstance(m, nn.Linear)


def search_fold_and_remove_bn(model):
    # 搜索并折叠BN层到卷积层，并将BN层替换为StraightThrough层
    model.eval()
    prev = None
    for n, m in model.named_children():
        if is_bn(m) and is_absorbing(prev):
            fold_bn_into_conv(prev, m)
            # 将BN层替换为StraightThrough层
            setattr(model, n, StraightThrough())
        elif is_absorbing(m):
            prev = m
        else:
            prev = search_fold_and_remove_bn(m)
    return prev


def search_fold_and_reset_bn(model):
    # 搜索并折叠BN层到卷积层，并重置BN层的参数
    model.eval()
    prev = None
    for n, m in model.named_children():
        if is_bn(m) and is_absorbing(prev):
            fold_bn_into_conv(prev, m)
            # 重置BN层的参数
            # reset_bn(m)
        else:
            search_fold_and_reset_bn(m)
        prev = m
