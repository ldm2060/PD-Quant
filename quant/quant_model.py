import torch.nn as nn
from .quant_block import specials, BaseQuantBlock
from .quant_layer import QuantModule, StraightThrough, UniformAffineQuantizer
from .fold_bn import search_fold_and_remove_bn


class QuantModel(nn.Module):
    # 初始化函数，接收原始模型，权重量化参数，激活量化参数，是否融合BN层
    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, is_fusing=True):
        super().__init__()
        if is_fusing:# 如果需要融合BN层
            search_fold_and_remove_bn(model)# 搜索并移除BN层
            self.model = model
            self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)# 重构量化模块
        else:
            self.model = model
            self.quant_module_refactor_wo_fuse(self.model, weight_quant_params, act_quant_params)# 不融合BN层的情况下重构量化模块

    # 递归地将普通的卷积层和全连接层替换为量化模块
    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, weight_quant_params, act_quant_params))
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)): # 如果子模块是卷积层或全连接层，替换为量化模块
                setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):# 如果子模块是ReLU或ReLU6
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module # 设置激活函数
                    setattr(module, name, StraightThrough()) # 替换为直通模块
                else:
                    continue
            # 如果子模块是直通模块，继续下一轮循环
            elif isinstance(child_module, StraightThrough):
                continue

            else:# 对其他类型的子模块递归调用此函数
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)
    # 不融合BN层的情况下，递归地将普通的卷积层和全连接层替换为量化模块
    def quant_module_refactor_wo_fuse(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, weight_quant_params, act_quant_params))
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, nn.BatchNorm2d):
                if prev_quantmodule is not None:
                    # 设置归一化函数
                    prev_quantmodule.norm_function = child_module
                    # 替换为直通模块
                    setattr(module, name, StraightThrough())
                else:
                    continue

            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue

            elif isinstance(child_module, StraightThrough):
                continue

            else: # 对其他类型的子模块递归调用此函数
                self.quant_module_refactor_wo_fuse(child_module, weight_quant_params, act_quant_params)
    # 设置量化状态，包括权重量化和激活量化
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    # 前向传播函数
    def forward(self, input):
        return self.model(input)

    # 将第一层和最后一层设置为8位
    def set_first_last_layer_to_8bit(self):
        w_list, a_list = [], []
        for module in self.model.modules():
            if isinstance(module, UniformAffineQuantizer):
                if module.leaf_param:
                    a_list.append(module)
                else:
                    w_list.append(module)
        w_list[0].bitwidth_refactor(8)
        w_list[-1].bitwidth_refactor(8)
        'the image input has been in 0~255, set the last layer\'s input to 8-bit'
        a_list[-2].bitwidth_refactor(8)
        # a_list[0].bitwidth_refactor(8)

    # 禁用网络输出的量化
    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[-1].disable_act_quant = True
