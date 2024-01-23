import torch  # 导入torch库
from .quant_layer import QuantModule  # 从当前目录下的quant_layer模块导入QuantModule类
from .quant_block import BaseQuantBlock  # 从当前目录下的quant_block模块导入BaseQuantBlock类
from .quant_model import QuantModel  # 从当前目录下的quant_model模块导入QuantModel类
from typing import Union  # 导入typing模块的Union，用于表示多种类型中的一种


def set_act_quantize_params(module: Union[QuantModel, QuantModule, BaseQuantBlock],
                            cali_data, batch_size: int = 256):
    '''
    设置或初始化量化模型、模块或块的激活量化参数
    '''
    module.set_quant_state(True, True)  # 设置模型的量化状态为True

    # 遍历模型中的所有模块
    for t in module.modules():
        # 如果模块是QuantModule或BaseQuantBlock类型
        if isinstance(t, (QuantModule, BaseQuantBlock)):
            t.act_quantizer.set_inited(False)  # 设置激活量化器的初始化状态为False

    '''set or init step size and zero point in the activation quantizer'''
    # 获取批处理大小和校准数据大小的较小值
    batch_size = min(batch_size, cali_data.size(0))
    # 禁止torch进行梯度计算
    with torch.no_grad():
        # 遍历校准数据，每次处理一个批次的数据
        for i in range(int(cali_data.size(0) / batch_size)):
            # 将校准数据的一个批次移动到GPU上，并传入模型进行处理
            module(cali_data[i * batch_size:(i + 1) * batch_size].cuda())
    # 清空GPU缓存
    torch.cuda.empty_cache()

    # 再次遍历模型中的所有模块
    for t in module.modules():
        # 如果模块是QuantModule或BaseQuantBlock类型
        if isinstance(t, (QuantModule, BaseQuantBlock)):
            t.act_quantizer.set_inited(True)  # 设置激活量化的初始化状态为True