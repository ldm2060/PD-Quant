# 导入QuantModule模块，这是一个用于量化的模块
from .quant_layer import QuantModule
# 导入save_inp_oup_data和save_dc_fp_data函数，这些函数用于保存输入输出数据和直流分量数据
from .data_utils import save_inp_oup_data, save_dc_fp_data

# 定义get_init函数，用于获取初始化的输入数据
def get_init(model, block, cali_data, batch_size, input_prob: bool = False, keep_gpu: bool=True):
    # 调用save_inp_oup_data函数保存输入输出数据，并返回
    cached_inps = save_inp_oup_data(model, block, cali_data, batch_size, input_prob=input_prob, keep_gpu=keep_gpu)
    return cached_inps


def get_dc_fp_init(model, block, cali_data, batch_size, input_prob: bool = False, keep_gpu: bool=True, lamb=50, bn_lr=1e-3):
    '''
    定义get_dc_fp_init函数，用于获取初始化的直流分量数据
    '''
    # 调用save_dc_fp_data函数保存直流分量数据，并返回
    cached_outs, cached_outputs, cached_sym = save_dc_fp_data(model, block, cali_data, batch_size, input_prob=input_prob, keep_gpu=keep_gpu, lamb=lamb, bn_lr=bn_lr)
    return cached_outs, cached_outputs, cached_sym

# 定义set_weight_quantize_params函数，用于设置模型权重的量化参数
def set_weight_quantize_params(model):
    # 遍历模型中的所有模块
    for module in model.modules():
        # 如果模块是QuantModule类型
        if isinstance(module, QuantModule):
            # 设置权重量化器的初始化状态为False
            module.weight_quantizer.set_inited(False)
            # 计算权重量化器的步长和零点
            module.weight_quantizer(module.weight)
            # 设置权重量化器的初始化状态为True
            module.weight_quantizer.set_inited(True)

# 定义save_quantized_weight函数，用于保存量化后的权重
def save_quantized_weight(model):
    # 遍历模型中的所有模块
    for module in model.modules():
        # 如果模块是QuantModule类型
        if isinstance(module, QuantModule):
            # 保存量化后的权重
            module.weight.data = module.weight_quantizer(module.weight)
