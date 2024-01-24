import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from .quant_layer import QuantModule, Union, lp_loss
from .quant_model import QuantModel
from .quant_block import BaseQuantBlock
from tqdm import trange


def save_dc_fp_data(model: QuantModel, layer: Union[QuantModule, BaseQuantBlock], cali_data: torch.Tensor,
                    batch_size: int = 32, keep_gpu: bool = True,
                    input_prob: bool = False, lamb=50, bn_lr=1e-3):
    """
    保存数据校正后的激活值。
    
    :param model: 量化模型。
    :param layer: 需要校正的层。
    :param cali_data: 校正数据。
    :param batch_size: 批大小。默认为32。
    :param keep_gpu: 是否保留在GPU上。默认为True。
    :param input_prob: 是否使用输入概率。默认为False。
    :param lamb: 校正参数。默认为50。
    :param bn_lr: BatchNorm层的学习率。默认为1e-3。

    :return:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 校正后的激活值。
            - cached_outs (torch.Tensor): 校正后的输出。
            - cached_outputs (torch.Tensor): 校正前的输出。
            - cached_sym (torch.Tensor): 输入概率。
    """
    device = next(model.parameters()).device
    get_inp_out = GetDcFpLayerInpOut(model, layer, device=device, input_prob=input_prob, lamb=lamb, bn_lr=bn_lr)
    cached_batches = []

    print("开始校正{}批次的数据！".format(int(cali_data.size(0) / batch_size)))
    for i in trange(int(cali_data.size(0) / batch_size)):
        if input_prob:
            cur_out, out_fp, cur_sym = get_inp_out(cali_data[i * batch_size:(i + 1) * batch_size])#获取校正后的输出、校正前的输出和输入概率
            cached_batches.append((cur_out.cpu(), out_fp.cpu(), cur_sym.cpu()))#将校正后的输出、校正前的输出和输入概率保存到cached_batches中
        else:
            cur_out, out_fp = get_inp_out(cali_data[i * batch_size:(i + 1) * batch_size])#获取校正后的输出和校正前的输出
            cached_batches.append((cur_out.cpu(), out_fp.cpu()))#将校正后的输出和校正前的输出保存到cached_batches中
    cached_outs = torch.cat([x[0] for x in cached_batches])
    cached_outputs = torch.cat([x[1] for x in cached_batches])
    if input_prob:
        cached_sym = torch.cat([x[2] for x in cached_batches])
    torch.cuda.empty_cache()#清空CUDA缓存
    if keep_gpu:
        cached_outs = cached_outs.to(device)
        cached_outputs = cached_outputs.to(device)
        if input_prob:
            cached_sym = cached_sym.to(device)
    if input_prob:
        cached_outs.requires_grad = False
        cached_sym.requires_grad = False#设置梯度为False
        return cached_outs, cached_outputs, cached_sym
    return cached_outs, cached_outputs


def save_inp_oup_data(model: QuantModel, layer: Union[QuantModule, BaseQuantBlock], cali_data: torch.Tensor,
                      batch_size: int = 32, keep_gpu: bool = True,
                      input_prob: bool = False):
    """
    保存在校准数据集上的特定层/块的输入数据和输出数据。

    :param model: QuantModel
    :param layer: QuantModule or QuantBlock
    :param cali_data: 校准数据集
    :param weight_quant: 权重量化
    :param act_quant: 激活量化
    :param batch_size: 用于校准的小批量大小
    :param keep_gpu: 将保存的数据放在 GPU 上以加快优化速度
    :return: 输入输出数据
    """
    device = next(model.parameters()).device
    get_inp_out = GetLayerInpOut(model, layer, device=device, input_prob=input_prob)#获取模型在给定层级的输入数据
    cached_batches = []

    for i in range(int(cali_data.size(0) / batch_size)):
        cur_inp = get_inp_out(cali_data[i * batch_size:(i + 1) * batch_size])#获取当前批次的输入数据
        cached_batches.append(cur_inp.cpu())#将当前批次的输入数据保存到cached_batches中
    cached_inps = torch.cat([x for x in cached_batches])#将cached_batches中的数据拼接起来
    torch.cuda.empty_cache()#清空CUDA缓存
    if keep_gpu:
        cached_inps = cached_inps.to(device)#将cached_inps放在GPU上

    return cached_inps


class StopForwardException(Exception):
    """
    用于抛出和捕获异常以停止遍历图
    """
    pass


class DataSaverHook:
    """
    存储块输入和输出的前向钩子
    """

    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input#是否保存输入
        self.store_output = store_output#是否保存输出
        self.stop_forward = stop_forward#是否停止遍历图

        self.input_store = None#输入数据
        self.output_store = None#输出数据

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


class input_hook(object):
    """
	前向钩子，用于获取中间层的输出
	"""
    def __init__(self, stop_forward=False):
        super(input_hook, self).__init__()
        self.inputs = None

    def hook(self, module, input, output):
        self.inputs = input

    def clear(self):
        self.inputs = None

class GetLayerInpOut:
    """
    获取并存储模型中特定层的输入数据
    """
    def __init__(self, model: QuantModel, layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device, input_prob: bool = False):
        """
        初始化GetLayerInpOut类的实例。

        参数：
        - model: QuantModel类型的模型对象。
        - layer: QuantModule或BaseQuantBlock类型的层对象。
        - device: torch.device类型的设备对象。
        - input_prob: 是否存储输入概率，默认为False。
        """
        self.model = model
        self.layer = layer
        self.device = device
        self.data_saver = DataSaverHook(store_input=True, store_output=False, stop_forward=True)
        self.input_prob = input_prob

    def __call__(self, model_input):
        """
        调用GetLayerInpOut实例。

        参数：
        - model_input: 输入模型的数据。

        返回：
        - 存储的输入数据。
        """
        handle = self.layer.register_forward_hook(self.data_saver)#注册前向钩子
        with torch.no_grad():#不计算梯度
            self.model.set_quant_state(weight_quant=True, act_quant=True)#设置模型的量化状态
            try:
                _ = self.model(model_input.to(self.device))#将输入数据放在GPU上
            except StopForwardException:
                pass#捕获异常

        handle.remove()#移除钩子

        return self.data_saver.input_store[0].detach()#返回存储的输入数据

class GetDcFpLayerInpOut:
    def __init__(self, model: QuantModel, layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device, input_prob: bool = False, lamb=50, bn_lr=1e-3):
        """
        初始化GetDcFpLayerInpOut类的实例。

        参数：
        - model: QuantModel类型的模型对象。
        - layer: QuantModule或BaseQuantBlock类型的层对象。
        - device: torch.device类型的设备对象。
        - input_prob: 是否存储输入概率，默认为False。
        - lamb: 约束损失的lambda值，默认为50。
        - bn_lr: 优化器的学习率，默认为1e-3。
        """
        self.model = model
        self.layer = layer
        self.device = device
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=False)
        self.input_prob = input_prob
        self.bn_stats = []
        self.eps = 1e-6 
        self.lamb=lamb 
        self.bn_lr=bn_lr
        for n, m in self.layer.named_modules():#遍历层中的所有模块
            if isinstance(m, nn.BatchNorm2d):
                # 获取BatchNorm层中的统计数据
                self.bn_stats.append(
                    (m.running_mean.detach().clone().flatten().cuda(),#均值
                    torch.sqrt(m.running_var +#方差
                                self.eps).detach().clone().flatten().cuda()))#标准差
    
    def own_loss(self, A, B):
        """
        计算两个张量之间的均方损失。

        参数：
        - A: 第一个张量。
        - B: 第二个张量。

        返回：
        - 两个张量之间的均方损失。
        """
        return (A - B).norm()**2 / B.size(0)
    
    def relative_loss(self, A, B):
        """
        计算两个张量之间的相对损失。

        参数：
        - A: 第一个张量。
        - B: 第二个张量。

        返回：
        - 两个张量之间的相对损失。
        """
        return (A-B).abs().mean()/A.abs().mean()

    def __call__(self, model_input):
        """
        对层进行分析。

        参数：
        - model_input: 模型的输入。

        返回：
        - 包含层的输出、模型的输出和模型的输入的元组。
        """
        self.model.set_quant_state(False, False)
        handle = self.layer.register_forward_hook(self.data_saver)
        hooks = []
        hook_handles = []
        for name, module in self.layer.named_modules():#遍历层中的所有模块
            if isinstance(module, nn.BatchNorm2d):#如果模块是BatchNorm2d类型
                hook = input_hook()
                hooks.append(hook)#将hook添加到hooks中
                hook_handles.append(module.register_forward_hook(hook.hook))
        assert len(hooks) == len(self.bn_stats)#断言hooks和bn_stats的长度相等

        with torch.no_grad():#不计算梯度
            try:
                output_fp = self.model(model_input.to(self.device))#将输入数据放在GPU上
            except StopForwardException:
                pass
            if self.input_prob:#如果使用输入概率
                input_sym = self.data_saver.input_store[0].detach()#获取存储的输入数据
            
        handle.remove()#移除钩子
        para_input = input_sym.data.clone()#克隆输入数据
        para_input = para_input.to(self.device)#将输入数据放在GPU上
        para_input.requires_grad = True#设置梯度为True
        optimizer = optim.Adam([para_input], lr=self.bn_lr)#设置优化器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        min_lr=1e-5,
                                                        verbose=False,
                                                        patience=100)#设置学习率衰减策略
        iters=500#迭代次数
        for iter in range(iters):
            self.layer.zero_grad()#梯度清零
            optimizer.zero_grad()
            for hook in hooks:
                hook.clear()#清除hook
            _ = self.layer(para_input)#将输入数据放在GPU上
            mean_loss = 0#均值损失
            std_loss = 0#标准差损失
            for num, (bn_stat, hook) in enumerate(zip(self.bn_stats, hooks)):#遍历bn_stats和hooks
                tmp_input = hook.inputs[0]#获取hook的输入数据
                bn_mean, bn_std = bn_stat[0], bn_stat[1]#获取均值和标准差
                tmp_mean = torch.mean(tmp_input.view(tmp_input.size(0),
                                                    tmp_input.size(1), -1),
                                    dim=2)#计算均值
                tmp_std = torch.sqrt(
                    torch.var(tmp_input.view(tmp_input.size(0),
                                            tmp_input.size(1), -1),
                            dim=2) + self.eps)#计算标准差
                mean_loss += self.own_loss(bn_mean, tmp_mean)#计算均值损失
                std_loss += self.own_loss(bn_std, tmp_std)# 计算标准差损失
            constraint_loss = lp_loss(para_input, input_sym) / self.lamb#计算PD损失
            total_loss = mean_loss + std_loss + constraint_loss #计算总损失
            total_loss.backward()#反向传播
            optimizer.step()#优化器更新
            scheduler.step(total_loss.item())#学习率衰减
            # if (iter+1) % 500 == 0:
            #     print('Total loss:\t{:.3f} (mse:{:.3f}, mean:{:.3f}, std:{:.3f})\tcount={}'.format(
            #     float(total_loss), float(constraint_loss), float(mean_loss), float(std_loss), iter))
                
        with torch.no_grad():#不计算梯度
            out_fp = self.layer(para_input)#将输入数据放在GPU上

        if self.input_prob:#如果使用输入概率
            return  out_fp.detach(), output_fp.detach(), para_input.detach()#返回输出、模型输出和模型输入
        return out_fp.detach(), output_fp.detach()#返回输出和模型输出