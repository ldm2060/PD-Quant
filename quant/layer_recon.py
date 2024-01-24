import torch
import torch.nn.functional as F
from .quant_layer import QuantModule, lp_loss
from .quant_model import QuantModel
from .block_recon import LinearTempDecay
from .adaptive_rounding import AdaRoundQuantizer
from .set_weight_quantize_params import get_init, get_dc_fp_init
from .set_act_quantize_params import set_act_quantize_params
from .quant_block import BaseQuantBlock, specials_unquantized

include = False
def find_unquantized_module(model: torch.nn.Module, module_list: list = [], name_list: list = []):
    '''在给定的模型中查找并存储未量化的模块'''
    #Store subsequent unquantized modules in a list
    global include #标记是否需要将模块添加到列表中
    for name, module in model.named_children():
        if isinstance(module, (QuantModule, BaseQuantBlock)): #是QuantModule或BaseQuantBlock的实例
            if not module.trained: #尚未训练
                include = True
                module.set_quant_state(False,False) #模块的量化状态设置为False
                name_list.append(name)
                module_list.append(module)
        elif include and type(module) in specials_unquantized: #include为True，并且模块的类型在specials_unquantized列表中
            name_list.append(name)
            module_list.append(module)
        else: 
            find_unquantized_module(module, module_list, name_list) #对子模块进行递归调用
    return module_list[1:], name_list[1:] #返回包含模块和模块名称的列表，但是排除了列表的第一个元素。这是因为在递归调用中，列表的第一个元素可能会被重复添加

def layer_reconstruction(model: QuantModel, fp_model: QuantModel, layer: QuantModule, fp_layer: QuantModule,
                        cali_data: torch.Tensor,batch_size: int = 32, iters: int = 20000, weight: float = 0.001,
                        opt_mode: str = 'mse', b_range: tuple = (20, 2),
                        warmup: float = 0.0, p: float = 2.0, lr: float = 4e-5, input_prob: float = 1.0, 
                        keep_gpu: bool = True, lamb_r: float = 0.2, T: float = 7.0, bn_lr: float = 1e-3, lamb_c=0.02):
    """
    用于重构量化模块的函数，用于AdaRound中的优化

    :param model: QuantModel 模型对象，用于量化的模型
    :param fp_model: QuantModel FP模型对象，用于参考
    :param layer: QuantModule 需要优化的量化模块
    :param fp_layer: QuantModule FP模块对象，用于参考
    :param cali_data: torch.Tensor 用于校准的数据，通常是1024个训练图像，如AdaRound中所述
    :param batch_size: int 重构的小批量大小，默认为32
    :param iters: int 重构的优化迭代次数，默认为20000
    :param weight: float 舍入正则化项的权重，默认为0.001
    :param opt_mode: str 优化模式，默认为'mse'
    :param b_range: tuple 温度范围，默认为(20, 2)
    :param warmup: float 温度调度的迭代比例，默认为0.0
    :param p: float L_p范数最小化，默认为2.0
    :param lr: float 激活函数学习率，默认为4e-5
    :param input_prob: float 输入概率，默认为1.0
    :param keep_gpu: bool 是否保持在GPU上，默认为True
    :param lamb_r: float 正则化的超参数，默认为0.2
    :param T: float KL散度的温度系数，默认为7.0
    :param bn_lr: float BN学习率，默认为1e-3
    :param lamb_c: float DC的超参数，默认为0.02
    """

    '''get input and set scale'''
    # 获取输入并设置缩放
    cached_inps = get_init(model, layer, cali_data, batch_size=batch_size,
                                        input_prob=True, keep_gpu=keep_gpu) #获取并保存模型在给定层级的输入数据
    cached_outs, cached_output, cur_syms = get_dc_fp_init(fp_model, fp_layer, cali_data, batch_size=batch_size,
                                        input_prob=True, keep_gpu=keep_gpu, bn_lr=bn_lr, lamb=lamb_c) #获取并保存模型在给定层级的直流分量数据
    set_act_quantize_params(layer, cali_data=cached_inps[:min(256, cached_inps.size(0))]) #设置或初始化激活量化器中的步长和零点

    '''set state'''
    # 设置状态
    cur_weight, cur_act = True, True
    
    global include
    module_list, name_list, include = [], [], False
    module_list, name_list = find_unquantized_module(model, module_list, name_list)
    layer.set_quant_state(cur_weight, cur_act)
    for para in model.parameters():
        para.requires_grad = False

    '''set quantizer'''
    # 设置量化器
    round_mode = 'learned_hard_sigmoid'
    w_para, a_para = [], []
    w_opt, a_opt = None, None
    scheduler, a_scheduler = None, None

    '''weight'''
    # 权重量化器
    layer.weight_quantizer = AdaRoundQuantizer(uaq=layer.weight_quantizer, round_mode=round_mode,
                                               weight_tensor=layer.org_weight.data)
    layer.weight_quantizer.soft_targets = True
    w_para += [layer.weight_quantizer.alpha]

    '''activation'''
    # 激活量化器
    if layer.act_quantizer.delta is not None:
        layer.act_quantizer.delta = torch.nn.Parameter(torch.tensor(layer.act_quantizer.delta))
        a_para += [layer.act_quantizer.delta]
    '''set up drop'''
    # 设置dropout
    layer.act_quantizer.is_training = True

    if len(w_para) != 0:
        w_opt = torch.optim.Adam(w_para, lr=3e-3)
    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=lr)
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=iters, eta_min=0.)
    
    loss_mode = 'relaxation'
    rec_loss = opt_mode
    loss_func = LossFunction(layer, round_loss=loss_mode, weight=weight,
                             max_count=iters, rec_loss=rec_loss, b_range=b_range,
                             decay_start=0, warmup=warmup, p=p, lam=lamb_r, T=T)
    device = 'cuda'
    sz = cached_inps.size(0)
    for i in range(iters):
        idx = torch.randint(0, sz, (batch_size,))
        cur_inp = cached_inps[idx].to(device)
        cur_sym = cur_syms[idx].to(device)
        output_fp = cached_output[idx].to(device)
        cur_out = cached_outs[idx].to(device)
        if input_prob < 1.0:
            drop_inp = torch.where(torch.rand_like(cur_inp) < input_prob, cur_inp, cur_sym)
        
        cur_inp = torch.cat((drop_inp, cur_inp))
        
        if w_opt:
            w_opt.zero_grad()
        if a_opt:
            a_opt.zero_grad()
        out_all = layer(cur_inp)
        
        '''forward for prediction difference'''
        # 用于预测差异的前向传播
        out_drop = out_all[:batch_size]
        out_quant = out_all[batch_size:]
        output = out_quant
        for num, module in enumerate(module_list):
            # 对于ResNet和RegNet
            if name_list[num] == 'fc':
                output = torch.flatten(output, 1)
            # 对于MobileNet和MNasNet
            if isinstance(module, torch.nn.Dropout):
                output = output.mean([2, 3])
            output = module(output)
        err = loss_func(out_drop, cur_out, output, output_fp)

        err.backward(retain_graph=True)
        if w_opt:
            w_opt.step()
        if a_opt:
            a_opt.step()
        if scheduler:
            scheduler.step()
        if a_scheduler:
            a_scheduler.step()
    torch.cuda.empty_cache()

    layer.weight_quantizer.soft_targets = False#标记权重量化的软目标为False
    layer.act_quantizer.is_training = False#标记激活量化模块不在训练
    layer.trained = True#标记模块已经训练过了


class LossFunction:
    def __init__(self,
                 layer: QuantModule,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.,
                 lam: float = 1.0,
                 T: float = 7.0):
        """
        初始化LossFunction对象。

        :param layer: 量化模块。
        :param round_loss: 舍入损失的类型。默认为'relaxation'。
        :param weight: 舍入损失的权重。默认为1.0。
        :param rec_loss: 重构损失的类型。默认为'mse'。
        :param max_count: 最大计数。默认为2000。
        :param b_range: b值的范围。默认为(10, 2)。
        :param decay_start: 衰减开始值。默认为0.0。
        :param warmup: 热身值。默认为0.0。
        :param p: p值。默认为2.0。
        :param lam: lam值。默认为1.0。
        :param T: T值。默认为7.0。
        """
        self.layer = layer
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.lam = lam
        self.T = T

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])#温度衰减
        self.count = 0
        self.pd_loss = torch.nn.KLDivLoss(reduction='batchmean')#KL散度损失

    def __call__(self, pred, tgt, output, output_fp):
        """
        计算自适应舍入的总损失：
        rec_loss是二次输出重构损失，round_loss是优化舍入策略的正则化项，pd_loss是预测差异损失。

        :param pred: 量化模型的输出
        :param tgt: FP模型的输出
        :param output: 量化模型的预测
        :param output_fp: FP模型的预测
        :return: 总损失函数
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)#均方误差损失
        else:
            raise ValueError('不支持的重构损失函数：{}'.format(self.rec_loss))

        pd_loss = self.pd_loss(F.log_softmax(output / self.T, dim=1), F.softmax(output_fp / self.T, dim=1)) / self.lam #KL散度损失

        b = self.temp_decay(self.count) #温度衰减
        if self.count < self.loss_start or self.round_loss == 'none':#如果计数小于损失开始值或者舍入损失为none
            b = round_loss = 0#舍入损失为0
        elif self.round_loss == 'relaxation':#如果舍入损失为relaxation
            round_loss = 0#舍入损失为0
            round_vals = self.layer.weight_quantizer.get_soft_targets()#获取量化器的软目标
            round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()#计算舍入损失
        else:
            raise NotImplementedError
        total_loss = rec_loss + round_loss + pd_loss#总损失
        if self.count % 500 == 0:
            print('总损失：\t{:.3f} (重构损失:{:.3f}, 预测差异损失:{:.3f}, 舍入损失:{:.3f})\tb={:.2f}\t计数={}'.format(
                float(total_loss), float(rec_loss), float(pd_loss), float(round_loss), b, self.count))
        return total_loss
