import torch
import torch.nn.functional as F
from .quant_layer import QuantModule, lp_loss
from .quant_model import QuantModel
from .quant_block import BaseQuantBlock, specials_unquantized
from .adaptive_rounding import AdaRoundQuantizer
from .set_weight_quantize_params import get_init, get_dc_fp_init
from .set_act_quantize_params import set_act_quantize_params

include = False
def find_unquantized_module(model: torch.nn.Module, module_list: list = [], name_list: list = []):
    """
    在模型中查找未量化的模块，并将其存储在列表中。

    参数：
    model: torch.nn.Module
        要查找未量化模块的模型。
    module_list: list, optional
        存储未量化模块的列表，默认为空列表。
    name_list: list, optional
        存储未量化模块名称的列表，默认为空列表。

    返回值：
    module_list: list
        存储未量化模块的列表，不包括输入模型本身。
    name_list: list
        存储未量化模块名称的列表，不包括输入模型本身。
    """
    global include
    for name, module in model.named_children():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            if not module.trained:
                include = True
                module.set_quant_state(False,False)
                name_list.append(name)
                module_list.append(module)
        elif include and type(module) in specials_unquantized:
            name_list.append(name)
            module_list.append(module)
        else:
            find_unquantized_module(module, module_list, name_list)
    return module_list[1:], name_list[1:]

def block_reconstruction(model: QuantModel, fp_model: QuantModel, block: BaseQuantBlock, fp_block: BaseQuantBlock,
                        cali_data: torch.Tensor, batch_size: int = 32, iters: int = 20000, weight: float = 0.01, 
                        opt_mode: str = 'mse', b_range: tuple = (20, 2),
                        warmup: float = 0.0, p: float = 2.0, lr: float = 4e-5,
                        input_prob: float = 1.0, keep_gpu: bool = True, 
                        lamb_r: float = 0.2, T: float = 7.0, bn_lr: float = 1e-3, lamb_c=0.02):
    """
    用于优化每个块的输出的重构。

    :param model: QuantModel
    :param block: 需要优化的BaseQuantBlock
    :param cali_data: 用于校准的数据，通常是1024个训练图像，如AdaRound中所述
    :param batch_size: 重构的小批量大小
    :param iters: 重构的优化迭代次数
    :param weight: 舍入正则化项的权重
    :param opt_mode: 优化模式
    :param asym: 在AdaRound中设计的非对称优化，使用量化输入来重构fp输出
    :param include_act_func: 优化激活函数后的输出
    :param b_range: 温度范围
    :param warmup: 温度调度中不进行调度的迭代比例
    :param lr: 学习率，用于学习act delta
    :param p: L_p范数最小化
    :param lamb_r: 正则化的超参数
    :param T: KL散度的温度系数
    :param bn_lr: DC的学习率
    :param lamb_c: DC的超参数
    """

    '''获取输入并设置比例'''
    cached_inps = get_init(model, block, cali_data, batch_size=batch_size, 
                                        input_prob=True, keep_gpu=keep_gpu)
    cached_outs, cached_output, cur_syms = get_dc_fp_init(fp_model, fp_block, cali_data, batch_size=batch_size, 
                                        input_prob=True, keep_gpu=keep_gpu, bn_lr=bn_lr, lamb=lamb_c)
    set_act_quantize_params(block, cali_data=cached_inps[:min(256, cached_inps.size(0))])

    '''set state'''
    cur_weight, cur_act = True, True
    
    global include
    module_list, name_list, include = [], [], False
    module_list, name_list = find_unquantized_module(model, module_list, name_list)
    block.set_quant_state(cur_weight, cur_act)
    for para in model.parameters():
        para.requires_grad = False

    '''set quantizer'''
    round_mode = 'learned_hard_sigmoid'
    # 将权重量化器替换为AdaRoundQuantizer
    w_para, a_para = [], []
    w_opt, a_opt = None, None
    scheduler, a_scheduler = None, None

    for module in block.modules():
        '''weight'''
        if isinstance(module, QuantModule):
            module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode=round_mode,
                                                        weight_tensor=module.org_weight.data)
            module.weight_quantizer.soft_targets = True
            w_para += [module.weight_quantizer.alpha]
        '''activation'''
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            if module.act_quantizer.delta is not None:
                module.act_quantizer.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer.delta))
                a_para += [module.act_quantizer.delta]
            '''set up drop'''
            module.act_quantizer.is_training = True

    if len(w_para) != 0:
        w_opt = torch.optim.Adam(w_para, lr=3e-3)#权重学习率
    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=lr)#激活函数学习率
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=iters, eta_min=0.)#余弦退火调度器

    loss_mode = 'relaxation'
    rec_loss = opt_mode
    loss_func = LossFunction(block, round_loss=loss_mode, weight=weight, max_count=iters, rec_loss=rec_loss,
                             b_range=b_range, decay_start=0, warmup=warmup, p=p, lam=lamb_r, T=T)
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
        
        out_all = block(cur_inp)

        '''forward for prediction difference'''
        out_drop = out_all[:batch_size]
        out_quant = out_all[batch_size:]
        output = out_quant
        for num, module in enumerate(module_list):
            # for ResNet and RegNet
            if name_list[num] == 'fc':
                output = torch.flatten(output, 1)
            # for MobileNet and MNasNet
            if isinstance(module, torch.nn.Dropout):
                output = output.mean([2, 3])
            output = module(output)
        err = loss_func(out_drop, cur_out, output, output_fp)

        err.backward(retain_graph=True)
        if w_opt:#更新权重
            w_opt.step()    
        if a_opt:#  更新激活函数
            a_opt.step()
        if scheduler:#更新学习率
            scheduler.step()
        if a_scheduler:#更新学习率
            a_scheduler.step()
    torch.cuda.empty_cache()

    for module in block.modules():
        if isinstance(module, QuantModule):
            '''weight '''
            module.weight_quantizer.soft_targets = False
        '''activation'''
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            module.act_quantizer.is_training = False
            module.trained = True
    for module in fp_block.modules():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            module.trained = True

class LossFunction:
    def __init__(self,
                 block: BaseQuantBlock,
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

        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.lam = lam
        self.T = T

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])#温度调度
        self.count = 0
        self.pd_loss = torch.nn.KLDivLoss(reduction='batchmean')#KL散度损失

    def __call__(self, pred, tgt, output, output_fp):
        """
        计算自适应舍入的总损失函数：
        rec_loss是输出重构损失，round_loss是优化舍入策略的正则化项，pd_loss是预测差异损失。

        :param pred: 量化模型的输出
        :param tgt: FP模型的输出
        :param output: 量化模型的预测
        :param output_fp: FP模型的预测
        :return: 总损失函数
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        else:
            raise ValueError('不支持的重构损失函数：{}'.format(self.rec_loss))

        pd_loss = self.pd_loss(F.log_softmax(output / self.T, dim=1), F.softmax(output_fp / self.T, dim=1)) / self.lam

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            for name, module in self.block.named_modules():
                if isinstance(module, QuantModule):
                    round_vals = module.weight_quantizer.get_soft_targets()
                    round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss + pd_loss
        if self.count % 500 == 0:
            print('总损失：\t{:.3f} (重构损失:{:.3f}, 预测差异损失:{:.3f}, 舍入损失:{:.3f})\tb={:.2f}\tcount={}'.format(
                float(total_loss), float(rec_loss), float(pd_loss), float(round_loss), b, self.count))
        return total_loss


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        用于温度b的余弦退火调度器。
        :param t: 当前时间步
        :return: 调度后的温度
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            # return self.end_b + 0.5 * (self.start_b - self.end_b) * (1 + np.cos(rel_t * np.pi))
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
