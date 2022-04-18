import math
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair,_triple
import pickle
import numpy as np
np.random.seed(1)

OUT_CHAN=2048
IN_CHAN=64
IFM_DIM=64
KERN=3
STD = 4.0

class DeformConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, offset, weight, bias=None, stride=1, padding=0, dilation=1,
                groups=1, deformable_groups=1 , KG=1, in_step=64):

        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.KG = KG
        ctx.in_step = in_step
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(0)  # fake tensor
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or offset.requires_grad or input.requires_grad:
            ctx.save_for_backward(input, offset, weight, bias)
        output = input.new_empty(DeformConv2dFunction._infer_shape(ctx, input, weight))
        '''
        MDCONV_CUDA.deform_conv2d_forward_cuda(
            input, weight, bias, offset, output,
            weight.shape[2],weight.shape[3],
            ctx.stride[0], ctx.stride[1],
            ctx.padding[0], ctx.padding[1],
            ctx.dilation[0],ctx.dilation[1],
            ctx.groups, ctx.deformable_groups, ctx.KG, ctx.in_step, ctx.with_bias)
        '''
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_output=grad_output.contiguous()
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        '''
        MDCONV_CUDA.deform_conv2d_backward_cuda(
            input, weight, bias, offset,
            grad_input, grad_weight,grad_bias,grad_offset, grad_output,
            weight.shape[2], weight.shape[3],
            ctx.stride[0], ctx.stride[1],
            ctx.padding[0], ctx.padding[1],
            ctx.dilation[0], ctx.dilation[1],
            ctx.groups, ctx.deformable_groups, ctx.KG, ctx.in_step,ctx.with_bias)
        '''
        if not ctx.with_bias:
            grad_bias = None

        return grad_input, grad_offset, grad_weight, grad_bias, None, None, None, None,None,None,None

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding[0] - (ctx.dilation[0] *(kernel_h - 1) + 1)) // ctx.stride[0] + 1
        width_out = (width + 2 * ctx.padding[1] - (ctx.dilation[1] *(kernel_w - 1) + 1)) // ctx.stride[1] + 1
        return n, channels_out, height_out, width_out

class FusedDeformConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, offset_weight=None, stride=1, padding=0, dilation=1,
                groups=1, deformable_groups=1, KG=1, in_step=64):

        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.KG = KG
        ctx.in_step = in_step
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(0)  # fake tensor                                                                                                                                          
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or offset_weight.requires_grad or input.requires_grad:
            ctx.save_for_backward(input, offset_weight, weight, bias)
        offset = input.new_empty(DeformConv2dFunction._infer_shape(ctx, input, offset_weight))
        output = input.new_empty(DeformConv2dFunction._infer_shape(ctx, input, weight))
        '''
        MDCONV_CUDA.fused_deform_conv2d_forward_cuda(
            input, weight, bias, offset_weight, offset, output,
            weight.shape[2],weight.shape[3],
            ctx.stride[0], ctx.stride[1],
            ctx.padding[0], ctx.padding[1],
            ctx.dilation[0],ctx.dilation[1],
            ctx.groups, ctx.deformable_groups, ctx.KG, ctx.in_step, ctx.with_bias)
        '''
        return output
        
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_offset):
        grad_output=grad_output.contiguous()
        grad_offset=grad_offset.contiguous()
        if not grad_output.is_cuda or grad_offset.is_cuda:
            raise NotImplementedError
        input, offset_weight, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset_weight = torch.zeros_like(offset_weight)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        '''
        MDCONV_CUDA.fused_deform_conv2d_backward_cuda(
            input, weight, bias, offset_weight, offset,
            grad_input, grad_weight,grad_bias,grad_offset_weight, grad_output, grad_offset,
            weight.shape[2], weight.shape[3],
            ctx.stride[0], ctx.stride[1],
            ctx.padding[0], ctx.padding[1],
            ctx.dilation[0], ctx.dilation[1],
            ctx.groups, ctx.deformable_groups,ctx.in_step,ctx.with_bias)
        '''
        if not ctx.with_bias:
            grad_bias = None

        return grad_input, grad_offset_weight, grad_weight, grad_bias, None, None, None, None,None,None

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding[0] - (ctx.dilation[0] *(kernel_h - 1) + 1)) // ctx.stride[0] + 1
        width_out = (width + 2 * ctx.padding[1] - (ctx.dilation[1] *(kernel_w - 1) + 1)) // ctx.stride[1] + 1
        return n, channels_out, height_out, width_out
    
deform_conv2d = DeformConv2dFunction.apply
fused_deform_conv2d = FusedDeformConv2dFunction.apply

class deform_conv2d_wrapper(nn.Module):
    def __init__(self, weight, bias, in_channels, out_channels, stride, padding, dilation,
                 groups, deformable_groups, KG, in_step=64):
        super(deform_conv2d_wrapper, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.KG = KG
        self.in_step = in_step
        
    def forward(self, x, offset):
        return deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation,
                             self.groups, self.deformable_groups, self.KG, self.in_step)

class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, KG=1, bias=False,in_step=64):
        super(DeformConv2d, self).__init__()
        assert in_channels % groups == 0, \
            'in_channels {} cannot be divisible by groups {}'.format(
                in_channels, groups)
        assert out_channels % groups == 0, \
            'out_channels {} cannot be divisible by groups {}'.format(
                out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.KG = KG
        self.in_step=in_step

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        self.with_bias=bias
        if self.with_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias=None

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.with_bias:
            self.bias.data.fill_(0)

    def forward(self, x, offset):
        return deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation,
                             self.groups, self.deformable_groups, self.KG, self.in_step)

    # def forward(ctx, input, offset, weight, bias=None, stride=1, padding=0, dilation=1,
    #             groups=1, deformable_groups=1 , in_step=64):

class FusedDeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, KG=1, bias=False,in_step=64):
        super(FusedDeformConv2d, self).__init__()
        assert in_channels % groups == 0, \
            'in_channels {} cannot be divisible by groups {}'.format(
                in_channels, groups)
        assert out_channels % groups == 0, \
            'out_channels {} cannot be divisible by groups {}'.format(
                out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.KG = KG
        self.in_step=in_step

        self.offset_conv_weight = nn.Parameter(
            torch.Tensor(self.deformable_groups*self.kernel_size[0]*self.kernel_size[1]*2, in_channels, *self.kernel_size))
        #self.offset_conv_weight = nn.Parameter(
        #    torch.Tensor(self.kernel_size[0]*self.kernel_size[1]*2, in_channels // self.deformable_groups, *self.kernel_size))
        
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        self.with_bias=bias
        if self.with_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias=None

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.offset_conv_weight.data.uniform_(-stdv, stdv)
        if self.with_bias:
            self.bias.data.fill_(0)

    def forward(self, x):

        return fused_deform_conv2d(x, self.weight, self.bias, self.offset_conv_weight, self.stride, self.padding, self.dilation,
                                   self.groups, self.deformable_groups,self.KG,self.in_step)
    
class DeformConv2dPack(DeformConv2d):
    def __init__(self, *args, **kwargs):
        super(DeformConv2dPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding),
            bias=True)
        self.init_offset()

        self.wrapper = deform_conv2d_wrapper(self.weight, self.bias, self.in_channels, self.out_channels, self.stride, self.padding, self.dilation, \
                                             self.groups, self.deformable_groups, self.in_step)

    def init_offset(self):

        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.conv_offset.weight.data.uniform_(-stdv, stdv)
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        return self.wrapper(x, offset)


def export(pathName, modelName, out_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # --- use pathName and modelName to load REAL model here ---
    model = DeformConv2dPack(in_channels=IN_CHAN, out_channels=OUT_CHAN, kernel_size=KERN, stride=1, padding=1, dilation=1, in_step=64)
    model = model.to(device)
    model.eval()
    models = []

    def extract(module, input):#, offset=None):
        if len(input[0].shape) < 4:
            try:
                a = input[0].detach().cpu().reshape(1, module.in_features, 1, 1)
            except:
                a = input[0].detach().cpu().reshape(-1, 1, 1)
                a = a[:module.in_features]
                a = a.reshape(1, module.in_features, 1, 1)
        else:
            a = input[0].detach().cpu()

        os = None # local offset
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, deform_conv2d_wrapper): #isinstance(module, DeformConv2dPack):
            layer = module.weight.view((module.out_channels, -1)).detach().cpu().numpy()
            weight = module.weight.detach().cpu().numpy()
            if isinstance(module, torch.nn.Conv2d):
                tp = "conv"
            elif isinstance(module, deform_conv2d_wrapper):
                tp = "deformconv"
                #os = input[1].detach().cpu().numpy()
                os = np.random.normal(0,STD,input[1].shape)
            stride = str(max(module.stride[0], module.stride[1]))
            padding = str(max(module.padding[0], module.padding[1]))
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = (weight.shape[2], weight.shape[3])
            padding = module.padding
            stride = module.stride
        elif isinstance(module, DeformConv2dPack):
            return
        elif isinstance(module, torch.nn.Linear) and (not skipLinearExport):
            layer = module.weight.view((module.out_features, -1)).detach().cpu().numpy()
            weight = module.weight.detach().cpu().reshape(module.weight.shape[0], module.weight.shape[1], 1, 1).numpy()
            tp = "fc"
            stride = str(1)
            padding = str(0)
            in_channels = module.in_features
            out_channels = module.out_features
            kernel_size = (1,1)
            padding = (0,0)
            stride = (1,1)
        else:
            print("{} does not exist".format(module))
            exit(1)
        name = '0' ## REPLACE WHEN USING REAL MODEL
        models.append({'in_channels': in_channels,
                       'out_channels': out_channels,
                       'kernel': kernel_size,
                       'name': tp+name,
                       'padding': padding,
                       'weights': weight,
                       'IFM': a.cpu().numpy(),
                       'offset': os,
                       'stride': stride
        })
    for n, m in model.named_modules():
        print(m)
        m.register_forward_pre_hook(extract)

    IFM = torch.rand(1, IN_CHAN, IFM_DIM, IFM_DIM).cuda()
    model(IFM)

    with open(out_path+modelName+"_std"+str(STD)+"_"+str(OUT_CHAN)+"_"+str(IN_CHAN)+"_"+str(IFM_DIM)+"_"+str(KERN)+".h5", "wb") as f:
        pickle.dump(models, f)

    print(models)
            
if __name__ == '__main__':
    export(pathName=None, modelName='DeformConv2dPack', out_path='/root/hostCurUser/reproduce/SparTen/data/')
    
    #self.conv_offset.weight
    #self.weight
    
