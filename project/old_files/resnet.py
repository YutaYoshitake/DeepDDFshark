import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, Union, Callable
import copy
# from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from torch import Tensor





class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            residual = self.downsample(x)

        out = out + residual      
        out = self.relu_inplace(out)

        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True, in_channel=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        # if output_stride == 16:
        #     strides = [1, 2, 2, 1]
        #     dilations = [1, 1, 1, 2]
        # elif output_stride == 8:
        #     strides = [1, 2, 1, 1]
        #     dilations = [1, 1, 2, 4]
        # else:
        #     raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)        
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dilation=1, BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilation=1, BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=1, BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=2, dilation=1, BatchNorm=BatchNorm)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # self.fc = nn.Linear(512*4, 512*4)

        self._init_weight()


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return self.avgpool(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



# def ResNet50(args, in_channel=3):
#     args.output_stride = 8
#     if args.gpu_num > 1:
#         BatchNorm = SynchronizedBatchNorm2d
#     else:
#         BatchNorm = nn.BatchNorm2d
#     model = ResNet(Bottleneck, [3, 4, 6, 3], args.output_stride, BatchNorm, in_channel=in_channel)
#     return model



class ResNet_wo_dilation(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True, in_channel=5, input_HW=256):
        self.inplanes = 64
        super(ResNet_wo_dilation, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 8:
            if input_HW == 128:
                strides = [1, 2, 1, 1]
            elif input_HW == 256:
                strides = [2, 2, 1, 1]
            else:
                raise NotImplementedError
            dilations = [1, 1, 1, 1]
            # dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)        
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        # #####
        # self.conv_1d = nn.Conv2d(2048, 512, 1)
        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # #####
        self._init_weight()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input): # input : torch.Size([10, 5, 256, 256])
        x = self.conv1(input) # torch.Size([15, 64, 64, 64]) / torch.Size([10, 64, 128, 128])
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # torch.Size([15, 64, 32, 32]) / torch.Size([10, 64, 64, 64])

        x = self.layer1(x) # torch.Size([15, 64, 16, 16]) / torch.Size([10, 256, 32, 32])
        x = self.layer2(x) # torch.Size([15, 128, 8, 8]) / torch.Size([10, 512, 16, 16])
        x = self.layer3(x) # torch.Size([15, 256, 8, 8]) / torch.Size([10, 1024, 16, 16])
        x = self.layer4(x) # torch.Size([15, 256, 8, 8]) / torch.Size([10, 2048, 16, 16])
        return x

        # x = self.layer1(x) # torch.Size([10, 256, 32, 32])
        # x = self.layer2(x) # torch.Size([10, 512, 16, 16])
        # x = self.layer3(x) # torch.Size([10, 1024, 16, 16])
        # x = self.layer4(x) # torch.Size([10, 2048, 16, 16])

        # x = self.conv_1d(x)
        # x = self.avgpool(x)
        # return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



def ResNet18_wo_dilation(in_channel=3, gpu_num=1):
    if gpu_num > 1:
        BatchNorm = SynchronizedBatchNorm2d
    else:
        BatchNorm = nn.BatchNorm2d
    model = ResNet_wo_dilation(BasicBlock, [2, 2, 2, 2], output_stride=8, BatchNorm=BatchNorm, in_channel=in_channel)
    return model



def ResNet50_wo_dilation(in_channel=3, gpu_num=1):
    if gpu_num > 1:
        BatchNorm = SynchronizedBatchNorm2d
    else:
        BatchNorm = nn.BatchNorm2d
    model = ResNet_wo_dilation(Bottleneck, [3, 4, 6, 3], output_stride=8, BatchNorm=BatchNorm, in_channel=in_channel)
    return model





# def _get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



# def _get_activation_fn(activation):
#     if activation == "relu":
#         return F.relu
#     elif activation == "gelu":
#         return F.gelu



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        # self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x



class IterationEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, 1, d_model) # max_len, frame, batch, dim
        pe[:, 0, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, length):
        if len(x.shape)==3:
            pe = [self.pe[itr_idx].expand(length_i, -1, -1) for itr_idx, length_i in enumerate(length)]
            pe = torch.cat(pe, dim=0)
            return x + pe
        elif len(x.shape)==4:
            pe = self.pe[torch.arange(x.shape[0])]
            return x + pe



# class TransformerEncoderLayer_woNorm(nn.Module):

#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
#         super(TransformerEncoderLayer_woNorm, self).__init__()
#         ##################################################
#         # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0)
#         # self.linear1 = nn.Linear(d_model, dim_feedforward)
#         # self.linear2 = nn.Linear(dim_feedforward, d_model)
#         ##################################################
#         dropout = 0.1
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         ##################################################
#         self.activation = _get_activation_fn(activation)

#     def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
#         src2, self.attn_output_weights = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask, attn_mask=src_mask)
#         ##################################################
#         # src = src + src2
#         # src2 = self.linear2(self.activation(self.linear1(src)))
#         # return src + src2
#         ##################################################
#         src = src + self.dropout1(src2)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         return src + self.dropout2(src2)



# class TransformerDecoderLayer_woNorm(nn.Module):

#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
#         super(TransformerDecoderLayer_woNorm, self).__init__()
        
#         dropout = 0.1
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)
#         self.activation = _get_activation_fn(activation)

#     def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
#                 tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
#         tgt2, self.self_attn_weights = self.self_attn(tgt, tgt, tgt)
#         tgt = tgt + self.dropout1(tgt2)
#         tgt2, self.mha_attn_weights = self.multihead_attn(tgt, memory, memory)
#         tgt = tgt + self.dropout2(tgt2)
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
#         return tgt + self.dropout3(tgt2)



# class CustomLayerNorm(nn.Module):

#     def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, variable_length='false'):
#         super(CustomLayerNorm, self).__init__()
        
#         self.eps = eps
#         self.normalized_shape = normalized_shape
#         self.elementwise_affine = elementwise_affine
#         if self.elementwise_affine:
#             self.weight = nn.Parameter(torch.empty(self.normalized_shape))
#             self.bias = nn.Parameter(torch.empty(self.normalized_shape))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)
        
#         self.variable_length = variable_length

#         self.reset_parameters()

#     def reset_parameters(self):
#         if self.elementwise_affine:
#             nn.init.ones_(self.weight)
#             nn.init.zeros_(self.bias)

#     def forward(self, inp, first_itr_or_optim_idx):
#         if self.variable_length=='false':
#             if first_itr_or_optim_idx:
#                 E_inp = inp.mean(-1, keepdim=True)
#                 Var_inp = (inp - E_inp).pow(2).mean(-1, keepdim=True)
#                 self.E_inp = E_inp.clone().detach()
#                 self.Var_inp = Var_inp.clone().detach()
#             else:
#                 E_inp = self.E_inp.detach()
#                 Var_inp = self.Var_inp.detach()
#             x = (inp - E_inp) / torch.sqrt(Var_inp + self.eps)
#             x = self.weight * x + self.bias
#             return x

#         if self.variable_length=='false_dec':
#             seq_len, batch, dim = inp.shape
#             if first_itr_or_optim_idx == 0:
#                 E_inp = inp.mean(-1, keepdim=True)
#                 Var_inp = (inp - E_inp).pow(2).mean(-1, keepdim=True)
#                 self.E_inp = E_inp.clone().detach()
#                 self.Var_inp = Var_inp.clone().detach()
#             elif first_itr_or_optim_idx > 0:
#                 E_inp = self.E_inp.tile(first_itr_or_optim_idx+1, 1, 1).detach()
#                 Var_inp = self.Var_inp.tile(first_itr_or_optim_idx+1, 1, 1).detach()
#             x = (inp - E_inp) / torch.sqrt(Var_inp + self.eps)
#             x = self.weight * x + self.bias
#             return x



# class TransformerEncoderLayer_CustomNorm(nn.Module):

#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", use_norm=False):
#         super(TransformerEncoderLayer_CustomNorm, self).__init__()

#         dropout = 0.1
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.d_model = d_model

#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.use_norm = use_norm
#         if self.use_norm:
#             self.norm1 = CustomLayerNorm(d_model, variable_length='false')
#             self.norm2 = CustomLayerNorm(d_model, variable_length='false')
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.activation = _get_activation_fn(activation)

#     def forward(self, src: Tensor, first_itr) -> Tensor:
#         src2, self.attn_output_weights = self.self_attn(src, src, src)
#         src = src + self.dropout1(src2)
#         if self.use_norm:
#             src = self.norm1(src, first_itr)
        
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         if self.use_norm:
#             src = self.norm2(src, first_itr)
#         return src



# class CustomTransformerEncoder(nn.Module):
#     __constants__ = ['norm']

#     def __init__(self, encoder_layer, num_layers):
#         super(CustomTransformerEncoder, self).__init__()

#         self.layers = _get_clones(encoder_layer, num_layers)
#         self.num_layers = num_layers

#     def forward(self, src: Tensor, first_itr, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
#         output = src
#         for mod in self.layers:
#             output = mod(output, first_itr)
#         return output



# class TransformerDecoderLayer_CustomNorm(nn.Module):

#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", use_norm=False):
#         super(TransformerDecoderLayer_CustomNorm, self).__init__()
#         dropout = 0.1

#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.d_model = d_model
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.use_norm = use_norm
#         if self.use_norm:
#             self.norm1 = CustomLayerNorm(d_model, variable_length='false_dec')
#             self.norm2 = CustomLayerNorm(d_model, variable_length='false_dec')
#             self.norm3 = CustomLayerNorm(d_model, variable_length='false_dec')
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)

#         self.activation = _get_activation_fn(activation)

#     def forward(self, tgt: Tensor, memory: Tensor, optim_idx) -> Tensor:
#         tgt2, self.self_attn_weights = self.self_attn(tgt, tgt, tgt)
#         tgt = tgt + self.dropout1(tgt2)
#         if self.use_norm:
#             tgt = self.norm1(tgt, optim_idx)
        
#         tgt2, self.mha_attn_weights = self.multihead_attn(tgt, memory, memory)
#         tgt = tgt + self.dropout2(tgt2)
#         if self.use_norm:
#             tgt = self.norm2(tgt, optim_idx)

#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
#         tgt = tgt + self.dropout3(tgt2)
#         if self.use_norm:
#             tgt = self.norm3(tgt, optim_idx)
#         return tgt



# class CustomTransformerDecoder(nn.Module):

#     def __init__(self, decoder_layer, num_layers):
#         super(CustomTransformerDecoder, self).__init__()
#         self.layers = _get_clones(decoder_layer, num_layers)
#         self.num_layers = num_layers

#     def forward(self, tgt: Tensor, memory: Tensor, optim_idx) -> Tensor:
#         output = tgt
#         for mod in self.layers:
#             output = mod(output, memory, optim_idx)
#         return output


