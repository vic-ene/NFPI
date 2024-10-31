import scipy
from scipy.linalg import logm
from scipy.stats import ortho_group, special_ortho_group

import torch
import torch.nn as nn
import torch.nn.functional as F


from models.utils import expm, series
import sys
eps = 1e-8

from models.torchcfm_attention.torchcfm_attention import *


class ActNorm(nn.Module):
    """
    Actnorm layers: y = softplus(scale) * x + shift which are data dependent, initialized
    such that the distribution of activations has zero mean and unit variance
    given an initial mini-batch of data.
    Normalize all activations independently instead of normalizing per channel.
    """

    def __init__(self, in_channels, h, w):
        super(ActNorm, self).__init__()
        self.in_channels = in_channels
        self.h = h
        self.w = w
        self.scale = nn.Parameter(torch.ones(in_channels, h, w))
        self.shift = nn.Parameter(torch.zeros(in_channels, h, w))

    
    def forward(self, x, y, reverse=False, init=False, init_scale=1.0, label_encoder=None):
        if init:
            mean = x.mean(dim=0)
            std = x.std(dim=0)
            inv_std = init_scale / (std + eps)
            self.scale.data.copy_(torch.log(-1 + torch.exp(inv_std)))
            self.shift.data.copy_(-mean * inv_std)

        if not reverse:
            scale = F.softplus(self.scale)
            x = scale * x + self.shift
            log_det = torch.log(scale).sum()
        else:
            scale = F.softplus(self.scale)
            x = (x - self.shift) / scale
            log_det = torch.log(scale).sum().mul(-1)
        return x, log_det

    def extra_repr(self):
        return 'in_channels={}, h={}, w={}'.format(self.in_channels, self.h, self.w)


class Norm(nn.Module):
    """
    Norm layers: y = scale * x +shift which are data dependent, initialized
    such that the distribution of activations per channel has zero mean and unit variance
    given an initial mini-batch of data.
    """

    def __init__(self, in_channels):
        super(Norm, self).__init__()
        self.in_channels = in_channels
        self.scale = nn.Parameter(torch.ones(in_channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(in_channels, 1, 1))

    def forward(self, x, init=False, init_scale=1.0):
        if init:
            out = x.transpose(0, 1).contiguous().view(self.in_channels, -1)
            mean = out.mean(dim=1).view(self.in_channels, 1, 1)
            std = out.std(dim=1).view(self.in_channels, 1, 1)
            inv_std = init_scale / (std + eps)
            self.scale.data.copy_(inv_std)
            self.shift.data.copy_(-mean * inv_std)

        x = self.scale * x + self.shift
        return x

    def extra_repr(self):
        return 'in_channels={}'.format(self.in_channels)


class Conv1x1(nn.Module):
    """
    1x1 convolutions have three types.
    Standard convolutions:  y = Wx
    PLU decomposition convolutions: y = PLUx
    Matrix exp convolutions: y = e^{W}x
    """

    def __init__(self, in_channels, conv_type):
        super(Conv1x1, self).__init__()
        self.in_channels = in_channels
        self.conv_type = conv_type
        if not conv_type == 'decomposition':
            self.weight = nn.Parameter(torch.rand(in_channels, in_channels))
        else:
            self.l = nn.Parameter(torch.rand(in_channels, in_channels))
            self.u = nn.Parameter(torch.rand(in_channels, in_channels))
            p = torch.rand(in_channels, in_channels)
            l_mask = torch.tril(torch.ones(in_channels, in_channels), diagonal=-1)
            identity = torch.eye(in_channels)
            u_mask = torch.tril(torch.ones(in_channels, in_channels), diagonal=0).t()
            self.register_buffer('p', p)
            self.register_buffer('l_mask', l_mask)
            self.register_buffer('identity', identity)
            self.register_buffer('u_mask', u_mask)

    def forward(self, x, y, reverse=False, init=False, label_encoder=None):
        if init:
            if self.conv_type == 'matrixexp':
                rand = special_ortho_group.rvs(self.in_channels)
                rand = logm(rand)
                rand = torch.from_numpy(rand.real)
                self.weight.data.copy_(rand)
            elif self.conv_type == 'standard':
                nn.init.orthogonal_(self.weight)
            elif self.conv_type == 'decomposition':
                w = ortho_group.rvs(self.in_channels)
                p, l, u = scipy.linalg.lu(w)
                self.p.copy_(torch.from_numpy(p))
                self.l.data.copy_(torch.from_numpy(l))
                self.u.data.copy_(torch.from_numpy(u))
            else:
                raise ValueError('wrong 1x1 conlution type')

        if not reverse:
            if self.conv_type == 'matrixexp':
                weight = expm(self.weight)
                x = F.conv2d(x, weight.view(self.in_channels, self.in_channels, 1, 1))
                log_det = torch.diagonal(self.weight).sum().mul(x.size(2) * x.size(3))
            elif self.conv_type == 'standard':
                x = F.conv2d(x, self.weight.view(self.in_channels, self.in_channels, 1, 1))
                _, log_det = torch.slogdet(self.weight)
                log_det = log_det.mul(x.size(2) * x.size(3))
            elif self.conv_type == 'decomposition':
                l = self.l * self.l_mask + self.identity
                u = self.u * self.u_mask
                weight = torch.matmul(self.p, torch.matmul(l, u))
                x = F.conv2d(x, weight.view(self.in_channels, self.in_channels, 1, 1))
                log_det = torch.diagonal(self.u).abs().log().sum().mul(x.size(2) * x.size(3))
            else:
                raise ValueError('wrong 1x1 conlution type')
        else:
            if self.conv_type == 'matrixexp':
                weight = expm(-self.weight)
                x = F.conv2d(x, weight.view(self.in_channels, self.in_channels, 1, 1))
                log_det = torch.diagonal(self.weight).sum().mul(x.size(2) * x.size(3)).mul(-1)
            elif self.conv_type == 'standard':
                x = F.conv2d(x, torch.inverse(self.weight).view(self.in_channels, self.in_channels, 1, 1))
                _, log_det = torch.slogdet(self.weight)
                log_det = log_det.mul(x.size(2) * x.size(3)).mul(-1)
            elif self.conv_type == 'decomposition':
                l = self.l * self.l_mask + self.identity
                u = self.u * self.u_mask
                weight = torch.matmul(self.p, torch.matmul(l, u))
                x = F.conv2d(x, torch.inverse(weight).view(self.in_channels, self.in_channels, 1, 1))
                log_det = torch.diagonal(self.u).sum().mul(x.size(2) * x.size(3)).mul(-1)
            else:
                raise ValueError('wrong 1x1 conlution type')
        return x, log_det

    def extra_repr(self):
        return 'in_channels={}, conv_type={}'.format(self.in_channels, self.conv_type)



class CouplingLayer(nn.Module):
    """
    Coupling layers have three types.
    Additive coupling layers: y2 = x2 + b(x1)
    Affine coupling layers: y2 = s(x1) * x2 + b(x1)
    Matrix exp coupling layers: y2 = e^{s(x1)}x2 + b(x1)
    """

    def __init__(self, flow_type, num_blocks, in_channels, hidden_channels, label_encoder, cfg=None):
        super(CouplingLayer, self).__init__()
        self.flow_type = flow_type
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.x2_channels = in_channels // 2
        self.x1_channels = in_channels - self.x2_channels
        if flow_type == 'additive':
            self.num_out = 1
        elif flow_type == 'affine':
            self.scale = nn.Parameter(torch.ones(1) / 8)
            self.shift = nn.Parameter(torch.zeros(1))
            self.rescale = nn.Parameter(torch.ones(1))
            self.reshift = nn.Parameter(torch.zeros(1))
            self.num_out = 2
        elif flow_type == 'matrixexp':
            self.scale = nn.Parameter(torch.ones(1) / 8)
            self.shift = nn.Parameter(torch.zeros(1))
            self.rescale = nn.Parameter(torch.ones(1) / self.x2_channels)
            self.reshift = nn.Parameter(torch.zeros(1))
            self.max_out = 24
            if self.x2_channels <= self.max_out:
                self.num_out = (self.x2_channels + 1)
            else:
                self.k = 3
                self.num_out = 2 * self.k + 1
        else:
            raise ValueError('wrong flow type')
                
        self.label_encoder = label_encoder

        in_channels_conv = self.x1_channels
        out_channels_conv = self.x2_channels * self.num_out
        # If we use a label encoder, we increase the number of channels if necessary
        if self.label_encoder != None:
            blank_h = 32; blank_w = 32
            in_channels_conv, blank_h, blank_w = self.label_encoder.get_ch_h_w(in_channels_conv, blank_h, blank_w)


        use_attention_after = False
        if "use_attention_after" in cfg:
            use_attention_after = cfg["use_attention_after"]

        if "coupling_block" in cfg:
            if cfg["coupling_block"] == "":
                self.net = ConvBlock(num_blocks, in_channels_conv, self.hidden_channels, out_channels_conv, use_attention_after)
            else:
                sys.exit("This coupling block is not implemented")
        else:
            self.net = ConvBlock(num_blocks, in_channels_conv, self.hidden_channels, out_channels_conv, use_attention_after)

    def forward(self, x, y, reverse=False, init=False, label_encoder=None):
        x1 = x[:, :self.x1_channels]
        x2 = x[:, self.x1_channels:]

        def append_encoding(x_, y_, label_encoder):
            if label_encoder == None:
                return x_
            else:
                return label_encoder.encode_batch_for_coupling_layer(x_, y_)
    
        if self.flow_type == 'additive':
            if not reverse:
                x2 = x2 + self.net(append_encoding(x1, y, label_encoder), init=init)
                out = torch.cat([x1, x2], dim=1)
                log_det = x.new_zeros(x.size(0))
            else:
                x2 = x2 - self.net(append_encoding(x1, y, label_encoder))
                out = torch.cat([x1, x2], dim=1)
                log_det = x.new_zeros(x.size(0))
        elif self.flow_type == 'affine':
            if not reverse:
                out = self.net(append_encoding(x1, y, label_encoder), init=init)
                outs = out.chunk(2, dim=1)
                shift = outs[0]
                log_scale = self.rescale * torch.tanh(self.scale * outs[1] + self.shift) + self.reshift
                x2 = torch.exp(log_scale) * x2 + shift
                out = torch.cat([x1, x2], dim=1)
                log_det = log_scale.sum([1, 2, 3])
            else:
                out = self.net(append_encoding(x1, y, label_encoder))
                outs = out.chunk(2, dim=1)
                shift = outs[0]
                log_scale = self.rescale * torch.tanh(self.scale * outs[1] + self.shift) + self.reshift
                x2 = torch.exp(-log_scale) * (x2 - shift)
                out = torch.cat([x1, x2], dim=1)
                log_det = log_scale.sum([1, 2, 3]).mul(-1)
        elif self.flow_type == 'matrixexp':
            if not reverse:
                if self.x2_channels <= self.max_out:
                    out = self.net(append_encoding(x1, y, label_encoder), init=init).unsqueeze(2)
                    outs = out.chunk(self.num_out, dim=1)
                    shift = outs[0].squeeze(2)
                    weight = torch.cat(outs[1:], dim=2).permute(0, 3, 4, 1, 2)
                    weight = self.rescale * torch.tanh(self.scale * weight + self.shift) + self.reshift
                    x2 = x2.unsqueeze(2).permute(0, 3, 4, 1, 2)
                    x2 = torch.matmul(expm(weight), x2).permute(0, 3, 4, 1, 2).squeeze(2) + shift
                    out = torch.cat([x1, x2], dim=1)
                    log_det = torch.diagonal(weight, dim1=-2, dim2=-1).sum([1, 2, 3])
                else:
                    out = self.net(append_encoding(x1, y, label_encoder), init=init).unsqueeze(2)
                    outs = out.chunk(self.num_out, dim=1)
                    shift = outs[0].squeeze(2)
                    weight1 = torch.cat(outs[1:self.k + 1], dim=2).permute(0, 3, 4, 2, 1)
                    weight2 = torch.cat(outs[self.k + 1:2 * self.k + 1], dim=2).permute(0, 3, 4, 1, 2)
                    weight1 = self.rescale * torch.tanh(self.scale * weight1 + self.shift) + self.reshift + eps
                    weight2 = self.rescale * torch.tanh(self.scale * weight2 + self.shift) + self.reshift + eps
                    weight3 = torch.matmul(weight1, weight2)
                    weight = torch.eye(self.x2_channels, device=x.device) + torch.matmul(
                        torch.matmul(weight2, series(weight3)), weight1)
                    x2 = x2.unsqueeze(2).permute(0, 3, 4, 1, 2)
                    x2 = torch.matmul(weight, x2).permute(0, 3, 4, 1, 2).squeeze(2) + shift
                    out = torch.cat([x1, x2], dim=1)
                    log_det = torch.diagonal(weight3, dim1=-2, dim2=-1).sum([1, 2, 3])
            else:
                if self.x2_channels <= self.max_out:
                    out = self.net(append_encoding(x1, y, label_encoder)).unsqueeze(2)
                    outs = out.chunk(self.num_out, dim=1)
                    shift = outs[0].squeeze(2)
                    weight = torch.cat(outs[1:], dim=2).permute(0, 3, 4, 1, 2)
                    weight = self.rescale * torch.tanh(self.scale * weight + self.shift) + self.reshift
                    x2 = (x2 - shift).unsqueeze(2).permute(0, 3, 4, 1, 2)
                    x2 = torch.matmul(expm(-weight), x2).permute(0, 3, 4, 1, 2).squeeze(2)
                    out = torch.cat([x1, x2], dim=1)
                    log_det = torch.diagonal(weight, dim1=-2, dim2=-1).sum([1, 2, 3]).mul(-1)
                else:
                    out = self.net(append_encoding(x1, y, label_encoder)).unsqueeze(2)
                    outs = out.chunk(self.num_out, dim=1)
                    shift = outs[0].squeeze(2)
                    weight1 = torch.cat(outs[1:self.k + 1], dim=2).permute(0, 3, 4, 2, 1)
                    weight2 = torch.cat(outs[self.k + 1:2 * self.k + 1], dim=2).permute(0, 3, 4, 1, 2)
                    weight1 = self.rescale * torch.tanh(self.scale * weight1 + self.shift) + self.reshift + eps
                    weight2 = self.rescale * torch.tanh(self.scale * weight2 + self.shift) + self.reshift + eps
                    weight3 = torch.matmul(weight1, weight2)
                    weight = torch.eye(self.x2_channels, device=x.device) - torch.matmul(
                        torch.matmul(weight2, series(-weight3)), weight1)
                    x2 = (x2 - shift).unsqueeze(2).permute(0, 3, 4, 1, 2)
                    x2 = torch.matmul(weight, x2).permute(0, 3, 4, 1, 2).squeeze(2)
                    out = torch.cat([x1, x2], dim=1)
                    log_det = torch.diagonal(weight3, dim1=-2, dim2=-1).sum([1, 2, 3]).mul(-1)
        else:
            raise ValueError('wrong flow type')

        return out, log_det

    def extra_repr(self):
        return 'in_channels={}, hidden_channels={}, out_channels={},flow_type={}'.format(self.in_channels,
                                                                                         self.hidden_channels,
                                                                                         self.in_channels,
                                                                                         self.flow_type)


class ConvBlock(nn.Module):
    def __init__(self, num_blocks, in_channels, hidden_channels, out_channels, use_attention_after):
        super(ConvBlock, self).__init__()
        layers = list()
        layers.append(Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, activ=True))
        for _ in range(num_blocks):
            layers.append(Block(hidden_channels))

        self.use_attention_after = use_attention_after
        if self.use_attention_after:
            num_heads = 6
            print("num heads", num_heads)
            for _ in range(2): print("using attention after")
            self.attention_block_after = AttentionBlock(channels=out_channels, num_heads=num_heads, num_head_channels=-1, use_checkpoint=False, use_new_attention_order=False)



        self.layers = nn.ModuleList(layers)
        self.out_layer = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        nn.init.constant_(self.out_layer.weight, 0.0)
        nn.init.constant_(self.out_layer.bias, 0.0)

    def forward(self, x, init=False):
        for layer in self.layers:
            x = layer(x, init=init)
        x = self.out_layer(x)

        if self.use_attention_after:
            x = self.attention_block_after(x)
        return x
    

    
class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out
    

    
class Block(nn.Module):
    def __init__(self, channels):
        super(Block, self).__init__()
        self.conv1 = Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv3 = Conv2d(channels, channels, kernel_size=3, padding=1)
        self.elu = nn.ELU()

    def forward(self, x, init=False):
        identity = x
        out = self.elu(self.conv1(x, init=init))
        out = self.elu(self.conv2(out, init=init))
        out = self.conv3(out, init=init, init_scale=0.0)
        out += identity
        out = self.elu(out)
        return out


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activ=False):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activ = activ
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.norm = Norm(out_channels)
        nn.init.kaiming_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)
        if activ:
            self.elu = nn.ELU(inplace=True)

    def forward(self, x, init=False, init_scale=1.0):
        if init:
            out = self.forward(x)
            n_channels = out.size(1)
            out = out.transpose(0, 1).contiguous().view(n_channels, -1)
            mean = out.mean(dim=1)
            std = out.std(dim=1)
            inv_std = 1.0 / (std + eps)
            self.conv.weight.data.mul_(inv_std.view(n_channels, 1, 1, 1))
            if self.conv.bias is not None:
                self.conv.bias.data.add_(-mean).mul_(inv_std)

        x = self.conv(x)
        x = self.norm(x, init=init, init_scale=init_scale)
        if self.activ:
            x = self.elu(x)
        return x

    def extra_repr(self):
        return 'in_channels={}, out_channels={}'.format(self.in_channels, self.out_channels)
