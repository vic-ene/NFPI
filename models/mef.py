import torch.nn as nn

from models.utils import squeeze2d, unsqueeze2d, split2d, unsplit2d
from models.flows import ActNorm, Conv1x1, CouplingLayer

from torch.profiler import profile, record_function, ProfilerActivity

from my_util.label_encoder import *

from models.utils import *
from models.gaussians_util import *
from my_util.reparam_classes import *


from models.flows import *

from math import log, pi
import sys



def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)

def gaussian_sample(eps, mean, log_sd, temperature):
    return mean + torch.exp(log_sd) * eps * temperature

class MEF(nn.Module):
    def __init__(self, cfg, num_levels, num_flows, conv_type, flow_type, num_blocks, hidden_channels, h, w, in_channels=3):
        super(MEF, self).__init__()

        self.cfg = cfg

        # Initialize encoder in the coupling layers if required        
        dataset_name = cfg["dataset_name"]
        self.num_classes = cfg["dataset_info"][dataset_name]["num_classes"]
      
        # Initizalize the gaussians, register their parameters, and create the Adam Optimizers
        cfg_gmm = self.cfg["val"]["gmm"]
        reparam_name = cfg_gmm["reparam_name"]
        reparam_cfg = cfg_gmm["reparam_method"][reparam_name]
        reparam_dict = {
            "no_reparam": NoReparam,
            "sigmoid": SigmoidReparam,
        }

        self.reparam = reparam_dict[reparam_name](**reparam_cfg)
        self.mu_arr, self.cov_lower_arr, self.scale_tril_arr = init_gaussians(
            cfg,
            method_name=cfg_gmm["gaussian_method_name"],
            reparam=self.reparam,
            num_classes=(self.num_classes),
            dtype=self.get_dtype()
        )
        self.mu_arr = nn.ParameterList(self.mu_arr)
        self.cov_lower_arr = nn.ParameterList(self.cov_lower_arr)
        self.scale_tril_arr = nn.ParameterList(self.scale_tril_arr)

        print("here", len(self.mu_arr), len(self.cov_lower_arr), len(self.scale_tril_arr))
        print("here", len(self.mu_arr), len(self.cov_lower_arr), len(self.scale_tril_arr))
        
        # Initialize encoder in the coupling layers if required        
        dataset_name = cfg["dataset_name"]
        num_classes = cfg["dataset_info"][dataset_name]["num_classes"]
        self.label_encoder = None
        self.cfg = cfg

        self.num_levels = num_levels
        level_idx = self.cfg["level_method"]["level_idx"]
        spcb = cfg["level_method"]["spcb"]

        blocks = []
        split_coupling_priors = list()
        for i in range(num_levels):
            in_channels = in_channels * 4
            h = h // 2
            w = w // 2

            c_num_flows = num_flows[i]
            flows = [
                Flow(conv_type, flow_type, num_blocks[i], in_channels, hidden_channels[i], h, w, cfg, flow_idx==c_num_flows) 
                for flow_idx in range(c_num_flows)
            ]
            blocks.append(Sequential(*flows))

            if i < num_levels - 1:
                in_channels = in_channels // 2

            if cfg["level_method"]["choice"] in [SPCB]:
                if i < num_levels - 1:
                    if spcb == "ZeroConv2d":
                        split_coupling_priors.append(ZeroConv2d(in_channels, in_channels*2))
                    else:
                        sys.exit("This split coupling prior block is not implemented yet")
                self.split_coupling_priors = nn.ModuleList(split_coupling_priors)

            elif cfg["level_method"]["choice"] in ["all_levels_idx"]:
                if i < level_idx - 1:
                    if spcb == "ZeroConv2d":
                        split_coupling_priors.append(ZeroConv2d(in_channels, in_channels*2))
                    else:
                        sys.exit("This split coupling prior block is not implemented yet")
                    self.split_coupling_priors = nn.ModuleList(split_coupling_priors)

        self.blocks = nn.ModuleList(blocks)

        # Will register the shape output from each level
        self.register_buffer(
            "level_out_shape_arr", 
            torch.tensor([(0, 0, 0) for _ in range(num_levels)])
        )


        self.recorded_outputs = []

        # Load noise for dequantization (calculated on the data)

        if "noise_dequant_path" in cfg["vae"]:
            noise_dequant_path = cfg["vae"]["noise_dequant_path"]
            if (noise_dequant_path != None) and (noise_dequant_path != ""): 
                noise_dequant = pickle_safe_load(noise_dequant_path)
                self.register_buffer("dequant_all", noise_dequant)
                self.register_buffer("dequant_class", noise_dequant)
            else:
                self.dequant_all = None
                self.dequant_class = None
        else:
            self.dequant_all = None
            self.dequant_class = None

        
    def get_dtype(self):
        if self.cfg["trainer_precision"] == 32:
            return torch.float32
        else:
            return torch.float64
    
   
    def forward(self, x, y, reverse=False, init=False, use_recorded_outputs=False):
        level_method = self.cfg["level_method"]["choice"]
        level_idx = self.cfg["level_method"]["level_idx"]

        
        out = x
        log_det_sum = x.new_zeros(x.size(0))
        log_p = x.new_zeros(x.size(0))
        outputs = []

        if level_method in ["all_levels"]:
            if not reverse:
                out = preprocess(out, y, self.cfg["bits"],  dequant_all=self.dequant_all, dequant_class=self.dequant_class, noise=torch.rand_like(out), cfg=self.cfg)
                for i in range(self.num_levels):
                    out = squeeze2d(out)
                    out, log_det = self.blocks[i](out, y, init=init, label_encoder=None)
                    log_det_sum = log_det_sum + log_det
                    if i < self.num_levels - 1:
                        out1, out2 = split2d(out, out.size(1) // 2)
                        outputs.append(out2)
                        out = out1

                    # Save the output shape after each level
                    if init != False:
                        self.level_out_shape_arr[i] = torch.tensor(tuple(out.shape[1:]))
                        print(self.level_out_shape_arr)

                out = unsqueeze2d(out)
                for _ in range(self.num_levels - 1):
                    out2 = outputs.pop()
                    out = unsqueeze2d(unsplit2d([out, out2]), factor=2)
            else:
                ch, h, w = self.cfg["ch"], self.cfg["h"], self.cfg["w"]
                out = out.reshape(out.shape[0], ch, h, w)
                out = squeeze2d(out)
                for _ in range(self.num_levels - 1):
                    out1, out2 = split2d(out, out.size(1) // 2)
                    outputs.append(out2)
                    out = squeeze2d(out1)
                for i in reversed(range(self.num_levels)):
                    if i < self.num_levels - 1:
                        out2 = outputs.pop()
                        out = unsplit2d([out, out2])
                    out, log_det = self.blocks[i](out, y, reverse=reverse, label_encoder=self.label_encoder)
                    log_det_sum = log_det_sum + log_det
                    out = unsqueeze2d(out, factor=2)
                
                out = postprocess(out, self.cfg["bits"], self.cfg)

        elif level_method in ["all_levels_idx"]:
            if not reverse:
                self.recorded_outputs = []
                out = preprocess(out, y, self.cfg["bits"],  dequant_all=self.dequant_all, dequant_class=self.dequant_class, noise=torch.rand_like(out), cfg=self.cfg)
                for i in range(self.num_levels):
                    out = squeeze2d(out)
                    out, log_det = self.blocks[i](out, y, init=init, label_encoder=None)
                    log_det_sum = log_det_sum + log_det
                   
                    if i < self.num_levels - 1:
                        out1, out2 = split2d(out, out.size(1) // 2)
                        if i < level_idx - 1:
                            mean, log_sd = self.split_coupling_priors[i](out1).chunk(2, 1)
                            log_p_temp = gaussian_log_p(out2, mean, log_sd).view(out1.shape[0], -1).sum(1)
                            log_p += log_p_temp
                            if use_recorded_outputs:
                                self.recorded_outputs.append(copy.deepcopy(out2))
                        else:
                            outputs.append(out2)
                        out = out1

                    if init != False:  # Save the shape before the split
                        self.level_out_shape_arr[i] = torch.tensor(tuple(out.shape[1:]))
                        print(self.level_out_shape_arr)

                out = unsqueeze2d(out)
                for _ in range(len(outputs)):
                    out2 = outputs.pop()
                    out = unsqueeze2d(unsplit2d([out, out2]), factor=2)
                
                if init != False:  # Save the shape before the split
                    for l in range(len(self.level_out_shape_arr)):
                        self.level_out_shape_arr[l] = torch.tensor(tuple(out.shape[1:]))
                    print(self.level_out_shape_arr)


            else:
                level_diff = (self.num_levels - level_idx)
                last_shape = self.level_out_shape_arr[-1]
                ch_l, h_l, w_l = last_shape[0].item(), last_shape[1].item(), last_shape[2].item()
                out = out.reshape(out.shape[0], ch_l, h_l, w_l)
                out = squeeze2d(out)
                for _ in range(level_diff):
                    out1, out2 = split2d(out, out.size(1) // 2)
                    outputs.append(out2)
                    out = squeeze2d(out1)
                for i in reversed(range(self.num_levels)):
                    if i < self.num_levels - 1: # Only do this after the first level in reverse
                        if len(outputs) != 0:
                            out2 = outputs.pop()
                            out = unsplit2d([out, out2])
                        else:
                            mean, log_sd = self.split_coupling_priors[i](out).chunk(2, 1)
                            out_new = gaussian_sample(torch.randn_like(log_sd), mean, log_sd, self.cfg["T"]).reshape(out.shape)
                            out = unsplit2d([out, out_new])

                    out, log_det = self.blocks[i](out, y, reverse=reverse, label_encoder=self.label_encoder)
                    log_det_sum = log_det_sum + log_det
                    out = unsqueeze2d(out, factor=2)
                
                out = postprocess(out, self.cfg["bits"], self.cfg)


        elif level_method in [SPCB]:
            if not reverse:
                self.recorded_outputs = []
                out = preprocess(out, y, self.cfg["bits"],  dequant_all=self.dequant_all, dequant_class=self.dequant_class, noise=torch.rand_like(out), cfg=self.cfg)
                for i in range(self.num_levels):
                    out = squeeze2d(out)
                    out, log_det = self.blocks[i](out, y, init=init, label_encoder=None)
                    log_det_sum = log_det_sum + log_det

                    if i < self.num_levels - 1:
                        out1, out2 = split2d(out, out.size(1) // 2)
                        out = out1  

                        mean, log_sd = self.split_coupling_priors[i](out).chunk(2, 1)
                        log_p_temp = gaussian_log_p(out2, mean, log_sd).view(out.shape[0], -1).sum(1)
                        log_p += log_p_temp

                        if use_recorded_outputs:
                            self.recorded_outputs.append(out2)

                    # Save the output shape after each level
                    if init != False:
                        self.level_out_shape_arr[i] = torch.tensor(tuple(out.shape[1:]))
                        print(self.level_out_shape_arr)
                    
            else:
                last_shape = self.level_out_shape_arr[-1]
                ch_l, h_l, w_l = last_shape[0].item(), last_shape[1].item(), last_shape[2].item()
                out = out.reshape(out.shape[0], ch_l, h_l, w_l)
                for i in reversed(range(self.num_levels)):
                    out, log_det = self.blocks[i](out, y, reverse=reverse, label_encoder=self.label_encoder)
                    log_det_sum = log_det_sum + log_det
                    out = unsqueeze2d(out, factor=2)

                    if i-1 >= 0:
                        mean, log_sd = self.split_coupling_priors[i-1](out).chunk(2, 1)
                        out_new = gaussian_sample(torch.randn_like(log_sd), mean, log_sd, self.cfg["T"]).reshape(out.shape)
                        if use_recorded_outputs:
                            out_new = self.recorded_outputs.pop()
                        out = unsplit2d([out, out_new])

                out = postprocess(out, self.cfg["bits"], self.cfg)
                self.recorded_outputs = []
        else:
            sys.exit(f"The level method {level_method} is not defined in the forward of MEF")


        return out, log_det_sum, log_p


class Sequential(nn.Sequential):

    def forward(self, x, y, reverse=False, init=False, label_encoder=None):
        if not reverse:
            log_det_sum = x.new_zeros(x.size(0))
            for module in self._modules.values():
                x, log_det = module(x, y, init=init, label_encoder=label_encoder)
                log_det_sum = log_det_sum + log_det
        else:
            log_det_sum = x.new_zeros(x.size(0))
            for module in reversed(self._modules.values()):
                x, log_det = module(x, y, reverse=reverse, label_encoder=label_encoder)
                log_det_sum = log_det_sum + log_det

        return x, log_det_sum


class Flow(Sequential):

    def __init__(self, conv_type, flow_type, num_blocks, in_channels, hidden_channels, h, w, cfg, is_last=0, label_encoder=None):
        super(Flow, self).__init__()
        self.add_module('actnorm', ActNorm(in_channels, h, w))
        self.add_module('conv1x1', Conv1x1(in_channels, conv_type))
        self.add_module('couplinglayer', CouplingLayer(flow_type, num_blocks, in_channels, hidden_channels, label_encoder=label_encoder, cfg=cfg))

