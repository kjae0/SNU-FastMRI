from typing import List, Tuple

import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_engine_cls(stage):
    if stage == 1:
        return Stage1Engine
    elif stage == 2:
        return Stage2Engine
    elif stage == 3:
        return Stage3Engine
    else:
        raise ValueError("Invalid stage number")

class Stage1Engine(nn.Module):
    def __init__(
        self,
        sens_net,
        model
    ):
        super().__init__()
        self.sens_net = sens_net
        self.model = model
        
    def train(self):
        self.sens_net.train()
        self.model.train()
        
    def eval(self):
        self.sens_net.eval()
        self.model.eval()

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor):
        sens_maps = self.sens_net(masked_kspace, mask.int())
        output = self.model(masked_kspace, mask, sens_maps, 0)
        
        return output
    
    
class Stage2Engine(nn.Module):
    def __init__(
        self,
        sens_net,
        model
    ):
        super().__init__()

        self.sens_net = sens_net
        self.model = model
        self.current_epoch = 0

    def train(self):
        self.sens_net.eval()
        self.model.train()
    
    def eval(self):
        self.sens_net.eval()
        self.model.eval()

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor, epoch=0):
        self.current_epoch = epoch
        
        self.sens_net.eval()
        with torch.no_grad():
            sens_maps = self.sens_net(masked_kspace, mask.int())
            
        output = self.model(masked_kspace, mask, sens_maps, 0)
        
        return output
    
class Stage3Engine(nn.Module):
    def __init__(
        self,
        sens_net,
        prev_models,
        model,
        cache=False
    ):
        super().__init__()

        assert isinstance(prev_models, list)
        
        self.cache = cache
        self.sens_net = sens_net
        self.prev_models = prev_models
        self.model = model
        self.current_epoch = 0
    
    def train(self):
        self.sens_net.eval()
        for prev_model in self.prev_models:
            prev_model.eval()
        self.model.train()
        
    def eval(self):
        self.sens_net.eval()
        for prev_model in self.prev_models:
            prev_model.eval()
        self.model.eval()

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor, epoch=0):
        self.current_epoch = epoch
        
        self.sens_net.eval()
        with torch.no_grad():
            sens_maps = self.sens_net(masked_kspace, mask.int())
            
            for idx, prev_model in enumerate(self.prev_models):
                prev_model.eval()
                if idx == 0:
                    output = prev_model(masked_kspace, mask, sens_maps, 0)
                else:
                    output = prev_model(output['kspace'], mask, sens_maps)
                    
        output = self.model(output['kspace'], mask, sens_maps)
        
        return output
    