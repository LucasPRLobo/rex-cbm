from torch import nn
import torch
def head_from_concepts(cbm, c):          # c: (N, 112) concept logits
    if cbm.name == "independent":   return cbm.net["head"](torch.sigmoid(c))
    if cbm.name == "sequential":    return cbm.net["head"](c)
    if cbm.name == "joint":         return cbm.net["joint"].sec_model(c)
    if cbm.name == "joint_sigmoid": return cbm.net["joint"].sec_model(torch.sigmoid(c))

class FrozenClassView(nn.Module):
    def __init__(self, cbm, frozen_idx, frozen_logits):
        super().__init__()
        self.cbm, self.idx, self.vals = cbm, frozen_idx, frozen_logits
    def forward(self, x):
        c = self.cbm.concept_logits(x).clone()
        c[:, self.idx] = self.vals          # pin the frozen concepts
        return head_from_concepts(self.cbm, c)   # (N, 200) class logits