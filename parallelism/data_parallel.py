import torch.nn as nn
from torch.nn.parallel import DataParallel as DP

class DataParallel(nn.Module):
    def __init__(self, model):
        super(DataParallel, self).__init__()
        self.model = DP(model)

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask)

