import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import WeightNorm


class Prototype_Classifier(nn.Module):

    def __init__(self, args, case=None):
        super().__init__()

        self.args = args
        if case == 'input_size':
            self.prototypes = nn.Parameter(torch.randn(self.args.input_size, self.args.num_proto), requires_grad=True)
        else:
            self.prototypes = nn.Parameter(torch.randn(self.args.input_emb, self.args.num_proto), requires_grad=True)

        self.pdist = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.transform = nn.Softmax(dim=1)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            nn.init.xavier_normal_(weight)
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, cls_token):

        cls_token = cls_token.unsqueeze(-1)
        cls_token_p = self.prototypes.expand(cls_token.shape[0], -1, -1)

        dist_sim = self.pdist(cls_token, cls_token_p)
        scores = self.transform(dist_sim)

        return scores

