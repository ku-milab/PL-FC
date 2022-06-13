import math
import numpy as np
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.autograd import Variable
from torch.nn.utils.weight_norm import WeightNorm


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_sinusoid_encoding_table(n_position, d_model):

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table)


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.3):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.gelu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


def attention(q, k, v, d_k, mask=None, dropout=None):
    """
    :param q: Batch x n_head x max_seq_len x variable
    :param k: Batch x n_head x max_seq_len x variable
    :param v: Batch x n_head x max_seq_len x variable
    :param d_k:
    :param mask: Batch x n_head x max_seq_len x max_seq_len
    :param dropout:
    :return:
    """

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [Batch x n_head x max_seq_len x max_seq_len]

    if mask is not None:
        scores = scores.masked_fill(mask == 1, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.3):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)  # [batch_size * len_q * n_heads * hidden_dim]
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)  # [batch_size * len_q * n_heads * hidden_dim]
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)  # [batch_size * len_q * n_heads * hidden_dim]

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)  # [batch_size * n_heads * len_q * hidden_dim]
        q = q.transpose(1, 2)  # [batch_size * n_heads * len_q * hidden_dim]
        v = v.transpose(1, 2)  # [batch_size * n_heads * len_q * hidden_dim]

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (max_seq_len ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (max_seq_len ** ((2 * (i + 1)) / d_model)))

        # pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, seq_len):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)

        # add constant to embedding
        # seq_len = x.size(1)
        for i, s in enumerate(seq_len):
            x[i, :s, :] = x[i, :s, :] + Variable(self.pe[:s], requires_grad=False).cuda()
        return x

class EncoderLayer(nn.Module):

    def __init__(self, d_model, heads, d_ff, dropout=0.3):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)

        self.ff = FeedForward(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):

        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2))

        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))

        return x


class Encoding_Block(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, num_stack):
        super().__init__()

        self.N = num_stack
        self.layers = get_clones(EncoderLayer(d_model, num_heads, d_ff), num_stack)
        self.norm = Norm(d_model)

    def forward(self, q):
        # MHA Encoding
        for i in range(self.N):
            q = self.layers[i](q)

        # Normalize
        encoded_data = self.norm(q)
        # encoded_data = q

        return encoded_data

class Classifier(nn.Module):
    def __init__(self, input, hidden, dropout=0.3):
        super().__init__()
        self.cls = nn.Sequential(
                                 nn.Linear(input, hidden),
                                 nn.BatchNorm1d(hidden),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden, 2)
                                 )

    def forward(self, x):
        return self.cls(x)

class NonLinearEmbedding(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.emb = nn.Sequential(nn.Linear(input_size, output_size),
                                 # nn.BatchNorm1d(output_size),
                                 nn.GELU(),
                                 nn.Dropout(0.3),
                                 nn.Linear(output_size, input_size),
                                 # nn.BatchNorm1d(output_size),
                                 nn.GELU())

    def forward(self, x):
        return self.emb(x)


# class NonLinearDecoding(nn.Module):
#     def __init__(self, input_size, output_size):
#         super().__init__()
#
#         self.dec = nn.Sequential(nn.Linear(input_size, output_size),
#                                  nn.BatchNorm1d(output_size),
#                                  nn.GELU(),
#                                  nn.Dropout(0.3),
#                                  nn.Linear(output_size, input_size),
#                                  nn.BatchNorm1d(output_size),
#                                  nn.Tanh())
#
#     def forward(self, x):
#         return self.dec(x)

class NonLinearDecoding(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.dec = nn.Linear(input_size, output_size)
        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(self.dec(x))

class LinearClassifier(nn.Module):
    def __init__(self, input):
        super().__init__()
        self.cls = nn.Linear(input, 2)

    def forward(self, x):
        return self.cls(x)


class FCTransformer_Cls(nn.Module):

    def __init__(self, args):
        # , input_dim, d_model, d_ff, num_stack, num_heads, max_length
        super().__init__()

        self.args = args

        self.embed_fc = NonLinearEmbedding(args.input_size, args.input_emb)
        if args.pos_type == 'wo':
            pass
        elif args.pos_type == 'sincos':
            self.embed_pos = PositionalEncoder(args.input_size, int(args.input_size + 1))

        elif args.pos_type == 'learn':
            self.embed_pos = nn.Parameter(torch.randn(1, int(args.input_size + 1), args.input_emb))

        self.encoding_block = Encoding_Block(args.input_size, args.num_heads, args.d_ff, args.num_stack)
        self.cls_token = nn.Parameter(torch.randn(1, 1, args.input_size), requires_grad=True)

        self.decod_fc = NonLinearDecoding(int(args.input_size*2), args.input_size)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            nn.init.xavier_normal_(weight)
            # stv = 1. / math.sqrt(weight.size(1))
            # nn.init.uniform_(weight, -stv, stv)

    def forward(self, data, mask=None):

        if mask != None:
            data = data * mask

        data_1 = self.embed_fc(data)
        cls = self.cls_token.repeat(data_1.shape[0], 1, 1)

        x_cls = torch.cat([cls, data_1], dim=1)

        if self.args.pos_type == 'wo':
            pass
        elif self.args.pos_type == 'sincos':
            x_cls = self.embed_pos(x_cls, np.repeat(self.args.input_size, data.shape[0]))

        elif self.args.pos_type == 'learn':
            x_cls += self.embed_pos

        middle_output = self.encoding_block(x_cls)
        cls_token = middle_output[:, 0, :]

        cls_token_copy = cls_token.unsqueeze(1).repeat(1, data.shape[1], 1)
        middle_output_cls = torch.cat([middle_output[:, 1:, :], cls_token_copy], -1)
        output = self.decod_fc(middle_output_cls)

        return output, cls_token




class distLinear(nn.Module):

    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm

        self.scale_factor = 1

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)

        cos_dist = self.L(x_normalized) # matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        # cos_dist = self.L(x) # matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor * (cos_dist)

        return scores
