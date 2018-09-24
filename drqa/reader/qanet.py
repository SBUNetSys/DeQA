# -*- coding: utf-8 -*-
"""
Main model architecture.
reference: https://github.com/andy840314/QANet-pytorch-
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.cnn import DepthwiseSeparableConv

# revised two things: head set to 1, d_model set to 96


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)  # !!!!!!!!!!!!!!!  do we need * mask after target?


class InitializedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, groups=1,
                 relu=False, bias=False):
        super().__init__()
        self.out = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding, groups=groups, bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu is True:
            return F.relu(self.out(x))
        else:
            return self.out(x)


def encode_position(x, min_timescale=1.0, max_timescale=1.0e4):
    x = x.transpose(1, 2)
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    if x.is_cuda:
        signal = signal.to(x.get_device())
    return (x + signal).transpose(1, 2)


def get_timing_signal(length, channels,
                      min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                        padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))


class Highway(nn.Module):
    def __init__(self, layer_num, size):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([InitializedConv1d(size, size, relu=False, bias=True) for _ in range(self.n)])
        self.gate = nn.ModuleList([InitializedConv1d(size, size, bias=True) for _ in range(self.n)])

    def forward(self, x):
        # x: shape [batch_size, hidden_size, length]
        dropout = 0.1
        for i in range(self.n):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = F.dropout(nonlinear, p=dropout, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
        return x


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_head, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.dropout = dropout
        self.mem_conv = InitializedConv1d(in_channels=d_model, out_channels=d_model * 2, kernel_size=1, relu=False,
                                          bias=False)
        self.query_conv = InitializedConv1d(in_channels=d_model, out_channels=d_model, kernel_size=1, relu=False,
                                            bias=False)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, queries, mask):
        memory = queries

        memory = self.mem_conv(memory)
        query = self.query_conv(queries)
        memory = memory.transpose(1, 2)
        query = query.transpose(1, 2)
        Q = self.split_last_dim(query, self.num_head)
        K, V = [self.split_last_dim(tensor, self.num_head) for tensor in torch.split(memory, self.d_model, dim=2)]

        key_depth_per_head = self.d_model // self.num_head
        Q *= key_depth_per_head ** -0.5
        x = self.dot_product_attention(Q, K, V, mask=mask)
        return self.combine_last_two_dim(x.permute(0, 2, 1, 3)).transpose(1, 2)

    def dot_product_attention(self, q, k, v, bias=False, mask=None):
        """dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        is_training: a bool of training
        scope: an optional string
        Returns:
        A Tensor.
        """
        logits = torch.matmul(q, k.permute(0, 1, 3, 2))
        if bias:
            logits += self.bias
        if mask is not None:
            shapes = [x if x != None else -1 for x in list(logits.size())]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])
            logits = mask_logits(logits, mask)
        weights = F.softmax(logits, dim=-1)
        # dropping out the attention links for each of the heads
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return torch.matmul(weights, v)

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)

    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret


class EmbeddingLayer(nn.Module):
    def __init__(self, wemb_dim, cemb_dim, d_model,
                 dropout_w=0.1, dropout_c=0.05):
        super().__init__()
        self.conv2d = nn.Conv2d(cemb_dim, d_model, kernel_size=(1, 5), padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        self.conv1d = InitializedConv1d(wemb_dim + d_model, d_model, bias=False)
        self.high = Highway(2, d_model)
        self.dropout_w = dropout_w
        self.dropout_c = dropout_c

    def forward(self, ch_emb, wd_emb, length=None):
        ch_emb = ch_emb.permute(0, 3, 1, 2)
        ch_emb = F.dropout(ch_emb, p=self.dropout_c, training=self.training)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)

        wd_emb = F.dropout(wd_emb, p=self.dropout_w, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)
        emb = torch.cat([ch_emb, wd_emb], dim=1)
        emb = self.conv1d(emb)
        emb = self.high(emb)
        return emb


class EncoderBlock(nn.Module):
    def __init__(self, conv_num, d_model, num_head, k, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(d_model, d_model, k) for _ in range(conv_num)])
        self.self_att = SelfAttention(d_model, num_head, dropout=dropout)
        self.FFN_1 = InitializedConv1d(d_model, d_model, relu=True, bias=True)
        self.FFN_2 = InitializedConv1d(d_model, d_model, bias=True)
        self.norm_C = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(conv_num)])
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.conv_num = conv_num
        self.dropout = dropout

    def forward(self, x, mask, l, blks):
        total_layers = (self.conv_num + 1) * blks
        dropout = self.dropout
        out = encode_position(x)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out.transpose(1, 2)).transpose(1, 2)
            if i % 2 == 0:
                out = F.dropout(out, p=dropout, training=self.training)
            out = conv(out)
            out = self.layer_dropout(out, res, dropout * float(l) / total_layers)
            l += 1
        res = out
        out = self.norm_1(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.self_att(out, mask)
        out = self.layer_dropout(out, res, dropout * float(l) / total_layers)
        l += 1
        res = out

        out = self.norm_2(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.FFN_1(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(out, res, dropout * float(l) / total_layers)
        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


class CQAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        w4C = torch.empty(d_model, 1)
        w4Q = torch.empty(d_model, 1)
        w4mlu = torch.empty(1, 1, d_model)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)
        self.dropout = dropout

    def forward(self, C, Q, Cmask, Qmask):
        C = C.transpose(1, 2)
        Q = Q.transpose(1, 2)
        batch_size_c = C.size()[0]
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        S = self.trilinear_for_attention(C, Q)
        Cmask = Cmask.view(batch_size_c, Lc, 1)
        Qmask = Qmask.view(batch_size_c, 1, Lq)
        S1 = F.softmax(mask_logits(S, Qmask), dim=2)
        S2 = F.softmax(mask_logits(S, Cmask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out.transpose(1, 2)

    def trilinear_for_attention(self, C, Q):
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        dropout = self.dropout
        C = F.dropout(C, p=dropout, training=self.training)
        Q = F.dropout(Q, p=dropout, training=self.training)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, Lc, -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1, 2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res


import logging

logger = logging.getLogger(__name__)


class Pointer(nn.Module):
    def __init__(self, d_model, normalize=True):
        super(Pointer, self).__init__()
        self.normalize = normalize
        self.w1 = InitializedConv1d(d_model * 2, 1)
        self.w2 = InitializedConv1d(d_model * 2, 1)

    def forward(self, M1, M2, M3, mask):
        x1 = torch.cat([M1, M2], dim=1)
        x2 = torch.cat([M1, M3], dim=1)
        y1 = mask_logits(self.w1(x1).squeeze(), mask)
        y2 = mask_logits(self.w2(x2).squeeze(), mask)
        # logger.info('y1: %s' % y1)
        # logger.info('y2: %s' % y2)
        if self.normalize:
            if self.training:
                # In training we output log-softmax for NLL
                p_s = F.log_softmax(y1, dim=1)  # [B, S]
                p_e = F.log_softmax(y2, dim=1)  # [B, S]
            else:
                # ...Otherwise 0-1 probabilities
                p_s = F.softmax(y1, dim=1)  # [B, S]
                p_e = F.softmax(y2, dim=1)  # [B, S]
        else:
            p_s = y1.exp()
            p_e = y2.exp()
        # logger.info('p_s: %s' % p_s)
        # logger.info('p_e: %s' % p_e)

        return p_s, p_e


class QANet(nn.Module):
    def __init__(self, args, normalize=True):
        super(QANet, self).__init__()
        # Store config
        self.args = args

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)

        # Char embeddings (+1 for padding)
        self.char_embedding = nn.Embedding(args.char_size,
                                           args.char_embedding_dim,
                                           padding_idx=0)

        d_model = args.hidden_size
        self.dropout = args.dropout
        self.emb = EmbeddingLayer(args.embedding_dim, args.char_embedding_dim, d_model)
        num_head = args.num_head
        self.emb_enc = EncoderBlock(conv_num=4, d_model=d_model, num_head=num_head, k=7, dropout=self.dropout)
        self.cq_att = CQAttention(d_model=d_model)
        self.cq_resizer = InitializedConv1d(d_model * 4, d_model)
        self.model_enc_blks = nn.ModuleList(
            [EncoderBlock(conv_num=2, d_model=d_model, num_head=num_head, k=5, dropout=self.dropout)
             for _ in range(7)])
        self.out = Pointer(d_model, normalize=normalize)
        self.PAD = 0

    def forward(self, cw_idx, cc_idx, c_f, c_mask, qw_idx, qc_idx, q_mask):
        c_mask = (torch.ones_like(cw_idx) * self.PAD != cw_idx).float()
        q_mask = (torch.ones_like(qw_idx) * self.PAD != qw_idx).float()
        cw_emb, cc_emb = self.embedding(cw_idx), self.char_embedding(cc_idx)
        qw_emb, qc_emb = self.embedding(qw_idx), self.char_embedding(qc_idx)
        if self.args.dropout_emb > 0:
            cw_emb = F.dropout(cw_emb, p=self.args.dropout_emb, training=self.training)
            qw_emb = F.dropout(qw_emb, p=self.args.dropout_emb, training=self.training)
            cc_emb = F.dropout(cc_emb, p=self.args.dropout_emb, training=self.training)
            qc_emb = F.dropout(qc_emb, p=self.args.dropout_emb, training=self.training)

        c_emb, q_emb = self.emb(cc_emb, cw_emb), self.emb(qc_emb, qw_emb)
        c_enc_emb = self.emb_enc(c_emb, c_mask, 1, 1)
        q_enc_emb = self.emb_enc(q_emb, q_mask, 1, 1)
        cq_sim_x = self.cq_att(c_enc_emb, q_enc_emb, c_mask, q_mask)
        m_0 = self.cq_resizer(cq_sim_x)
        # m_0 = F.dropout(m_0, p=self.dropout, training=self.training)
        enc = [m_0]
        for i in range(3):
            if i % 2 == 0:  # dropout every 2 blocks
                enc[i] = F.dropout(enc[i], p=self.dropout, training=self.training)

            blk_outputs = [enc[i]]
            for j, blk in enumerate(self.model_enc_blks):
                blk_in = blk_outputs[-1]
                blk_outputs.append(blk(blk_in, c_mask, j * (2 + 2) + 1, 7))
            enc.append(blk_outputs[-1])
        start_scores, end_scores = self.out(enc[1], enc[2], enc[3], c_mask)

        return start_scores, end_scores
