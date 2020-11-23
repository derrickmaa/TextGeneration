import torch.nn as nn
import random


class Decoder(nn.Module):

    def __init__(self, hidden_size, data_size, attention, dropout):

        super().__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(data_size, hidden_size)
        self.GRU = nn.GRU(hidden_size, hidden_size)
        self.attn = attention
        self.dec_out = nn.Linear(hidden_size, data_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pre_t, dec_hidden, enc_output, teach, g_t, coverage, dec_coverage=True):  # g_t is ground truth

        if g_t is None:
            dec_emb = self.embedding(pre_t)  # 1, batch, hidden
        else:
            if random.random() <= teach:
                dec_emb = self.embedding(g_t)  # 1, batch, hidden
            else:
                dec_emb = self.embedding(pre_t)  # 1, batch, hidden
        dec_output, dec_hidden = self.GRU(dec_emb, dec_hidden)  # tgt_len, batch, hidden; 1, batch, hidden
        dec_output_attn, dec_cov = self.attn(dec_output, enc_output,
                                             coverage)  # tgt_len, batch, hidden; tgt_len, batch, src_len
        if dec_coverage:
            coverage = dec_cov if coverage is None else dec_cov + coverage  # tgt_len, batch, src_len
        dec_out = self.dec_out(dec_output_attn.contiguous().view(-1, self.hidden_size))  # tgt_len, batch, output
        # dec_out = self.dropout(dec_out)

        return dec_out, dec_hidden, dec_cov, coverage
        # tgt_len, batch, output; 1, batch, hidden; tgt_len, batch, src_len; tgt_len, batch, src_len
