import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, hidden_size):

        super().__init__()

        self.hidden_size = hidden_size
        self.linear_query = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_context = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.linear_out = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.linear_cover = nn.Linear(1, hidden_size, bias=False)

    def score(self, dec_o, enc_o):
        batch_size, tgt_len, _ = dec_o.size()
        src_len = enc_o.size(1)
        a1 = self.linear_query(dec_o.view(-1, self.hidden_size))  # batch*tgt_len, hidden
        a1 = a1.view(batch_size, tgt_len, 1, self.hidden_size)  # batch, tgt_len, 1, hidden
        a1 = a1.expand(batch_size, tgt_len, src_len, self.hidden_size)  # batch, tgt_len, src_len, hidden

        a2 = self.linear_context(enc_o.contiguous().view(-1, self.hidden_size))  # batch*src_len, hidden
        a2 = a2.view(batch_size, 1, src_len, self.hidden_size)
        a2 = a2.expand(batch_size, tgt_len, src_len, self.hidden_size)  # batch, tgt_len, src_len, hidden

        a = torch.tanh(a1 + a2)  # batch, tgt_len, src_len, hidden_size

        return self.v(a.view(-1, self.hidden_size)).view(batch_size, tgt_len, src_len)  # batch, tgt_len, src_len

    def forward(self, attn_dec_state, attn_enc_state, attn_coverage):
        dec_out = attn_dec_state.permute(1, 0, 2)  # batch, tgt_len, hidden
        enc_out = attn_enc_state.permute(1, 0, 2)  # batch, src_len, hidden
        batch_size, target_l, _ = dec_out.size()
        source_l = enc_out.size(1)

        if attn_coverage is not None:
            cover = attn_coverage.view(-1).unsqueeze(1)  # tgt_len*batch*src_len, 1
            a_o = self.linear_cover(cover).view(batch_size, source_l, self.hidden_size)  # batch, src_len, hidden
            enc_out = enc_out + a_o  # batch, src_len, hidden
            enc_out = torch.tanh(enc_out)  # batch, src_len, hidden

        align = self.score(dec_out.contiguous(), enc_out.contiguous())  # batch, tgt_len, src_len
        align_vectors = F.softmax(align.view(batch_size * target_l, source_l), -1)  # batch*tgt_len, src_len
        align_vectors = align_vectors.view(batch_size, target_l, source_l)  # batch, tgt_len, src_len
        c = torch.bmm(align_vectors, enc_out)  # batch, tgt_len, hidden

        concat_c = torch.cat([c, dec_out], 2).view(batch_size * target_l,
                                                   self.hidden_size * 2)  # batch, tgt_len, hidden*2
        attn_h = self.linear_out(concat_c).view(batch_size, target_l, self.hidden_size)  # batch, tgt_len, hidden
        attn_h = attn_h.permute(1, 0, 2).contiguous()  # tgt_len, batch, hidden
        align_vectors = align_vectors.permute(1, 0, 2).contiguous()  # tgt_len, batch, src_len

        return attn_h, align_vectors  # tgt_len, batch, hidden; tgt_len, batch, src_len
