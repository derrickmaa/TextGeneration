import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, data_size, hidden_size, dropout):

        super().__init__()

        self.hidden_size = hidden_size
        self.theme_embedding = nn.Embedding(data_size, hidden_size)
        self.keyword_embedding = nn.Embedding(data_size, hidden_size)
        self.src_embedding = nn.Embedding(data_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.enc_w_weight = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.enc_theme_weight = nn.Linear(hidden_size, hidden_size)
        self.enc_keyword_weight = nn.Linear(hidden_size, hidden_size)
        self.enc_weight = nn.Linear(hidden_size, 1)
        self.enc_output_out = nn.Linear(hidden_size * 3, hidden_size, bias=True)
        self.enc_hidden_out = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def point(self, enc_theme, enc_keyword, enc_w_1):
        p_s_l = enc_w_1.size(1)
        batch_size, p_t_l, _ = enc_keyword.size()
        b1 = self.enc_w_weight(enc_w_1.contiguous().view(-1, self.hidden_size * 2))
        b1 = b1.view(batch_size, p_s_l, 1, self.hidden_size)
        b1 = b1.expand(batch_size, p_s_l, p_t_l, self.hidden_size)  # batch, src_len, keyword_len, hidden

        b22 = self.enc_theme_weight(enc_theme.contiguous().view(-1, self.hidden_size))
        b2 = b22.view(batch_size, 1, 1, self.hidden_size)  # batch, 1, 1, hidden(len of theme is 1)
        b2 = b2.expand(batch_size, p_s_l, p_t_l, self.hidden_size)  # batch, src_len, keyword_len, hidden

        b3 = self.enc_keyword_weight(enc_keyword.contiguous().view(-1, self.hidden_size))
        b3 = b3.view(batch_size, 1, p_t_l, self.hidden_size)
        b3 = b3.expand(batch_size, p_s_l, p_t_l, self.hidden_size)  # batch, src_len, keyword_len, hidden

        b = torch.tanh(b1 + b2 + b3)  # batch, src_len, keyword_len, hidden_size

        enc_w = self.enc_weight(b.view(-1, self.hidden_size)).view(batch_size, p_s_l,
                                                                   p_t_l)  # batch, src_len, keyword_len

        return torch.bmm(enc_w, b22.view(batch_size, 1,
                         self.hidden_size).expand(batch_size, p_t_l, self.hidden_size))  # batch, src_len, hidden

    def forward(self, theme, keyword, src):
        batch_size = src.size(1)
        src = src.view(-1, batch_size)  # src_len, batch
        theme = theme.view(-1, batch_size)  # theme_len, batch
        keyword = keyword.view(-1, batch_size)  # keyword_len, batch
        s_l = src.size(0)

        src_emb = self.src_embedding(src)  # src_len, batch, hidden
        theme_emb = self.theme_embedding(theme)  # theme_len, batch, hidden
        keyword_emb = self.keyword_embedding(keyword)  # keyword_len, batch, hidden

        enc_output, enc_hidden = self.rnn(src_emb)  # src_len, batch, hidden*2; 2, batch, hidden
        enc_hidden = torch.cat([enc_hidden[0:enc_hidden.size(0):2],
                                enc_hidden[1:enc_hidden.size(0):2]], 2)  # 1, batch, hidden*2
        enc_w_1 = torch.cat([enc_output[:, :, :self.hidden_size],
                             enc_output[:, :, self.hidden_size:]], 2)  # src_len, batch, hidden*2
        enc_theme = theme_emb.transpose(0, 1)  # batch, theme_len, hidden
        enc_w_1 = enc_w_1.transpose(0, 1)  # batch, src_len, hidden*2
        enc_keyword = keyword_emb.transpose(0, 1)  # batch, keyword_len, hidden
        enc_w_2 = self.point(enc_theme, enc_keyword, enc_w_1)  # batch, src_len, hidden
        concat_enc_w = torch.cat([enc_w_2, enc_w_1], 2).view(batch_size * s_l,
                                                             self.hidden_size * 3)  # batch*src_len, hidden*3
        enc_out = self.enc_output_out(concat_enc_w).view(batch_size, s_l, self.hidden_size)  # batch, src_len, hidden
        enc_out = enc_out.transpose(0, 1).contiguous()  # src_len, batch, hidden
        enc_hidden = self.enc_hidden_out(enc_hidden)  # 1, batch, hidden
        # enc_out = self.dropout(dec_out)

        return enc_out, enc_hidden  # src_len, batch, hidden; 1, batch, hidden
