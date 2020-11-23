import torch
import torch.nn as nn
import torch.nn.functional as F


class S2S(nn.Module):

    def __init__(self, encoder, decoder, output_size, gen_len, device):

        super().__init__()

        self.gen_len = gen_len - 1  # delete the <sos> in sentences
        self.encoder = encoder
        self.decoder = decoder
        self.output_size = output_size
        self.device = device

    def forward(self, theme, keyword, src, tgt, teach):
        attns = dict()
        attns["std"] = []
        attns["coverage"] = []
        tgt_len, batch_size = tgt.size()
        encoder_output, encoder_hidden = self.encoder(theme,
                                                      keyword, src)  # src_len, batch, hidden; 1, batch, hidden
        decoder_hidden = encoder_hidden.view(1, batch_size, -1)  # 1, batch, hidden
        decoder_outputs = torch.ones(self.gen_len, batch_size, self.output_size, device=self.device)  # tgt_len-1, batch
        tgt_start = torch.full((1, batch_size), 2, dtype=torch.long, device=self.device)  # 1, batch
        decoder_attn_coverage = None
        for i in range(tgt_len - 1):
            if i == 0:
                cur_tgt = tgt_start
                g_t = None
            else:
                g_t = tgt[i].view(1, batch_size)  # 1, batch
            decoder_output, decoder_hidden, decoder_attn_std, decoder_attn_coverage = self.decoder(cur_tgt,
                                                                                                   decoder_hidden,
                                                                                                   encoder_output,
                                                                                                   teach,
                                                                                                   g_t,
                                                                                                   decoder_attn_coverage)

            if attns["std"] is None:
                attns["std"] = decoder_attn_std
                attns["coverage"] = decoder_attn_coverage
            else:
                attns["std"].append(decoder_attn_std)
                attns["coverage"].append(decoder_attn_coverage)
            tem = decoder_output.view(1, batch_size, -1)  # 1, batch, output
            tem = F.log_softmax(tem, -1)  # 1, batch, output
            decoder_outputs[i] = tem
            cur_tgt = tem.max(2)[1].view(1, batch_size)  # 1, batch

        return decoder_outputs, attns
