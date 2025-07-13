import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1)].to(x.device)


def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1).to(torch.float32)


class MT3Encoder(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, n_heads, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
            ),
            num_layers=n_layers,
        )

    def forward(self, x, src_key_padding_mask=None):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        enc = self.encoder_layers(x, src_key_padding_mask=src_key_padding_mask)
        return enc


class MT3Decoder(nn.Module):
    def __init__(
        self, vocab_size, d_model, n_layers, n_heads, dropout=0.1, max_len=1024
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model, max_len=max_len)
        self.decoder_layers = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
            ),
            num_layers=n_layers,
        )
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = self.embed(tgt)
        x = self.pos_decoder(x)
        output = self.decoder_layers(
            x,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.out_proj(output)


class MT3Model(nn.Module):
    def __init__(
        self, input_dim, vocab_size, d_model=512, n_layers=8, n_heads=8, dropout=0.1
    ):
        super().__init__()
        self.encoder = MT3Encoder(input_dim, d_model, n_layers, n_heads, dropout)
        self.decoder = MT3Decoder(vocab_size, d_model, n_layers, n_heads, dropout)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)

        tgt_len = tgt.shape[1]
        tgt_mask = generate_square_subsequent_mask(tgt_len).to(tgt.device)
        logits = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return logits
