import torch
import torch.nn as nn
import torch.nn.functional as F
import math

MAX_LEN = 2048


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
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


class MT3Encoder(nn.Module):
    def __init__(
        self, input_dim, d_model, nhead, dim_feedforward, num_layers, dropout=0.1
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=MAX_LEN)
        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(self, x, src_key_padding_mask=None):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        enc = self.encoder_layers(x, src_key_padding_mask=src_key_padding_mask)
        return enc


class MT3Decoder(nn.Module):
    def __init__(
        self, vocab_size, d_model, nhead, dim_feedforward, num_layers, dropout=0.1
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model, max_len=MAX_LEN)
        self.decoder_layers = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
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
        self,
        input_dim,
        vocab_size,
        d_model=512,
        nhead=6,
        dim_feedforward=1024,
        num_layers=8,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = MT3Encoder(
            input_dim, d_model, nhead, dim_feedforward, num_layers, dropout
        )
        self.decoder = MT3Decoder(
            vocab_size, d_model, nhead, dim_feedforward, num_layers, dropout
        )

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)

        tgt_len = tgt.shape[1]
        tgt_mask = self.generate_square_subsequent_mask(tgt_len, device=tgt.device)
        logits = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return logits

    def generate_square_subsequent_mask(self, sz, device=None):
        mask = torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)
        if device is not None:
            mask = mask.to(device)
        return mask
