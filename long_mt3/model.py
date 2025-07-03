import torch
import torch.nn as nn
import torch.nn.functional as F


class MT3Encoder(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, n_heads, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout),
            num_layers=n_layers,
        )

    def forward(self, x, src_key_padding_mask=None):
        # x: (batch, frames, input_dim)
        x = self.input_proj(x)
        # Transformer expects (seq, batch, d_model)
        x = x.transpose(0, 1)
        enc = self.encoder_layers(x, src_key_padding_mask=src_key_padding_mask)
        return enc.transpose(0, 1)  # (batch, frames, d_model)


class MT3Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.decoder_layers = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout),
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
        # tgt: (batch, tgt_len)
        x = self.embed(tgt)  # (batch, tgt_len, d_model)
        x = x.transpose(0, 1)  # (tgt_len, batch, d_model)
        output = self.decoder_layers(
            x,
            memory.transpose(0, 1),
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        output = output.transpose(0, 1)  # (batch, tgt_len, d_model)
        logits = self.out_proj(output)  # (batch, tgt_len, vocab_size)
        return logits


class MT3Model(nn.Module):
    def __init__(
        self, input_dim, vocab_size, d_model=512, n_layers=8, n_heads=8, dropout=0.1
    ):
        super().__init__()
        self.encoder = MT3Encoder(input_dim, d_model, n_layers, n_heads, dropout)
        self.decoder = MT3Decoder(vocab_size, d_model, n_layers, n_heads, dropout)

    def forward(
        self,
        src,
        tgt,
        src_key_padding_mask=None,
        tgt_mask=None,
        tgt_key_padding_mask=None,
    ):
        # src: (batch, frames, input_dim)
        # tgt: (batch, tgt_len)
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        logits = self.decoder(
            tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask
        )
        return logits  # (batch, tgt_len, vocab_size)


if __name__ == "__main__":
    from .vocabularies import build_codec, VocabularyConfig

    config = VocabularyConfig()
    codec = build_codec(config)

    batch_size = 2
    frames = 1024
    n_mels = 229
    tgt_len = 128

    dummy_src = torch.randn(batch_size, frames, n_mels)
    dummy_tgt = torch.randint(0, codec.num_classes, (batch_size, tgt_len))

    model = MT3Model(input_dim=n_mels, vocab_size=codec.num_classes)
    logits = model(dummy_src, dummy_tgt)

    print("Logits shape:", logits.shape)  # should be (2, 128, vocab_size)
