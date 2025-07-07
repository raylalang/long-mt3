import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from .transcription import MT3Transcription
from .data_pipeline import MT3DataPipeline
from .vocabularies import build_codec, VocabularyConfig
from .contrib.spectrograms import SpectrogramConfig


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    vocab_config = VocabularyConfig()
    codec = build_codec(vocab_config)

    spec_config = SpectrogramConfig(**cfg.data.spectrogram_config)
    data_module = MT3DataPipeline(
        data_list_path=cfg.data.data_list_path,
        spectrogram_config=spec_config,
        codec=codec,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    model_config = {
        "input_dim": cfg.model.input_dim,
        "vocab_size": codec.num_classes,
        "n_layers": cfg.model.n_layers,
        "n_heads": cfg.model.n_heads,
        "d_model": cfg.model.d_model,
        "dropout": cfg.model.dropout,
    }

    model = MT3Transcription(
        model_config=model_config, learning_rate=cfg.train.learning_rate
    )
    trainer = pl.Trainer(max_epochs=cfg.train.max_epochs)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
