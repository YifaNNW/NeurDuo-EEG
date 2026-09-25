"""Pretraining wrapper in which the backbone predicts the frozen tokenizer's code of the next chunk for every channel."""

from dataclasses import fields

import torch
from torch import nn

from eeg_fm.config import EEGFMConfig
from eeg_fm.loss import next_token_loss
from eeg_fm.model import EEGFM
from eeg_fm.state import EEGMetadataBatch


class EEGFMPretrainer(nn.Module):
    def __init__(
        self,
        config: EEGFMConfig,
        *,
        tokenizer_ckpt: str,
    ) -> None:
        super().__init__()
        self.model = EEGFM(config)
        self.tokenizer = self._load_tokenizer(config, tokenizer_ckpt)

    @staticmethod
    def _load_tokenizer(config: EEGFMConfig, checkpoint_path: str):
        from eeg_fm.modules.vq_tokenizer import SpectralVQTokenizer
        from eeg_fm.tokenizer_config import TokenizerConfig

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        known = {f.name for f in fields(TokenizerConfig)}
        tokenizer_config = TokenizerConfig(**{k: v for k, v in payload["config"].items() if k in known})
        tokenizer = SpectralVQTokenizer(tokenizer_config)
        tokenizer.load_state_dict(payload["tokenizer"])

        if tokenizer_config.codebook_size != config.codebook_size:
            raise ValueError(
                "Tokenizer and model codebook sizes differ: "
                f"{tokenizer_config.codebook_size} != {config.codebook_size}."
            )

        for parameter in tokenizer.parameters():
            parameter.requires_grad = False
        tokenizer.eval()
        return tokenizer

    def train(self, mode: bool = True) -> "EEGFMPretrainer":
        super().train(mode)
        self.tokenizer.eval()
        return self

    @staticmethod
    def _flatten_channel_targets(target_codes: torch.Tensor) -> torch.Tensor:
        batch_size, num_steps, num_channels = target_codes.shape
        return target_codes.permute(0, 2, 1).reshape(
            batch_size * num_channels,
            num_steps,
        )

    def forward(
        self,
        chunks: torch.Tensor,
        metadata: EEGMetadataBatch,
    ) -> dict:
        fused_states = self.model(chunks, metadata)
        predictions = self.model.future_latent_predictor(fused_states)

        with torch.no_grad(), torch.autocast(device_type=chunks.device.type, enabled=False):
            target_codes = self.tokenizer.get_codebook_indices(chunks, metadata)
        target_codes = self._flatten_channel_targets(target_codes)

        loss = next_token_loss(predictions, target_codes)
        return {"loss": loss}
