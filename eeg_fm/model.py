"""NeurDuo-EEG backbone: chunk encoder, per-channel fast stream, memory writer, slow stream and memory retrieval."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from eeg_fm.config import EEGFMConfig
from eeg_fm.modules.fast_backbone import FastCausalMamba
from eeg_fm.modules.heads import FutureTokenPredictor
from eeg_fm.modules.memory_context import MemoryContext
from eeg_fm.modules.memory_writer import MemoryWriter
from eeg_fm.modules.metadata import MetadataEncoder
from eeg_fm.modules.readout import SplicedReadout
from eeg_fm.modules.signal_patch import SignalPatchEncoder
from eeg_fm.modules.slow_backbone import SlowCausalSSM
from eeg_fm.modules.spatial_mixer import LightweightSpatialMixer
from eeg_fm.state import EEGMetadataBatch


class EEGFM(nn.Module):
    def __init__(self, config: EEGFMConfig) -> None:
        super().__init__()
        self.config = config
        self.signal_patch_encoder = SignalPatchEncoder(config)
        self.metadata_encoder = MetadataEncoder(config)
        self.spatial_mixer = LightweightSpatialMixer(config)
        self.fast_backbone = FastCausalMamba(config)
        self.memory_writer = MemoryWriter(config)
        self.slow_backbone = SlowCausalSSM(config)
        self.memory_context = MemoryContext(config)
        self.readout = SplicedReadout(config)
        self.future_latent_predictor = FutureTokenPredictor(config)

    def _frontend_segments(self, chunks: torch.Tensor, metadata: EEGMetadataBatch):
        if chunks.dim() != 4:
            raise ValueError(
                "Expected chunks with shape [batch, num_steps, num_channels, chunk_samples], "
                f"got {tuple(chunks.shape)}."
            )
        batch_size, num_steps, num_channels, chunk_samples = chunks.shape
        device = chunks.device
        use_ckpt = self.config.use_gradient_checkpointing and self.training and torch.is_grad_enabled()
        metadata_tokens = self.metadata_encoder(metadata, num_channels, device=device)

        max_rows = max(1, int(getattr(self.config, "frontend_max_rows", 8192)))
        rows_per_step = max(1, batch_size * num_channels)
        steps_per_seg = max(1, min(num_steps, max_rows // rows_per_step))

        for start in range(0, num_steps, steps_per_seg):
            end = min(start + steps_per_seg, num_steps)
            seg = end - start
            flat = chunks[:, start:end].reshape(batch_size * seg, num_channels, chunk_samples)
            if use_ckpt:
                signal_tokens = checkpoint(self.signal_patch_encoder, flat, use_reentrant=False)
            else:
                signal_tokens = self.signal_patch_encoder(flat)
            summed = signal_tokens.view(batch_size, seg, num_channels, -1) + metadata_tokens.unsqueeze(1)
            summed = summed.reshape(batch_size * seg, num_channels, -1)
            yield batch_size, seg, num_channels, summed

    def encode_sequence(
        self,
        chunks: torch.Tensor,
        metadata: EEGMetadataBatch,
    ) -> torch.Tensor:
        segments = []
        for batch_size, seg, num_channels, summed in self._frontend_segments(chunks, metadata):
            channel_tokens = self.spatial_mixer.norm(summed)
            segments.append(channel_tokens.view(batch_size, seg, num_channels, -1))
        return torch.cat(segments, dim=1)

    def _memory_slot_index(self, num_steps: int, device: torch.device) -> torch.Tensor:
        n = self.config.write_every_n_steps
        steps = torch.arange(num_steps, device=device)
        return (steps + 1) // n - 1

    def _encode_fast_sequence(
        self,
        chunks: torch.Tensor,
        metadata: EEGMetadataBatch,
    ) -> torch.Tensor:
        batch_size, num_steps, num_channels, _ = chunks.shape
        channel_tokens = self.encode_sequence(chunks, metadata)
        chunk_tokens = channel_tokens.permute(0, 2, 1, 3).reshape(
            batch_size * num_channels,
            num_steps,
            -1,
        )
        missing = metadata.missing_mask
        if missing is not None:
            missing = missing.to(chunks.device)
        fast_states = self.fast_backbone.forward_sequence(
            chunk_tokens,
            num_channels=num_channels,
            missing=missing,
        )
        return fast_states

    def _write_memory(
        self,
        fast_states: torch.Tensor,
    ) -> torch.Tensor | None:
        num_steps = fast_states.size(1)
        config = self.config
        write_steps = [
            step
            for step in range(num_steps)
            if (step + 1) % config.write_every_n_steps == 0
        ]
        if not write_steps:
            return None

        window_sizes = {
            step: min(step + 1, config.queue_size, config.writer_window)
            for step in write_steps
        }
        full_steps = [
            step for step in write_steps if window_sizes[step] == config.writer_window
        ]
        token_by_step: dict[int, torch.Tensor] = {}

        if full_steps:
            indices = torch.tensor(
                [
                    [step - config.writer_window + 1 + offset for offset in range(config.writer_window)]
                    for step in full_steps
                ],
                device=fast_states.device,
                dtype=torch.long,
            )
            windows = fast_states[:, indices.reshape(-1)]
            windows = windows.reshape(
                fast_states.size(0) * len(full_steps),
                config.writer_window,
                -1,
            )
            tokens = self.memory_writer(windows)
            tokens = tokens.reshape(fast_states.size(0), len(full_steps), -1)
            for index, step in enumerate(full_steps):
                token_by_step[step] = tokens[:, index]

        for step in write_steps:
            if step in token_by_step:
                continue
            size = window_sizes[step]
            token_by_step[step] = self.memory_writer(
                fast_states[:, step - size + 1 : step + 1]
            )

        return torch.stack([token_by_step[step] for step in write_steps], dim=1)

    def _scan_slow_sequence(
        self,
        fast_states: torch.Tensor,
        memory_tokens: torch.Tensor | None,
        slot_index: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_steps = fast_states.shape[:2]
        if memory_tokens is None:
            return torch.zeros(
                batch_size,
                num_steps,
                self.config.slow_dim,
                device=fast_states.device,
                dtype=fast_states.dtype,
            )

        slow_sequence = self.slow_backbone.forward_sequence(memory_tokens)
        gather_index = slot_index.clamp(min=0).view(1, num_steps, 1)
        gather_index = gather_index.expand(batch_size, num_steps, slow_sequence.size(-1))
        slow_states = torch.gather(slow_sequence, 1, gather_index)
        available = (slot_index >= 0).view(1, num_steps, 1).to(slow_states.dtype)
        return slow_states * available

    def _add_memory_context(
        self,
        fast_states: torch.Tensor,
        slow_states: torch.Tensor,
        memory_tokens: torch.Tensor | None,
        slot_index: torch.Tensor,
    ) -> torch.Tensor:
        tokens = memory_tokens
        if tokens is None:
            tokens = fast_states[:, :0, : self.config.slow_dim]
        context = self.memory_context.forward_sequence(
            fast_states,
            tokens,
            slot_index,
            self.config.memory_bank_size,
        )
        return slow_states + context

    def forward(
        self,
        chunks: torch.Tensor,
        metadata: EEGMetadataBatch,
    ) -> torch.Tensor:
        _, num_steps, _, _ = chunks.shape
        device = chunks.device
        fast_states = self._encode_fast_sequence(chunks, metadata)
        memory_tokens = self._write_memory(fast_states)
        slot_index = self._memory_slot_index(num_steps, device)
        slow_states = self._scan_slow_sequence(fast_states, memory_tokens, slot_index)
        slow_states = self._add_memory_context(
            fast_states,
            slow_states,
            memory_tokens,
            slot_index,
        )
        return self.readout(fast_states, slow_states)
