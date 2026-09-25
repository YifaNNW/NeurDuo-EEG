"""Architecture hyperparameters of the NeurDuo-EEG backbone."""

from dataclasses import dataclass, field


@dataclass
class EEGFMConfig:
    chunk_samples: int = 128
    chunk_stride_samples: int = 128
    token_dim: int = 256
    patch_hidden_dim: int = 128
    patch_kernel_sizes: tuple[int, ...] = field(default_factory=lambda: (3, 7, 15))
    metadata_hidden_dim: int = 64
    max_reference_types: int = 64
    coord_dim: int = 3

    fast_dim: int = 256
    slow_dim: int = 192
    fast_d_state: int = 16
    slow_d_state: int = 16
    fast_num_layers: int = 6
    slow_num_layers: int = 3
    fast_expand_factor: int = 2
    slow_expand_factor: int = 2
    fast_conv_kernel_size: int = 4
    slow_conv_kernel_size: int = 3

    channel_attn_every: int = 2
    channel_attn_heads: int = 4

    queue_size: int = 16
    writer_window: int = 8
    memory_bank_size: int = 8
    write_every_n_steps: int = 4

    codebook_size: int = 2048
    predictor_hidden_dim: int = 256

    use_gradient_checkpointing: bool = True
    frontend_max_rows: int = 8192

    @property
    def readout_dim(self) -> int:
        return self.fast_dim + self.slow_dim

    def __post_init__(self) -> None:
        self.patch_kernel_sizes = tuple(self.patch_kernel_sizes)
        if self.fast_num_layers < 1 or self.slow_num_layers < 1:
            raise ValueError("fast_num_layers and slow_num_layers must both be at least 1.")
        if self.fast_expand_factor < 1 or self.slow_expand_factor < 1:
            raise ValueError("expand factors must both be at least 1.")
        if self.fast_conv_kernel_size < 1 or self.slow_conv_kernel_size < 1:
            raise ValueError("conv kernel sizes must both be at least 1.")
