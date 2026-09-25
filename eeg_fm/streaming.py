"""Stateful inference that runs the backbone one 0.5 s chunk at a time."""

from __future__ import annotations

import torch

from eeg_fm.modules.memory_bank import MemoryBank
from eeg_fm.modules.short_queue import ShortQueue
from eeg_fm.state import MambaBlockState, MambaStackState


class StreamRunner:
    def __init__(self, model, metadata, num_channels: int, batch: int = 1,
                 device=None, dtype=torch.float32):
        self.m, self.md, self.C, self.B = model, metadata, num_channels, batch
        self.cfg = model.config
        self.dev = device or next(model.parameters()).device
        self.dtype = dtype
        self.short_queue = ShortQueue(self.cfg)
        self.memory_bank = MemoryBank(self.cfg)
        self.reset()

    def reset(self):
        eb = self.B * self.C
        self.t = 0
        self.fast_state = self.m.fast_backbone.init_state(eb, device=self.dev, dtype=self.dtype)
        self.slow_state = self.m.slow_backbone.init_state(eb, device=self.dev, dtype=self.dtype)
        self.queue = self.short_queue.init_state(eb, device=self.dev, dtype=self.dtype)
        self.bank = self.memory_bank.init_state(eb, device=self.dev, dtype=self.dtype)
        self.slow_out = None
        self.missing = (self.md.missing_mask.to(self.dev)
                        if self.md.missing_mask is not None else None)

    def state_bytes(self) -> int:
        n = 0
        for obj in (self.fast_state, self.slow_state, self.queue, self.bank):
            for v in _tensors(obj):
                n += v.numel() * v.element_size()
        return n

    def _fast_step(self, x_t):
        st = self.fast_state
        stack = self.m.fast_backbone.stack
        hidden = self.m.fast_backbone.input_proj(x_t)
        new_ls = []
        for i, (norm, block, ls) in enumerate(zip(stack.norms, stack.blocks, st.layer_states)):
            residual = hidden
            out, conv, ssm = block.step(norm(hidden).unsqueeze(1), ls.conv_state, ls.ssm_state)
            hidden = residual + out.squeeze(1)
            new_ls.append(MambaBlockState(conv_state=conv, ssm_state=ssm))
            if stack.channel_attn is not None and (i + 1) % stack.channel_attn_every == 0:
                j = (i + 1) // stack.channel_attn_every - 1
                hidden = hidden + stack.channel_attn[j](
                    hidden.unsqueeze(1), self.C, self.missing).squeeze(1)
        h = stack.final_norm(hidden)
        self.fast_state = MambaStackState(layer_states=tuple(new_ls), last_output=h)
        return h

    @torch.no_grad()
    def step(self, chunk):
        cfg = self.cfg
        tok = self.m.encode_sequence(chunk.unsqueeze(1), self.md)
        x_t = tok.permute(0, 2, 1, 3).reshape(self.B * self.C, -1)

        h = self._fast_step(x_t)
        self.queue = self.short_queue.append(self.queue, h)

        if (self.t + 1) % cfg.write_every_n_steps == 0:
            size = min(min(self.t + 1, cfg.queue_size), cfg.writer_window)
            win = self.short_queue.get_window(self.queue, size)
            mk = self.m.memory_writer(win)
            s_out, self.slow_state = self.m.slow_backbone.forward_step(mk, self.slow_state)
            self.slow_out = s_out
            self.bank = self.memory_bank.append(self.bank, mk)

        Ds = cfg.slow_dim
        slow = (self.slow_out if self.slow_out is not None
                else torch.zeros(self.B * self.C, Ds, device=self.dev, dtype=h.dtype))
        slow = slow + self.m.memory_context(h, self.bank)
        self.t += 1
        return self.m.readout(h, slow)


def _tensors(obj):
    if torch.is_tensor(obj):
        yield obj
        return
    if isinstance(obj, (list, tuple)):
        for o in obj:
            yield from _tensors(o)
        return
    for a in getattr(obj, "__dataclass_fields__", {}) or {}:
        yield from _tensors(getattr(obj, a))
