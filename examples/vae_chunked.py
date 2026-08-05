"""Chunked VAE decoding for video models whose internal tiling still OOMs.

Motivation, measured on a 24GB RTX 3090 with the MiniMax H3 video VAE at
480x640 / 22 frames, on an otherwise-empty GPU:

    output size          internal tiling      peak VRAM
    128x128              on                    1.83 GB   ok
    256x256              on                    7.23 GB   ok
    320x416              on                   22.60 GB   OOM
    480x640              on                   22.49 GB   OOM

    480x640, 6 tiles driven manually, one vae.decode() per tile
                                              7.23 GB   ok, 6.8s

Two things are going on:

1. ``comfy/sd.py``'s memory estimate for this VAE is 8-30x low
   (``estimate_decode_memory`` predicts 0.22-0.33 GB against 1.8-7.2 GB actual),
   so model_management never frees enough and the tiled fallback is chosen too
   late to help.
2. ``MiniMaxH3VideoVAE.decode_tiled`` ignores ComfyUI's tile_x/tile_y/overlap
   entirely (``comfy/ldm/minimax/vae.py:678`` is ``return self.decode(z)``), and
   its own ``tiled_decode`` decodes every tile inside a single call, so per-tile
   activations accumulate in the caching allocator instead of being returned
   between tiles.

Issuing one ``vae.decode()`` per spatial tile sidesteps both: each call gets its
own memory reservation, and the result lands on the VAE's output_device (CPU)
before the next tile starts. Overlap blending is done here, on CPU, because the
model's own blending only runs on the path that OOMs.
"""

from __future__ import annotations

import gc
from typing import Any

__all__ = ["decode_video_chunked"]


def _ramp(n: int, device, dtype):
    import torch

    if n <= 0:
        return torch.ones(0, device=device, dtype=dtype)
    return torch.linspace(0.0, 1.0, n + 2, device=device, dtype=dtype)[1:-1]


def _tile_starts(total: int, size: int, overlap: int) -> list[int]:
    """Tile offsets covering ``total`` with the given overlap, last tile flush."""
    if total <= size:
        return [0]
    stride = size - overlap
    starts = list(range(0, total - size + 1, stride))
    if starts[-1] != total - size:
        starts.append(total - size)
    return starts


def decode_video_chunked(
    vae: Any,
    samples: Any,
    *,
    tile_latent: int = 16,
    overlap_latent: int = 4,
    tokens_per_chunk: int = 1,
    progress: bool = False,
):
    """Decode a video latent in spatial tiles and temporal chunks.

    ``samples`` is either a LATENT dict or the raw tensor ``[B, C, T, H, W]``.
    Returns ``[B*T, H*r, W*r, 3]`` — the same layout ``VAEDecode`` produces, so
    it is a drop-in replacement for that node's output.

    tile_latent / overlap_latent are in *latent* units; for MiniMax H3 the
    spatial ratio is 16, so the default 16/4 means 256px tiles with 64px overlap
    — the largest tile measured to fit comfortably.

    tokens_per_chunk is ``k`` in the model's 5k+2 latent / 17k+5 frame grid.
    k=1 decodes 22 frames at a time. Consecutive chunks share 2 latent tokens,
    whose 5 leading frames are dropped from every chunk after the first.
    """
    import torch

    z = samples["samples"] if isinstance(samples, dict) else samples

    # MiniMax H3 generates video and audio jointly, so the sampler hands back a
    # comfy.nested_tensor.NestedTensor pair: video [B,24,T,H/16,W/16] and audio
    # [B,32,2,T40]. VAEDecode picks the video half with unbind()[0]
    # (nodes.py:331-333) and VAEDecodeAudio takes the audio half; do the same
    # rather than trying to index the pair.
    if getattr(z, "is_nested", False):
        z = z.unbind()[0]

    if z.ndim != 5:
        raise ValueError(f"expected a 5D video latent [B,C,T,H,W], got {tuple(z.shape)}")

    # Loading the VAE leaves large cached blocks in the allocator. Without this
    # the *first* tile OOMs, ComfyUI catches it and retries via the tiled path
    # (comfy/sd.py:1212), which then OOMs for real. Every tile after the first is
    # fine because the loop clears between them — so the symptom is a first-tile
    # failure that looks like the whole approach not working.
    gc.collect()
    torch.cuda.empty_cache()

    ratio = vae.upscale_ratio[1] if isinstance(vae.upscale_ratio, tuple) else 16
    frames_for = (
        vae.upscale_ratio[0]
        if isinstance(vae.upscale_ratio, tuple)
        else (lambda t: t)
    )

    _, _, total_tokens, lat_h, lat_w = z.shape
    out_h, out_w = lat_h * ratio, lat_w * ratio

    # --- temporal segmentation on the 5k+2 grid -----------------------------
    k = max(1, int(tokens_per_chunk))
    usable = (total_tokens - 2) // 5
    if usable < 1:
        segments = [(0, total_tokens, 0)]
    else:
        k = min(k, usable)
        while usable % k:
            k -= 1
        seg_tokens = 5 * k + 2
        segments = [
            (i * 5 * k, seg_tokens, 0 if i == 0 else 5)
            for i in range(usable // k)
        ]

    y_starts = _tile_starts(lat_h, min(tile_latent, lat_h), overlap_latent)
    x_starts = _tile_starts(lat_w, min(tile_latent, lat_w), overlap_latent)
    ty = min(tile_latent, lat_h)
    tx = min(tile_latent, lat_w)

    # Without no_grad every decoded tile keeps its autograd graph alive, and
    # `canvas += tile * w` chains those graphs together — GPU allocation then
    # climbs 7.2 GB per tile until it OOMs. ComfyUI's executor wraps node calls
    # in inference mode; calling nodes directly does not.
    out_parts = []
    for seg_index, (t0, t_len, drop) in enumerate(segments):
        seg_z = z[:, :, t0 : t0 + t_len]
        seg_frames = frames_for(seg_z.shape[2]) - drop

        canvas = None
        weights = None
        for yi in y_starts:
            for xi in x_starts:
                with torch.no_grad():
                    tile = vae.decode(seg_z[:, :, :, yi : yi + ty, xi : xi + tx])
                tile = tile.detach()
                if drop:
                    tile = tile[:, drop:]
                tile = tile.float()

                if canvas is None:
                    canvas = torch.zeros(
                        tile.shape[0], seg_frames, out_h, out_w, tile.shape[-1],
                        dtype=torch.float32,
                    )
                    weights = torch.zeros(1, 1, out_h, out_w, 1, dtype=torch.float32)

                py, px = yi * ratio, xi * ratio
                th, tw = tile.shape[2], tile.shape[3]

                w = torch.ones(1, 1, th, tw, 1, dtype=torch.float32)
                ov = overlap_latent * ratio
                if yi != y_starts[0] and ov:
                    w[:, :, :ov, :, :] *= _ramp(ov, w.device, w.dtype).view(1, 1, -1, 1, 1)
                if yi != y_starts[-1] and ov:
                    w[:, :, -ov:, :, :] *= _ramp(ov, w.device, w.dtype).flip(0).view(1, 1, -1, 1, 1)
                if xi != x_starts[0] and ov:
                    w[:, :, :, :ov, :] *= _ramp(ov, w.device, w.dtype).view(1, 1, 1, -1, 1)
                if xi != x_starts[-1] and ov:
                    w[:, :, :, -ov:, :] *= _ramp(ov, w.device, w.dtype).flip(0).view(1, 1, 1, -1, 1)

                canvas[:, :, py : py + th, px : px + tw, :] += tile * w
                weights[:, :, py : py + th, px : px + tw, :] += w

                del tile, w
                gc.collect()
                torch.cuda.empty_cache()

            if progress:
                print(f"    seg {seg_index + 1}/{len(segments)} row {yi} done")

        canvas /= weights.clamp_min(1e-6)
        out_parts.append(canvas)

    out = torch.cat(out_parts, dim=1) if len(out_parts) > 1 else out_parts[0]
    return out.reshape(-1, *out.shape[-3:])
