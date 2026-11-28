"""
Fast MiniMax-H3-Image encoding for the online translator producer: the released tiling (256 px tiles, >= 64 px
blend overlap) with all tiles of a batch encoded in one encoder call (static shapes, compilable), then stitched with
the diffusers blend. On a single frame every causal 3D convolution only sees its last temporal kernel slice (the
causal padding is zeros), so the convolutions run as exact 2D convolutions with a third of the work. Matches the
diffusers tiled encode up to floating point error; bf16 autocast is optional.
"""

import types

import torch
import torch.nn as nn
import torch.nn.functional as functional

from stdev.vaes.minimax_h3_image_vae import MiniMaxH3ImageVAE


def single_frame_causal_conv_forward(self: nn.Conv3d, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    args:
        hidden_states: [B, C, 1, H, W]
    returns:
        hidden_states: [B, C_out, 1, H', W'], equal to the causal 3D convolution of a single frame
    """
    features = hidden_states[:, :, 0]
    if self.spatial_padding > 0:
        padding = self.spatial_padding
        features = functional.pad(features, (padding, padding, padding, padding), mode=self.spatial_padding_mode)
    features = functional.conv2d(features, self.weight[:, :, -1], self.bias, stride=self.stride[1:], dilation=self.dilation[1:])
    return features[:, :, None]


def use_single_frame_convolutions(encoder: nn.Module) -> int:
    """
    Replaces the forward of every causal 3D convolution whose causal padding aligns the only frame with the last
    temporal kernel slice. Returns the number of replaced convolutions.
    """
    replaced = 0
    for module in encoder.modules():
        if hasattr(module, "temporal_padding") and isinstance(module, nn.Conv3d):
            if module.temporal_padding != module.kernel_size[0] - 1 or module.dilation[0] != 1:
                raise ValueError(f"unexpected temporal layout {module.kernel_size} {module.temporal_padding}")
            module.forward = types.MethodType(single_frame_causal_conv_forward, module)
            replaced += 1
    return replaced


class BatchedTileH3Encoder:
    """
    Wraps a MiniMaxH3ImageVAE; `encode` returns normalized H3 latents like `MiniMaxH3ImageVAE.encode`.
    """

    def __init__(self, vae: MiniMaxH3ImageVAE, use_bfloat16: bool, compile_encoder: bool, tiles_per_call: int = 32) -> None:
        self.vae = vae
        self.tiles_per_call = tiles_per_call
        self.model = vae.model
        self.use_bfloat16 = use_bfloat16
        if use_bfloat16:
            self.model.encoder.to(torch.bfloat16)
            self.model.quant_conv.to(torch.bfloat16)
        use_single_frame_convolutions(self.model.encoder)
        use_single_frame_convolutions(self.model.quant_conv)
        self.tile_encoder = torch.compile(self._encode_tiles, dynamic=False) if compile_encoder else self._encode_tiles

    def _encode_tiles(self, tiles: torch.Tensor) -> torch.Tensor:
        """
        args:
            tiles: [N, 3, 256, 256], ImageNet normalized pixels
        returns:
            latents: [N, 24, 16, 16], raw (unnormalized) posterior mode
        """
        moments = self.model.quant_conv(self.model.encoder(tiles[:, :, None]))[:, :, -1]
        return moments[:, : moments.shape[1] // 2]

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """
        args:
            pixels: [B, 3, H, W], channel_order: RGB, pixel_values: [0...1], H and W multiples of 16 and >= 256
        returns:
            latents: [B, 24, H / 16, W / 16], normalized H3 latents, dtype: torch.float32
        """
        model = self.model
        image = (pixels - self.vae.pixel_mean) / self.vae.pixel_std
        batch_size, _, height, width = image.shape
        y_indices, y_lengths, y_overlaps = model._split_tiles(height, model.tile_sample_min_height, model.tile_sample_min_overlap_height)
        x_indices, x_lengths, x_overlaps = model._split_tiles(width, model.tile_sample_min_width, model.tile_sample_min_overlap_width)
        tiles = torch.cat(
            [
                image[..., y_position : y_position + y_length, x_position : x_position + x_length]
                for y_position, y_length in zip(y_indices, y_lengths)
                for x_position, x_length in zip(x_indices, x_lengths)
            ]
        )  # [tiles * B, 3, 256, 256], tile major
        dtype = torch.bfloat16 if self.use_bfloat16 else torch.float32
        tile_count = tiles.shape[0]
        padding = -tile_count % self.tiles_per_call
        tiles = torch.cat([tiles, tiles[:1].expand(padding, -1, -1, -1)]).to(dtype)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bfloat16):
            encoded = torch.cat([self.tile_encoder(chunk).float() for chunk in tiles.split(self.tiles_per_call)])[:tile_count]
        encoded_tiles = encoded.split(batch_size)
        columns = len(x_indices)
        rows = [list(encoded_tiles[row * columns : (row + 1) * columns]) for row in range(len(y_indices))]
        ratio = model.spatial_compression_ratio
        latents = model._stitch_tiles(rows, [overlap // ratio for overlap in y_overlaps], [overlap // ratio for overlap in x_overlaps])
        return (latents - self.vae.mean[:, :, 0]) / self.vae.std[:, :, 0]
