from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx2coreai import ConversionConfig, WeightQuantization, prepare_quantized_linears
from mlx2coreai.recipe import Build, Component
from .adapter import audio_encoder, language_encoder, token_embedding, vision_encoder


DEFAULT_SOURCE = "nativ-community/embeddinggemma-2-bf16"


def load_source(source, *, revision=None):
    try:
        from mlx_vlm.utils import load
    except ImportError as exc:
        raise ImportError("EmbeddingGemma 2 requires mlx-vlm with embedding_gemma2 support.") from exc
    options = {} if revision is None else {"revision": revision}
    model, processor = load(source, strict=True, **options)
    return model, processor


def from_model(
    model,
    source,
    *,
    revision=None,
    sequence_length=768,
    vision_tokens=2520,
    audio_frames=99,
    weight_format="fp32",
):
    if getattr(model, "model_type", None) != "embedding_gemma2":
        raise ValueError("Expected an EmbeddingGemma 2 source model.")
    for name, value in {
        "sequence_length": sequence_length,
        "vision_tokens": vision_tokens,
        "audio_frames": audio_frames,
    }.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if weight_format not in ("fp32", "uint4", "uint8", "source"):
        raise ValueError("weight_format must be fp32, uint4, uint8, or source.")
    model.eval()
    model.set_dtype(mx.float32)
    packed = prepare_quantized_linears(model) if weight_format == "source" else None
    mx.eval(model.parameters())
    quantization = (
        WeightQuantization(bits=int(weight_format[-1]), group_size=32)
        if weight_format in ("uint4", "uint8")
        else None
    )
    text = model.config.text_config
    vision = model.config.vision_config
    metadata = {
        "source": str(Path(source).resolve()) if Path(source).is_dir() else str(source),
        "revision": revision,
        "sequence_length": sequence_length,
        "vision_tokens": vision_tokens,
        "audio_frames": audio_frames,
        "hidden_size": int(text.hidden_size),
        "embedding_dim": int(text.embedding_dim),
        "pad_token_id": int(text.pad_token_id),
        "image_token_id": int(model.config.image_token_id),
        "video_token_id": int(model.config.video_token_id),
        "audio_token_id": int(model.config.audio_token_id),
        "vision_pooling_stride": int(vision.pooling_kernel_size**2),
        "weight_format": weight_format,
        "batch_size": 1,
        "preprocessing": "mlx-vlm EmbeddingGemma2Processor outside the CoreAI assets",
    }
    rng = np.random.default_rng(649)
    config = ConversionConfig(
        optimize=False,
        graph_transform=packed,
        weight_quantization=quantization,
    )
    components = {
        "token_embedding": Component(
            lambda input_ids: token_embedding(model, input_ids),
            {"input_ids": np.zeros((1, sequence_length), np.int32)},
            ("inputs_embeds",),
            config,
        ),
        "vision_encoder": Component(
            lambda pixel_values, position_ids: vision_encoder(model, pixel_values, position_ids),
            {
                "pixel_values": rng.normal(size=(1, vision_tokens, vision.hidden_size)).astype(np.float32),
                "position_ids": np.zeros((1, vision_tokens, 2), np.int32),
            },
            ("vision_tokens", "vision_mask"),
            config,
        ),
        "audio_encoder": Component(
            lambda input_features, input_features_mask: audio_encoder(
                model, input_features, input_features_mask
            ),
            {
                "input_features": rng.normal(size=(1, audio_frames, 128)).astype(np.float32),
                "input_features_mask": np.ones((1, audio_frames), np.int32),
            },
            ("audio_tokens", "audio_mask"),
            config,
        ),
        "language_encoder": Component(
            lambda inputs_embeds, attention_mask: language_encoder(
                model, inputs_embeds, attention_mask
            ),
            {
                "inputs_embeds": rng.normal(size=(1, sequence_length, text.hidden_size)).astype(np.float32),
                "attention_mask": np.ones((1, sequence_length), np.int32),
            },
            ("embedding",),
            config,
        ),
    }
    return Build("embedding_gemma2", components, metadata=metadata, runtime_metadata=metadata)


def build(source=DEFAULT_SOURCE, *, revision=None, **kwargs):
    model, _ = load_source(source, revision=revision)
    return from_model(model, source, revision=revision, **kwargs)
