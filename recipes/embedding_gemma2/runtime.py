from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Request:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    pixel_values: np.ndarray | None = None
    image_position_ids: np.ndarray | None = None
    input_features: np.ndarray | None = None
    input_features_mask: np.ndarray | None = None
    pixel_values_videos: np.ndarray | None = None
    video_position_ids: np.ndarray | None = None


def _padded(values, shape, fill, dtype):
    result = np.full(shape, fill, dtype=dtype)
    slices = tuple(slice(0, size) for size in values.shape)
    result[slices] = values.astype(dtype)
    return result


async def _vision(session, values, positions, metadata):
    count = metadata["vision_tokens"]
    values = _padded(values, (1, count, values.shape[-1]), 0, np.float32)
    positions = _padded(positions, (1, count, 2), -1, np.int32)
    output = await session.run(
        "vision_encoder",
        {"pixel_values": values, "position_ids": positions},
        readback=True,
    )
    return output["vision_tokens"][output["vision_mask"].astype(bool)]


async def run(session, request: Request):
    if session.bundle.manifest["recipe"] != "embedding_gemma2":
        raise ValueError("Expected an EmbeddingGemma 2 recipe bundle.")
    metadata = session.bundle.metadata
    ids = np.asarray(request.input_ids)
    attention = np.asarray(request.attention_mask)
    if ids.ndim != 2 or ids.shape[0] != 1 or attention.shape != ids.shape:
        raise ValueError("input_ids and attention_mask must have matching [1, sequence] shapes.")
    length = ids.shape[1]
    capacity = metadata["sequence_length"]
    if length > capacity:
        raise ValueError(f"Combined sequence length {length} exceeds the fixed limit of {capacity}.")
    padded_ids = _padded(ids, (1, capacity), metadata["pad_token_id"], np.int32)
    padded_attention = _padded(attention, (1, capacity), 0, np.int32)
    modality = np.isin(
        padded_ids,
        [metadata["image_token_id"], metadata["video_token_id"], metadata["audio_token_id"]],
    )
    embedding_ids = np.where(modality, metadata["pad_token_id"], padded_ids).astype(np.int32)
    output = await session.run("token_embedding", {"input_ids": embedding_ids}, readback=True)
    embeddings = output["inputs_embeds"]
    features = []
    if request.pixel_values is not None:
        if request.image_position_ids is None:
            raise ValueError("image_position_ids are required with pixel_values.")
        features.append(
            await _vision(
                session,
                np.asarray(request.pixel_values),
                np.asarray(request.image_position_ids),
                metadata,
            )
        )
    if request.input_features is not None:
        if request.input_features_mask is None:
            raise ValueError("input_features_mask is required with input_features.")
        audio = await session.run(
            "audio_encoder",
            {
                "input_features": np.asarray(request.input_features, dtype=np.float32),
                "input_features_mask": np.asarray(request.input_features_mask, dtype=np.int32),
            },
            readback=True,
        )
        features.append(audio["audio_tokens"][audio["audio_mask"].astype(bool)])
    if request.pixel_values_videos is not None:
        if request.video_position_ids is None:
            raise ValueError("video_position_ids are required with pixel_values_videos.")
        for values, positions in zip(
            np.asarray(request.pixel_values_videos),
            np.asarray(request.video_position_ids),
            strict=True,
        ):
            features.append(await _vision(session, values[None], positions[None], metadata))
    if features:
        media = np.concatenate(features)
        if media.shape[0] != int(modality.sum()):
            raise ValueError(
                f"Media token count {int(modality.sum())} does not match feature count {media.shape[0]}."
            )
        embeddings[modality] = media.astype(embeddings.dtype)
    elif modality.any():
        raise ValueError("The input contains media placeholders but no matching media features.")
    output = await session.run(
        "language_encoder",
        {"inputs_embeds": embeddings, "attention_mask": padded_attention},
        readback=True,
    )
    embedding = output["embedding"]
    if embedding.shape != (1, metadata["embedding_dim"]) or not np.isfinite(embedding).all():
        raise RuntimeError("EmbeddingGemma 2 produced an invalid embedding.")
    return embedding
