import mlx.core as mx


def token_embedding(model, input_ids):
    embeddings = model.language_model.embed_tokens(input_ids)
    return embeddings * mx.array(model.config.text_config.hidden_size**0.5, dtype=embeddings.dtype)


def vision_encoder(model, pixel_values, position_ids):
    features = model.get_image_features(pixel_values, position_ids)
    stride = model.config.vision_config.pooling_kernel_size**2
    mask = position_ids[:, ::stride, 0] >= 0
    return features, mask


def audio_encoder(model, input_features, input_features_mask):
    invalid = ~input_features_mask.astype(mx.bool_)
    features, invalid = model.audio_tower(input_features, invalid)
    return model.embed_audio(features), ~invalid


def language_encoder(model, inputs_embeds, attention_mask):
    hidden = model.language_model(inputs_embeds, attention_mask)
    expanded = attention_mask[..., None].astype(mx.float32)
    pooled = mx.sum(hidden * expanded, axis=1) / mx.maximum(mx.sum(expanded, axis=1), 1e-9)
    return pooled / mx.maximum(mx.linalg.norm(pooled, axis=-1, keepdims=True), 1e-9)
