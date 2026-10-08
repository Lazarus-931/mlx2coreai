import asyncio

import numpy as np
import pytest

from recipes.embedding_gemma2.runtime import Request, run


class Bundle:
    manifest = {"recipe": "embedding_gemma2"}
    metadata = {
        "sequence_length": 4,
        "vision_tokens": 9,
        "hidden_size": 3,
        "embedding_dim": 2,
        "pad_token_id": 0,
        "image_token_id": 10,
        "video_token_id": 11,
        "audio_token_id": 12,
    }


class Session:
    bundle = Bundle()

    async def run(self, component, inputs, readback):
        if component == "token_embedding":
            return {"inputs_embeds": np.zeros((1, 4, 3), np.float32)}
        if component == "vision_encoder":
            return {
                "vision_tokens": np.arange(3, dtype=np.float32).reshape(1, 1, 3),
                "vision_mask": np.array([[True]]),
            }
        if component == "audio_encoder":
            return {
                "audio_tokens": np.arange(3, dtype=np.float32).reshape(1, 1, 3),
                "audio_mask": np.array([[True]]),
            }
        assert component == "language_encoder"
        assert inputs["inputs_embeds"].shape == (1, 4, 3)
        return {"embedding": np.array([[0.6, 0.8]], np.float32)}


def test_runtime_text_and_media_composition():
    session = Session()

    async def check():
        text = await run(session, Request(np.array([[1, 2]]), np.ones((1, 2), np.int32)))
        np.testing.assert_allclose(text, [[0.6, 0.8]])
        image = await run(
            session,
            Request(
                np.array([[10]]),
                np.ones((1, 1), np.int32),
                pixel_values=np.zeros((1, 1, 3), np.float32),
                image_position_ids=np.zeros((1, 1, 2), np.int32),
            ),
        )
        np.testing.assert_allclose(image, [[0.6, 0.8]])
        audio = await run(
            session,
            Request(
                np.array([[12]]),
                np.ones((1, 1), np.int32),
                input_features=np.zeros((1, 1, 128), np.float32),
                input_features_mask=np.ones((1, 1), np.int32),
            ),
        )
        np.testing.assert_allclose(audio, [[0.6, 0.8]])

    asyncio.run(check())


def test_runtime_rejects_invalid_contracts():
    session = Session()

    async def check():
        with pytest.raises(ValueError, match="matching"):
            await run(session, Request(np.ones((1, 2)), np.ones((1, 3))))
        with pytest.raises(ValueError, match="fixed limit"):
            await run(session, Request(np.ones((1, 5)), np.ones((1, 5))))
        with pytest.raises(ValueError, match="no matching media"):
            await run(session, Request(np.array([[10]]), np.ones((1, 1))))

    asyncio.run(check())
