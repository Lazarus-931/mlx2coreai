# EmbeddingGemma 2

This recipe converts the shared text, vision/video, and audio paths of an MLX EmbeddingGemma 2 checkpoint into four CoreAI components. The host runtime inserts projected modality tokens into the shared language input, avoiding a separate copy of the language backbone for every modality.

## Requirements

Install an `mlx-vlm` revision containing the `embedding_gemma2` model and processor. The default source is `nativ-community/embeddinggemma-2-bf16`.

## Convert

```bash
python -m recipes.embedding_gemma2 convert \
  nativ-community/embeddinggemma-2-bf16 \
  --output artifacts/recipes/embedding_gemma2_fp32
```

The default contract uses batch size 1, a fixed combined sequence length of 768, 2,520 padded vision patches, and 99 audio feature frames. Enumerate additional sequence lengths as separate bundles when short-input latency matters.

Use the shared weight compressor for smaller bundles:

```bash
python -m recipes.embedding_gemma2 convert \
  nativ-community/embeddinggemma-2-bf16 \
  --weight-format uint8 \
  --output artifacts/recipes/embedding_gemma2_uint8
```

`source` preserves compatible affine quantization already present in an MLX checkpoint. `uint4` and `uint8` quantize eligible FP32 projection weights during export. Quantization changes must be validated for retrieval quality, not only tensor similarity.

## Validate

```bash
python -m recipes.embedding_gemma2 validate \
  --model nativ-community/embeddinggemma-2-bf16 \
  --output artifacts/recipes/embedding_gemma2_validation
```

Validation compares final normalized embeddings with MLX for text, image, audio, and two-frame video inputs. The default minimum cosine is 0.998.

## Run

Preprocess inputs with `EmbeddingGemma2Processor`, save its arrays to NPZ, and run the bundle on CPU or GPU:

```bash
python -m recipes.embedding_gemma2 run \
  artifacts/recipes/embedding_gemma2_fp32 \
  --inputs request.npz \
  --output embedding.npy \
  --compute gpu
```

The NPZ must contain `input_ids` and `attention_mask`. It may additionally contain the matching processor outputs for image, audio, or video. Documents are embedded through extracted text rather than direct binary document ingestion.

## Runtime policy

CoreAI compute selection belongs to the caller. For an application that also runs an MLX generation server, prefer CPU while the server is active and GPU while it is stopped. This avoids direct GPU contention, but continuous CPU embedding can still consume shared memory bandwidth; benchmark the intended indexing rate rather than running an unbounded background loop.

## Limitations

- Shapes are fixed at conversion time.
- Video frames reuse the vision component serially.
- Media decoding and processor transforms remain outside the CoreAI assets.
- Apple Neural Engine placement is not implied by CoreAI conversion and must be measured on the target OS and hardware.
