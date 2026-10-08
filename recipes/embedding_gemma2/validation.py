import json

import mlx.core as mx
import numpy as np

from mlx2coreai.recipe import export
from .build import from_model, load_source
from .runtime import Request, run


def _request(inputs):
    names = Request.__dataclass_fields__
    values = {
        name: np.asarray(value)
        for name, value in inputs.items()
        if name in names
    }
    return Request(**values)


def _cases(processor):
    image = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    audio = np.sin(np.linspace(0, 440 * 2 * np.pi, 16000, dtype=np.float32))
    video = np.stack([image, np.flipud(image).copy()])
    arguments = {
        "text": {"text": ["A red fox running through a green field."]},
        "image": {"text": ["<|image|>"], "images": [image]},
        "audio": {"text": ["<|audio|>"], "audio": [audio]},
        "video": {"text": ["<|video|>"], "videos": [video]},
    }
    for name, values in arguments.items():
        yield name, processor(return_tensors="mlx", padding=True, **values)


async def validate(args):
    from coreai.runtime import ComputeUnitKind, SpecializationOptions

    model, processor = load_source(args.model, revision=args.revision)
    plan = from_model(
        model,
        args.model,
        revision=args.revision,
        sequence_length=args.sequence_length,
        weight_format=args.weight_format,
    )
    bundle = export(plan, args.output)
    options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
    report = {
        "model": args.model,
        "revision": args.revision,
        "sequence_length": args.sequence_length,
        "weight_format": args.weight_format,
        "results": [],
    }
    async with bundle.session(specialization_options=options) as session:
        for name, inputs in _cases(processor):
            expected = np.asarray(model(**inputs).text_embeds)
            actual = await run(session, _request(inputs))
            if expected.shape != actual.shape or not np.isfinite(actual).all():
                raise AssertionError((name, expected.shape, actual.shape))
            cosine = float(np.sum(expected * actual) / (np.linalg.norm(expected) * np.linalg.norm(actual)))
            maximum = float(np.max(np.abs(expected - actual)))
            if cosine < args.minimum_cosine:
                raise AssertionError(f"{name} cosine {cosine} is below {args.minimum_cosine}")
            row = {"case": name, "cosine": cosine, "max_abs_error": maximum}
            report["results"].append(row)
            print(json.dumps(row), flush=True)
            mx.clear_cache()
    destination = args.output / "validation.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(report['results'])} modalities; saved {destination}", flush=True)
    return report
