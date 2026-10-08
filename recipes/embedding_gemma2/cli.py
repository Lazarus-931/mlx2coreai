import argparse
import asyncio
from pathlib import Path

import numpy as np

from mlx2coreai.recipe import Bundle, export
from .build import DEFAULT_SOURCE, build
from .runtime import Request, run


def _request(path):
    values = np.load(path, allow_pickle=False)
    names = Request.__dataclass_fields__
    return Request(**{name: values[name] for name in values.files if name in names})


async def embed(args):
    from coreai.runtime import ComputeUnitKind, SpecializationOptions

    bundle = Bundle.open(args.bundle)
    options = (
        SpecializationOptions.cpu_only()
        if args.compute == "cpu"
        else SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
    )
    async with bundle.session(specialization_options=options) as session:
        embedding = await run(session, _request(args.inputs))
    np.save(args.output, embedding, allow_pickle=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert", help="Build an EmbeddingGemma 2 recipe bundle.")
    convert.add_argument("source", nargs="?", default=DEFAULT_SOURCE)
    convert.add_argument("--revision")
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--sequence-length", type=int, default=768)
    convert.add_argument("--weight-format", choices=("fp32", "uint4", "uint8", "source"), default="fp32")
    run_parser = commands.add_parser("run", help="Embed preprocessed arrays from an NPZ file.")
    run_parser.add_argument("bundle", type=Path)
    run_parser.add_argument("--inputs", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--compute", choices=("cpu", "gpu"), default="gpu")
    validate = commands.add_parser("validate", help="Build and compare all modalities with MLX.")
    validate.add_argument("--model", default=DEFAULT_SOURCE)
    validate.add_argument("--revision")
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--sequence-length", type=int, default=768)
    validate.add_argument("--weight-format", choices=("fp32", "uint4", "uint8", "source"), default="fp32")
    validate.add_argument("--minimum-cosine", type=float, default=0.998)
    args = parser.parse_args(argv)
    if args.command == "convert":
        export(
            build(
                args.source,
                revision=args.revision,
                sequence_length=args.sequence_length,
                weight_format=args.weight_format,
            ),
            args.output,
        )
    elif args.command == "run":
        asyncio.run(embed(args))
    else:
        from .validation import validate as validate_model

        asyncio.run(validate_model(args))
