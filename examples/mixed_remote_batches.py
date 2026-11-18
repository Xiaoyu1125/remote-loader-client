"""Mix locally grouped server-delivered VLA samples with complete remote VQA v2 batches.

Each output selects VQA with probability 20% or a local VLA group with 80%.
The proportions and --batches count output groups, not samples or HC frames.
A full VLA group consumes --vla-batch-size one-sample frames; one VQA output
consumes one complete-batch frame. SDK batches_* metrics count HC frames instead.
VLA samples retain their server-defined schema and requested uint8 or processed
float32 pixels. This example does not run a training preprocessor; VQA retains
its complete six-field batch. Tensors from different sources are never
concatenated. The mix seed controls group type selection only, and finite runs
need not have an exact 1:4 ratio. Use the TUI's Start Training flow to register both sources and wait for
their nodes before running this example.

Uses the existing client configuration and stored token without changing them.
Datasets are selected by scheduler registration, not by this example or its groups.
JSON keys batches/counts/ratios refer to output groups; batch_sizes records local
VLA grouping size and remote VQA batch size, respectively.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from remote_loader import BatchSpec, RemoteDataLoader, VQABatchSpec
from remote_loader.auth.token_store import TokenStore
from remote_loader.client_config import ClientConfig
from remote_loader.exceptions import ConfigurationError, RemoteLoaderError


def mix_batches(
    vla: Iterator[list[dict[str, Any]]],
    vqa: Iterator[dict[str, Any]],
    *,
    batches: int,
    seed: int,
) -> Iterator[tuple[str, list[dict[str, Any]] | dict[str, Any]]]:
    """Consume one chosen output group; stream failures never change the weights."""
    rng = random.Random(seed)
    for completed in range(batches):
        source = "vqa" if rng.random() < 0.2 else "vla"
        try:
            batch = next(vqa if source == "vqa" else vla)
        except StopIteration as exc:
            raise RuntimeError(f"{source} stream ended after {completed} mixed output groups") from exc
        yield source, batch


def summarize(item: Any) -> Any:
    """Summarize samples/groups/batches without pixels or literal JSON values."""
    if hasattr(item, "shape") and hasattr(item, "dtype"):
        return {"shape": list(item.shape), "dtype": str(item.dtype)}
    if isinstance(item, dict):
        return {name: summarize(value) for name, value in item.items()}
    if isinstance(item, list):
        return {"type": "list", "length": len(item), "first": summarize(item[0]) if item else None}
    return {"type": type(item).__name__}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vla-spec",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "batch.yaml",
        help="single-sample VLA request YAML; no remote batch size",
    )
    parser.add_argument(
        "--vla-batch-size",
        type=int,
        required=True,
        help="local canonical samples per VLA group; never sent to the service",
    )
    parser.add_argument(
        "--vqa-batch-size",
        type=int,
        default=32,
        help="VQA output batch size, independent of scheduler row-grant size",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=100,
        help="number of mixed output groups, not samples or total HC frames",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for local output group type selection only"
    )
    args = parser.parse_args()
    if args.batches <= 0 or args.vqa_batch_size <= 0 or args.vla_batch_size <= 0:
        parser.error("--batches, --vla-batch-size and --vqa-batch-size must be positive")

    config = ClientConfig.load()
    if config.scheduler is None or config.training.training_id is None:
        raise ConfigurationError(
            "configure a scheduler and use Start Training to register both nodes and vqa_nodes"
        )
    vla_spec = BatchSpec.from_yaml(args.vla_spec)
    if isinstance(vla_spec, VQABatchSpec):
        raise ConfigurationError("--vla-spec must describe raw/VLA canonical samples, not VQA batches")
    token = TokenStore().load()
    vqa_spec = VQABatchSpec(batch={"size": args.vqa_batch_size})
    print(
        json.dumps(
            {
                "event": "start",
                "batches": args.batches,
                "seed": args.seed,
                "probabilities": {"vqa": 0.2, "vla": 0.8},
                "batch_sizes": {"vqa": vqa_spec.batch.size, "vla": args.vla_batch_size},
                "training_id": config.training.training_id,
            }
        ),
        flush=True,
    )

    counts = {"vqa": 0, "vla": 0}
    # These two streams must not overwrite the CLI's shared single-transfer status.
    with (
        RemoteDataLoader(vla_spec, token=token, prefetch_batches=1, _track_transfer_state=False) as vla,
        RemoteDataLoader(vqa_spec, token=token, prefetch_batches=1, _track_transfer_state=False) as vqa,
    ):
        started = time.perf_counter()
        read_started = started
        # Group one-sample HC frames locally; mixing probabilities count output groups.
        # This inspection path preserves server-delivered samples, including any
        # configured image processing. A trainer supplies remaining per-sample
        # preprocessing and native collation without repeating server image steps.
        # Raw frames are consumed before preprocess/collate; failures do not roll back.
        vla_groups = vla.to_torch_dataloader(
            batch_size=args.vla_batch_size, collate_fn=lambda samples: samples
        )
        for step, (source, batch) in enumerate(
            mix_batches(iter(vla_groups), vqa, batches=args.batches, seed=args.seed), start=1
        ):
            read_seconds = time.perf_counter() - read_started
            counts[source] += 1
            print(
                json.dumps(
                    {
                        "step": step,
                        "source": source,
                        "read_seconds": round(read_seconds, 6),
                        "counts": counts,
                        "ratios": {name: count / step for name, count in counts.items()},
                        "fields": summarize(batch),
                    }
                ),
                flush=True,
            )
            read_started = time.perf_counter()
        elapsed = time.perf_counter() - started
    print(
        json.dumps({"event": "complete", "counts": counts, "consume_seconds": round(elapsed, 6)}),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (RemoteLoaderError, RuntimeError, ValueError) as exc:
        print(json.dumps({"event": "error", "error": str(exc)}), file=sys.stderr, flush=True)
        raise SystemExit(1) from None
