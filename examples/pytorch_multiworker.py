"""Read a bounded number of batches with a user-owned native PyTorch DataLoader.

Run against an existing allocation:
    python examples/pytorch_multiworker.py batch.yaml --num-workers 8 --batch-size 8

This example consumes data only; it does not construct a model or run training.
Model-specific CPU preprocessing can be supplied to RemoteIterableDataset as a
module-level callable. Customize collate_fn for ragged fields or JSON nulls.
"""

from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from remote_loader import RemoteDataLoader
from remote_loader.pytorch import RemoteIterableDataset, close_torch_dataloader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_yaml")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--pin-memory", action="store_true")
    args = parser.parse_args()
    if args.batches <= 0:
        parser.error("--batches must be positive")

    remote = RemoteDataLoader.from_yaml(args.batch_yaml)
    if remote.spec.is_vqa:
        parser.error("VQA already supplies complete batches; this example batches canonical samples")
    worker_options = (
        {
            "multiprocessing_context": "spawn",
            "prefetch_factor": 2,
            "persistent_workers": True,
        }
        if args.num_workers > 0
        else {}
    )
    with remote:
        dataset = RemoteIterableDataset(remote)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            **worker_options,
        )
        iterator = None
        try:
            iterator = iter(dataloader)
            for index, batch in enumerate(iterator, start=1):
                print(f"batch={index} fields={list(batch)}", flush=True)
                if index >= args.batches:
                    break
        finally:
            # Release DataLoader resources before leaving the loader context.
            close_torch_dataloader(dataloader, iterator=iterator)


if __name__ == "__main__":
    main()
