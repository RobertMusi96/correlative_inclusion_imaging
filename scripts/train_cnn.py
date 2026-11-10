"""Train and cross-validate the CNN on a dataset (needs: uv sync --extra cnn).

Input: dataset_vN/ of export_dataset.py (dataset.csv, crops/, masks/). Only the
training / validation particles are used; the test set is never loaded.

Per fold k (default all): train on the other folds, validate on fold k. Same
settings for all folds; the score is the balanced accuracy after the last epoch.

Output: dataset_vN/runs/<name>/ (config.json, history.csv, predictions.csv,
metrics.json, curves.png, confusion.png, model_fold<k>.pt), see cii.cnn.train.

Input channels (--channels): rgb (bright field), mask, and lighting modes added
with scripts/add_lighting.py (e.g. df, ml). With --require df,ml only particles
that have these crops are used, also when they are not among the channels.

Examples:
    python scripts/train_cnn.py data/datasets/dataset_v1 --name resnet18_rgb
    python scripts/train_cnn.py data/datasets/dataset_v1 --name quick --folds 0 --epochs 3
    python scripts/train_cnn.py data/datasets/dataset_v1 --name rgb_same --require df,ml
    python scripts/train_cnn.py data/datasets/dataset_v1 --name rgb_df --channels rgb,mask,df --require df,ml
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main():
    """Command line: runs the cross-validation with the given settings into a run folder."""
    from cii.cnn.train import TrainParams, cross_validate

    d = TrainParams()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, help="dataset folder (dataset.csv, crops/, masks/)")
    ap.add_argument("--name", required=True, help="name of the run folder under <dataset>/runs/")
    ap.add_argument("--folds", help="folds to run, e.g. 0 or 0,1 (default all)")
    ap.add_argument("--epochs", type=int, default=d.epochs, help=f"default {d.epochs}")
    ap.add_argument("--batch-size", type=int, default=d.batch_size, help=f"default {d.batch_size}")
    ap.add_argument("--lr", type=float, default=d.lr, help=f"learning rate at the start (default {d.lr:g})")
    ap.add_argument("--weight-decay", type=float, default=d.weight_decay, help=f"default {d.weight_decay:g}")
    ap.add_argument("--brightness", type=float, default=d.brightness,
                    help=f"augmentation: brightness factor 1 ± b (default {d.brightness:g})")
    ap.add_argument("--no-pretrained", action="store_true", help="random start instead of ImageNet weights")
    ap.add_argument("--no-tta", action="store_true", help="validate without the 8 orientations")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--threads", type=int, default=d.threads, help="CPU threads (default: torch default)")
    ap.add_argument("--channels", default=",".join(d.channels),
                    help="input channels: rgb, mask and added lighting modes, e.g. rgb,mask,df (default rgb,mask)")
    ap.add_argument("--require", default="", help="lighting modes whose crops every particle must have, e.g. df,ml "
                                                  "(to compare runs with and without them on the same particles)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing run folder")
    args = ap.parse_args()

    csv = args.dataset / "dataset.csv"
    if not csv.exists():
        raise SystemExit(f"{csv} not found")

    out = args.dataset / "runs" / args.name
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists - give another --name or --overwrite")
        shutil.rmtree(out)

    p = TrainParams(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
                    brightness=args.brightness, pretrained=not args.no_pretrained, tta=not args.no_tta,
                    seed=args.seed, threads=args.threads, channels=tuple(args.channels.split(",")),
                    require=tuple(m for m in args.require.split(",") if m))
    folds = [int(k) for k in args.folds.split(",")] if args.folds else None

    ds = pd.read_csv(csv, low_memory=False)
    cross_validate(ds, args.dataset, out, p, folds)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
