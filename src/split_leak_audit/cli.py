import json
import sys
from pathlib import Path

import click
import pandas as pd

from .core import audit, summarize


@click.command()
@click.argument("train", type=click.Path(exists=True, dir_okay=False))
@click.argument("test", type=click.Path(exists=True, dir_okay=False))
@click.option("--id-col", default="patient_id", show_default=True)
@click.option("--label-col", default=None, help="Excluded from the feature comparison.")
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False), default=None,
              help="JSON with {\"fit_on\": \"train\" | ...} describing preprocessing fit.")
@click.option("--near-threshold", default=0.05, show_default=True)
@click.option("--out", type=click.Path(dir_okay=False), default=None)
def main(train, test, id_col, label_col, manifest, near_threshold, out):
    """Audit a train/test split for patient overlap and leakage. Exits 1 on any error."""
    train_df = pd.read_csv(train)
    test_df = pd.read_csv(test)
    for name, df in [("train", train_df), ("test", test_df)]:
        if id_col not in df.columns:
            raise click.BadParameter(f"{id_col!r} not in {name} file")
    manifest_data = json.loads(Path(manifest).read_text()) if manifest else None
    report = summarize(audit(train_df, test_df, id_col, label_col, manifest_data, near_threshold))
    text = json.dumps(report, indent=2)
    if out:
        Path(out).write_text(text)
    click.echo(text)
    if report["n_errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
