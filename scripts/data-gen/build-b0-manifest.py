"""Generate the canonical B0 hard-subset manifest for one processed MazeHard corpus."""

from __future__ import annotations

from pathlib import Path

from typer import Option, Typer, echo

from ehc_sn.data.benchmarks.mazehard_b0 import b0_hard_subset_path, build_b0_hard_subset_manifest, write_mazehard_subset_manifest

app = Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    dataset_root: Path = Option(Path("data/processed/maze-30x30-hard-1k"), "--dataset-root", help="Processed MazeHard dataset root."),
    split: str = Option("test", "--split", help="Dataset split used to derive the benchmark hard subset."),
    quantile: float = Option(0.9, "--quantile", min=0.0, max=1.0, help="Inclusive difficulty quantile threshold."),
    output_path: Path | None = Option(None, "--output-path", help="Optional explicit output path for the benchmark manifest."),
) -> None:
    """Write the deterministic B0 hard-subset manifest for the provided dataset root."""
    manifest = build_b0_hard_subset_manifest(dataset_root, split=split, quantile=quantile)
    resolved_output_path = output_path if output_path is not None else b0_hard_subset_path(dataset_root)
    write_mazehard_subset_manifest(manifest, resolved_output_path)
    echo(f"Wrote B0 hard-subset manifest with {manifest.n_selected} entries to {resolved_output_path}")


if __name__ == "__main__":
    app()
