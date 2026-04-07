"""Generate the canonical M0 replay manifest for one processed dungeon corpus."""

from __future__ import annotations

from pathlib import Path

from typer import Option, Typer, echo

from ehc_sn.data.benchmarks.dungeon_m0 import build_m0_replay_manifest, m0_trajectory_manifest_path, write_m0_replay_manifest

app = Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    dataset_root: Path = Option(Path("data/processed/dungeons"), "--dataset-root", help="Processed dungeon dataset root."),
    output_path: Path | None = Option(None, "--output-path", help="Optional explicit output path for the replay manifest."),
) -> None:
    """Write the deterministic M0 replay manifest for the provided dataset root."""
    manifest = build_m0_replay_manifest(dataset_root)
    resolved_output_path = output_path if output_path is not None else m0_trajectory_manifest_path(dataset_root)
    write_m0_replay_manifest(manifest, resolved_output_path)
    echo(f"Wrote {len(manifest.entries)} M0 replay entries to {resolved_output_path}")


if __name__ == "__main__":
    app()
