"""B0 HRM Deliberation Bridge — entry point stub.

Reusable B0 benchmark orchestration is not implemented yet.  Per
spec/spec-benchmark-suite.md §6, shared orchestration, corpus contracts, and
evaluator logic must live in ``ehc_sn.benchmarks``, not in this script.

This stub fails fast so contributors cannot accidentally treat it as a working
evaluator.
"""


def main() -> None:
    raise NotImplementedError(
        "B0 benchmark orchestration is not yet implemented.\n"
        "Reusable evaluator logic must live in ehc_sn.benchmarks, not in this script.\n"
        "See spec/spec-benchmark-suite.md §6 for ownership rules."
    )


if __name__ == "__main__":
    main()
