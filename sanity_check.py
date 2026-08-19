"""Framework self-test (anti self-deception):
1) data with an injected cross-sectional reversal effect -> the matching factor
   must pass the core statistical gates (G4/G5/G10/G11);
2) the same factor on pure-noise data -> overall verdict must be REJECT.

If either assertion fails, do NOT trust any result from this pipeline.
"""
import json

import yaml

import expression as ex
import gates
from synthetic import make_synthetic


def show(tag, rep):
    print(f"\n### {tag}")
    print(f"    verdict = {'PASS' if rep['verdict'] else 'REJECT'}")
    for r in rep["results"]:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"  [{mark}] {r['gate']:<22} {json.dumps(r['detail'], default=str)[:120]}")


if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    tree = ex.parse("reverse(ts_delta(close, 5))")

    fields_good = make_synthetic(n_symbols=30, n_days=750, effect=0.12, seed=1)
    rep_good = gates.run_gates(tree, fields_good, cfg, seed=0)
    show("factor matching an injected effect (expect core gates PASS)", rep_good)

    fields_noise = make_synthetic(n_symbols=30, n_days=750, effect=0.0, seed=2)
    rep_noise = gates.run_gates(tree, fields_noise, cfg, seed=0)
    show("same factor on pure noise (expect REJECT)", rep_noise)

    core = {"G4_is_ic", "G5_placebo", "G10_oos", "G11_bootstrap"}
    good_core = all(r["passed"] for r in rep_good["results"] if r["gate"] in core)
    assert good_core, "FRAMEWORK BROKEN: failed to detect a real injected effect"
    assert not rep_noise["verdict"], "FRAMEWORK BROKEN: failed to reject noise"
    print("\nSANITY CHECK OK: detects injected effect, rejects noise.")
