"""
Command-line interface: python -m metrics METRIC PREDICTION REFERENCE [options]

  lve, fdd, mod   PREDICTION and REFERENCE are FLAME parameters (.npz) or vertices (.npy).
  bas             PREDICTION and REFERENCE are FLAME parameter files.
  pbas            PREDICTION is a FLAME parameter file and REFERENCE the speech (.wav).

With --pairs FILE (two paths per line: prediction, reference) several pairs are
scored. Each result is printed as one JSON line; --output also writes them to a
JSON file. The metrics are computed on the meshes as given, with their global
head rotation; the canonical-orientation protocol of the thesis is implemented
in scripts/evaluation/run_canonical_metrics.py.
"""
import argparse
import json
from pathlib import Path

FORMATS = ["th1kh", "ensemble", "artalk"]


def read_pairs(path):
    pairs = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            prediction, reference = line.split()[:2]
            pairs.append((prediction, reference))
    return pairs


def make_scorer(args):
    if args.metric in ("lve", "fdd", "mod"):
        from metrics import FDDCalculator, LVECalculator, MODCalculator
        from utils.flame_utils import load_mesh_sequence

        calc = {"lve": LVECalculator, "fdd": FDDCalculator, "mod": MODCalculator}[args.metric](
            device=args.device
        )
        compute = getattr(calc, {
            "lve": "calculate_sequence_lve", "fdd": "calculate_fdd", "mod": "calculate_mod"
        }[args.metric])

        def score(prediction, reference):
            pred = load_mesh_sequence(prediction, calc.flame_model, calc.device, args.pred_format)
            ref = load_mesh_sequence(reference, calc.flame_model, calc.device, args.ref_format)
            n = min(pred.shape[0], ref.shape[0])
            return {args.metric: compute(pred[:n], ref[:n])}

        return score

    if args.metric == "bas":
        from metrics import BASCalculator

        calc = BASCalculator(device=args.device, sigma=args.sigma)
        return lambda prediction, reference: calc.calculate_bas_from_files(
            reference, prediction, args.ref_format, args.pred_format
        )

    from metrics import PBASCalculator

    calc = PBASCalculator(device=args.device, sigma=args.sigma)
    return lambda prediction, reference: calc.calculate_pbas_from_files(
        prediction, reference, args.pred_format
    )


def main():
    parser = argparse.ArgumentParser(
        prog="python -m metrics", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("metric", choices=["lve", "fdd", "mod", "bas", "pbas"])
    parser.add_argument("prediction", nargs="?", help="Predicted motion")
    parser.add_argument("reference", nargs="?", help="Ground-truth motion, or speech audio for pbas")
    parser.add_argument("--pairs", help="File with one 'prediction reference' pair per line")
    parser.add_argument("--pred-format", choices=FORMATS, default="ensemble",
                        help="Format of FLAME parameter files of the prediction (default: ensemble)")
    parser.add_argument("--ref-format", choices=FORMATS, default="th1kh",
                        help="Format of FLAME parameter files of the reference (default: th1kh)")
    parser.add_argument("--sigma", type=float, default=3.0, help="Kernel width in frames for bas/pbas")
    parser.add_argument("--device", default=None, help="'cpu' or 'cuda' (default: automatic)")
    parser.add_argument("--output", help="Write the results to this JSON file")
    args = parser.parse_args()

    if args.pairs:
        pairs = read_pairs(args.pairs)
    elif args.prediction and args.reference:
        pairs = [(args.prediction, args.reference)]
    else:
        parser.error("give PREDICTION and REFERENCE, or --pairs FILE")

    score = make_scorer(args)
    results = []
    for prediction, reference in pairs:
        result = {"prediction": prediction, "reference": reference, **score(prediction, reference)}
        print(json.dumps(result))
        results.append(result)

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
