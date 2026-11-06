import argparse
import json
from pathlib import Path
from .counterfactual import collect
from .model import fit_models
from .evaluation import evaluate, METHODS, EXTRA_METHODS
from .report import summarize, plot, plot_envelopes


def main():
    parser = argparse.ArgumentParser(description="Delayed Advice Audit state diagnostic research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", required=True)
    c.add_argument("--backend", choices=["toy", "fetch"], default="toy")
    c.add_argument("--episodes", type=int, default=40)
    c.add_argument("--seed", type=int, default=100)
    c.add_argument("--horizon", type=int, default=120)
    c.add_argument("--delays", type=int, nargs="+", default=[0, 2, 6, 12])
    c.add_argument("--chunk", type=int, default=12)
    c.add_argument("--rollout-horizon", type=int, default=24)
    c.add_argument("--shift-scale", type=float, default=.15)
    c.add_argument("--interval", type=int, default=10)
    c.add_argument("--audit-deadline", action="store_true", help="Also measure single-intervention gain through the original episode deadline")
    c.add_argument("--request-period", type=int, help="Match fixed closed-loop request slots, including valid near-deadline arrivals")
    c.add_argument("--include-clock", action="store_true", help="Append public remaining-budget and original-horizon features")
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--cost", type=float, default=.01)
    t.add_argument("--alpha", type=float, default=.2)
    t.add_argument("--split-manifest", help="Frozen four-way episode split JSON")
    t.add_argument("--target", choices=["gain", "gain_deadline"], default="gain")
    t.add_argument("--feature-set", choices=["full", "with_clock", "no_clock", "no_age", "no_candidate", "current_only"], default="full")
    t.add_argument("--linear", action="store_true", help="Linear ridge arrival baseline, no random features/bootstrap")
    e = sub.add_parser("evaluate")
    e.add_argument("--out", required=True)
    e.add_argument("--model-dir")
    e.add_argument("--backend", choices=["toy", "fetch"], default="toy")
    e.add_argument("--methods", nargs="+", choices=METHODS + EXTRA_METHODS, default=list(METHODS))
    e.add_argument("--episodes", type=int, default=20)
    e.add_argument("--seed", type=int, default=10000)
    e.add_argument("--delays", nargs="+", type=int, default=[0, 6, 12, 24])
    e.add_argument("--horizon", type=int, default=120)
    e.add_argument("--chunk", type=int, default=12)
    e.add_argument("--cost", type=float, default=.01)
    e.add_argument("--shift-scale", type=float, default=.15)
    e.add_argument("--budget", type=int, default=12)
    e.add_argument("--period", type=int, default=12)
    e.add_argument("--risk-threshold", type=float, default=.10)
    e.add_argument("--random-probability", type=float, default=.25)
    e.add_argument("--ttl", type=int, default=6)
    e.add_argument("--audit", action="store_true")
    e.add_argument("--audit-deadline", action="store_true")
    e.add_argument("--audit-epsilon", type=float, default=1e-6)
    e.add_argument("--label-horizon", type=int, default=24)
    e.add_argument("--fixed-schedule", action="store_true")
    e.add_argument("--drift-threshold", type=float, default=.05)
    s = sub.add_parser("report")
    s.add_argument("--dir", required=True)
    s.add_argument("--plot", action="store_true")
    v = sub.add_parser("curves")
    v.add_argument("--data", required=True)
    v.add_argument("--out", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "collect":
        args["path"] = args.pop("out")
        result = collect(**args)
    elif command == "train":
        args["dataset"], args["outdir"] = args.pop("data"), args.pop("out")
        result = fit_models(**args)
    elif command == "evaluate":
        args["outdir"] = args.pop("out")
        rows = evaluate(**args)
        result = {"episodes_written": len(rows), "directory": args["outdir"]}
    elif command == "curves":
        plot_envelopes(args["data"], args["out"])
        result = {"saved": args["out"]}
    else:
        result = summarize(args["dir"])
        if args["plot"]:
            plot(args["dir"])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
