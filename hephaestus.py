"""
Hephaestus CLI — Entry Point
Usage:
    python hephaestus.py analyze
    python hephaestus.py optimize --target cooldown_ms
    python hephaestus.py tensorize --input tick_data.csv
"""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT.parent))


def cmd_analyze(args):
    """Run attribution analysis"""
    from modules.attribution import AttributionAnalyzer
    import pandas as pd
    import numpy as np

    analyzer = AttributionAnalyzer()

    if args.source:
        source = Path(args.source)
        if source.exists():
            df = pd.read_csv(source)
            print(f"Loaded {len(df):,} rows from {source}")
        else:
            print(f"Source not found: {source}, using demo data")
            df = None
    else:
        df = None

    if df is None:
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-04-01", periods=n, freq="1min"),
            "price": 50000 + np.cumsum(np.random.randn(n) * 10),
            "side": np.random.choice([-1, 1], size=n),
            "size": np.random.exponential(0.1, size=n),
            "is_filled": np.random.choice([True, False], size=n, p=[0.85, 0.15]),
            "returns": np.random.randn(n) * 0.0001,
        })
        print(f"Using {n} rows of demo data")

    metrics, diagnostic = analyzer.analyze(df)
    report = analyzer.generate_report(metrics, diagnostic)

    # Windows terminal can't print emoji/unicode
    safe_report = report.encode('ascii', errors='replace').decode('ascii')
    print(safe_report)

    out = PROJECT_ROOT / "reports" / "attribution_report.md"
    analyzer.generate_report(metrics, diagnostic, output_path=str(out))
    print(f"Report saved: {out.name}")

    return 0


def cmd_optimize(args):
    """Run Bayesian optimization"""
    from modules.crucible import BayesianOptimizer
    import random

    def objective(params):
        return random.random()  # placeholder — wire to real backtest

    search_space = {}
    if args.target:
        for t in args.target.split(","):
            t = t.strip()
            search_space[t] = {
                "type": "int" if t == "cooldown_ms" else "float",
                "low": 100 if t == "cooldown_ms" else 0.0,
                "high": 2000 if t == "cooldown_ms" else 1.0,
            }
    else:
        search_space = None  # use defaults

    opt = BayesianOptimizer(objective, search_space, n_trials=args.n_trials)
    result = opt.optimize()
    print(opt.generate_report())
    print(f"\nBest: {result['best_params']} = {result['best_score']:.4f}")
    return 0


def cmd_tensorize(args):
    """Convert tick CSV to 3D tensor"""
    from modules.forge.tensor_stream import TensorStream, TensorStreamConfig
    import pandas as pd
    import numpy as np

    if not args.input:
        print("Error: --input required")
        print("Example: python hephaestus.py tensorize --input tick_data.csv --seq-len 60")
        return 1

    src = Path(args.input)
    if not src.exists():
        print(f"File not found: {src}")
        print(f"Creating demo data for demo...")
        np.random.seed(42)
        n = 2000
        demo = pd.DataFrame({
            "timestamp": pd.date_range("2026-04-01", periods=n, freq="1s"),
            "mid_price": 50000 + np.cumsum(np.random.randn(n) * 2),
            "bid_price": 50000 + np.cumsum(np.random.randn(n) * 2) - 1,
            "ask_price": 50000 + np.cumsum(np.random.randn(n) * 2) + 1,
            "bid_size_1": np.random.exponential(1, size=n),
            "ask_size_1": np.random.exponential(1, size=n),
            "size": np.random.exponential(0.1, size=n),
            "side": np.random.choice([-1, 1], size=n),
            "returns": np.random.randn(n) * 0.0001,
        })
        demo_path = PROJECT_ROOT / "data" / "demo_tick_data.csv"
        demo_path.parent.mkdir(exist_ok=True)
        demo.to_csv(demo_path, index=False)
        src = demo_path
        print(f"Demo data saved to {src}")

    config = TensorStreamConfig(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        feature_dim=args.feature_dim,
    )
    stream = TensorStream(str(src), config)
    stream.load()
    X, y = stream.to_tensors(use_torch=False)

    print(f"Tensor shape:  X={X.shape}  y={y.shape}")
    print(f"Data info: {stream.info}")
    print(f"First window features (first 5 dims):\n{X[0, 0, :5]}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Hephaestus - Quantitative Research Forge",
        prog="python hephaestus.py",
    )
    subparsers = parser.add_subparsers(dest="command")

    p_analyze = subparsers.add_parser("analyze", help="Run attribution analysis")
    p_analyze.add_argument("--source", help="Path to backtest_results.csv")

    p_optimize = subparsers.add_parser("optimize", help="Run Bayesian optimization")
    p_optimize.add_argument("--target", help="Params to optimize, comma-separated")
    p_optimize.add_argument("--n-trials", type=int, default=50, help="Number of trials")

    p_tensorize = subparsers.add_parser("tensorize", help="Convert tick data to tensor")
    p_tensorize.add_argument("--input", required=True, help="Input CSV file")
    p_tensorize.add_argument("--seq-len", type=int, default=60, help="Sequence length")
    p_tensorize.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p_tensorize.add_argument("--feature-dim", type=int, default=24, help="Feature dimension")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print()
        print("Quick start:")
        print("  python hephaestus.py analyze")
        print("  python hephaestus.py optimize --target cooldown_ms --n-trials 10")
        print("  python hephaestus.py tensorize --input data/tick_data.csv")
        return 1

    cmds = {
        "analyze": cmd_analyze,
        "optimize": cmd_optimize,
        "tensorize": cmd_tensorize,
    }
    return cmds[args.command](args)


if __name__ == "__main__":
    sys.exit(main())