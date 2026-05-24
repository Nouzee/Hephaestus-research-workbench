"""
L2 Data Loader — 加载 OKX 400 档 L2 订单簿 + 逐笔成交
输入: tar.gz (orderbook JSONL) + zip (trades CSV)
输出: 统一的 events Parquet 文件
"""
from __future__ import annotations

import json
import tarfile
import zipfile
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl


class L2DataLoader:
    """
    OKX BTC-USDT-SWAP L2 数据加载器

    数据格式:
    - Orderbook: tar.gz 内含 .data 文件，每行 JSON
        {"ts": "1767225600004", "action": "snapshot"|"update",
         "asks": "[['price','size','n_orders'],...]",
         "bids": "[['price','size','n_orders'],...]"}
    - Trades: zip 内含单个 CSV
        instrument_name, trade_id, side, price, size, created_time
    """

    def __init__(self, data_root: str = r"D:\btc\raw data"):
        self.data_root = Path(data_root)
        self.order_dir = self.data_root / "order" / "okx ob btc 2026-01" / "okx ob btc 2026-01"
        self.trade_dir = self.data_root / "trades" / "okx trades btc 2026-01" / "okx trades btc 2026-01"
        self.cache_dir = Path(__file__).parent.parent.parent / "data"

    def load_day(self, date_str: str, use_cache: bool = True) -> pl.DataFrame:
        """加载单日数据，返回合并后的 events DataFrame"""
        cache_file = self.cache_dir / f"events_{date_str}.parquet"
        if use_cache and cache_file.exists():
            return pl.read_parquet(str(cache_file))

        print(f"  [L2] Loading orderbook snapshots for {date_str}...")
        book = self._load_orderbook_snapshots(date_str)

        print(f"  [L2] Loading trades for {date_str}...")
        trades = self._load_trades(date_str)

        print(f"  [L2] Merging trades + orderbook (asof join)...")
        events = self._merge_trades_book(trades, book)

        print(f"  [L2] Computing features...")
        events = self._add_features(events)

        self.cache_dir.mkdir(exist_ok=True)
        events.write_parquet(str(cache_file))
        print(f"  [L2] Cached to {cache_file.name}")

        return events

    def _load_orderbook_snapshots(self, date_str: str) -> pl.DataFrame:
        """从 tar.gz 提取 orderbook snapshot（只取 snapshot，不处理增量更新）"""
        tar_path = self.order_dir / f"BTC-USDT-SWAP-L2orderbook-400lv-{date_str}.tar.gz"

        if not tar_path.exists():
            raise FileNotFoundError(f"Orderbook file not found: {tar_path}")

        rows = []
        with tarfile.open(str(tar_path), "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".data"):
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    for line in f:
                        try:
                            obj = json.loads(line)
                            if obj.get("action") != "snapshot":
                                continue

                            ts = int(obj["ts"])
                            # asks/bids are already parsed as lists by outer json.loads
                            asks = obj["asks"]
                            bids = obj["bids"]

                            # Top-of-book
                            best_ask = float(asks[0][0])
                            best_bid = float(bids[0][0])
                            ask_sz = float(asks[0][1])
                            bid_sz = float(bids[0][1])

                            # Depth (sum of top 5 levels)
                            ask_depth_5 = sum(float(a[1]) for a in asks[:5])
                            bid_depth_5 = sum(float(b[1]) for b in bids[:5])

                            # Full depth
                            total_ask_depth = sum(float(a[1]) for a in asks)
                            total_bid_depth = sum(float(b[1]) for b in bids)

                            rows.append({
                                "ts_ms": ts,
                                "best_ask": best_ask,
                                "best_bid": best_bid,
                                "ask_sz": ask_sz,
                                "bid_sz": bid_sz,
                                "ask_depth_5": ask_depth_5,
                                "bid_depth_5": bid_depth_5,
                                "total_ask_depth": total_ask_depth,
                                "total_bid_depth": total_bid_depth,
                            })
                        except (json.JSONDecodeError, IndexError, KeyError):
                            continue
                    break  # only the .data file

        df = pl.DataFrame(rows).sort("ts_ms").unique(subset=["ts_ms"], keep="last")
        return df

    def _load_trades(self, date_str: str) -> pl.DataFrame:
        """从 zip 加载逐笔成交"""
        zip_path = self.trade_dir / f"BTC-USDT-SWAP-trades-{date_str}.zip"

        if not zip_path.exists():
            raise FileNotFoundError(f"Trades file not found: {zip_path}")

        with zipfile.ZipFile(str(zip_path)) as zf:
            for name in zf.namelist():
                if name.endswith(".csv"):
                    with zf.open(name) as f:
                        df = pl.read_csv(
                            f,
                            columns=["side", "price", "size", "created_time"],
                        )
                        break

        # Standardize
        df = df.with_columns([
            pl.col("created_time").cast(pl.Int64).alias("ts_ms"),
            pl.col("price").cast(pl.Float64).alias("trade_px"),
            pl.col("size").cast(pl.Float64).alias("trade_sz"),
            pl.when(pl.col("side") == "buy").then(1).otherwise(-1)
              .cast(pl.Int8).alias("trade_side"),
        ]).drop(["side", "price", "size", "created_time"])

        # Filter: positive price and size
        df = df.filter(
            (pl.col("trade_px") > 0) &
            (pl.col("trade_sz") > 0)
        ).sort("ts_ms")

        return df

    def _merge_trades_book(
        self,
        trades: pl.DataFrame,
        book: pl.DataFrame,
        tolerance_ms: Optional[int] = None,
    ) -> pl.DataFrame:
        """Asof join: 每笔成交匹配最近的 orderbook 快照"""
        events = trades.join_asof(
            book,
            on="ts_ms",
            strategy="backward",
            tolerance=tolerance_ms,  # None = no tolerance, match the last snapshot
        )

        # Drop rows where no book match was found
        events = events.filter(pl.col("best_ask").is_not_null())
        return events

    def _add_features(self, events: pl.DataFrame) -> pl.DataFrame:
        """计算衍生特征 — 分步 with_columns，确保列间引用正确"""
        # Step 1: basic derived columns
        events = events.with_columns([
            ((pl.col("best_bid") + pl.col("best_ask")) / 2.0).alias("mid_px"),
            (pl.col("best_ask") - pl.col("best_bid")).alias("spread"),
            (pl.col("bid_sz") / (pl.col("bid_sz") + pl.col("ask_sz") + 1e-12)).alias("imbalance"),
            (pl.col("bid_sz") - pl.col("ask_sz")).alias("signed_imbalance"),
            (pl.col("bid_sz") + pl.col("ask_sz")).alias("total_depth"),
        ])

        # Step 2: columns that reference Step 1 outputs
        events = events.with_columns([
            (pl.col("spread") / pl.col("mid_px") * 10000).alias("spread_bps"),
            (pl.col("best_bid")
             + (pl.col("best_ask") - pl.col("best_bid"))
             * pl.col("bid_sz") / (pl.col("bid_sz") + pl.col("ask_sz") + 1e-12)
             ).alias("micro_px"),
        ])

        # Step 3: duration
        events = events.with_columns([
            pl.col("ts_ms").diff().clip(0, 60000).alias("duration_ms"),
        ])

        return events

    def load_range(self, start_date: str, end_date: str) -> pl.DataFrame:
        """加载日期范围，按日加载后拼接"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        frames = []
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            try:
                df = self.load_day(date_str)
                frames.append(df)
            except FileNotFoundError as e:
                print(f"  [WARN] {e}")
            current = pd.Timestamp(current) + pd.Timedelta(days=1)
            current = current.to_pydatetime()

        if not frames:
            raise RuntimeError("No data loaded")

        return pl.concat(frames).sort("ts_ms")


def quick_stats(df: pl.DataFrame) -> Dict:
    """快速统计"""
    return {
        "n_events": len(df),
        "date_range": f"{df['ts_ms'].min()} - {df['ts_ms'].max()}",
        "avg_spread_bps": df["spread_bps"].mean(),
        "avg_imbalance": df["imbalance"].mean(),
        "total_volume": df["trade_sz"].sum(),
        "n_buys": df.filter(pl.col("trade_side") == 1).height,
        "n_sells": df.filter(pl.col("trade_side") == -1).height,
    }


def create_l2_loader(data_root: str = r"D:\btc\raw data") -> L2DataLoader:
    return L2DataLoader(data_root)
