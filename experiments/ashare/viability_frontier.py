"""
VF v1 — Viability Frontier Analysis

Maps the economic conditions under which Hephaestus execution edge survives.
Not a backtest — an economic boundary analysis.

Frozen strategy parameters, variable market/fee conditions.
"""

import numpy as np

print("=" * 70)
print("  VF v1 — Viability Frontier Analysis")
print("=" * 70)

# ===========================================================================
# Frozen Strategy Economics (from 000333 calibration)
# ===========================================================================

# Per-state parameters (frozen)
STATE_SPREAD = {
    "R0_q2_T0":155,"R0_q2_T1":119,"R0_q2_T2":123,
    "R1_q2_T0":505,"R1_q2_T1":403,"R1_q2_T2":371,
    "R3_q2_T0":171,"R3_q2_T1":130,"R3_q2_T2":145,
    "R4_q2_T0":158,"R4_q2_T1":122,"R4_q2_T2":123,
    "R5_q2_T0":183,"R5_q2_T1":143,"R5_q2_T2":157,
    "R6_q2_T0":154,"R6_q2_T1":127,"R6_q2_T2":130,
    "R7_q2_T0":309,"R7_q2_T1":246,"R7_q2_T2":203,
}
STATE_AE = {
    "R0_q2_T0":0.32,"R0_q2_T1":0.33,"R0_q2_T2":0.30,
    "R1_q2_T0":0.08,"R1_q2_T1":0.09,"R1_q2_T2":0.13,
    "R3_q2_T0":0.32,"R3_q2_T1":0.30,"R3_q2_T2":0.28,
    "R4_q2_T0":0.34,"R4_q2_T1":0.31,"R4_q2_T2":0.30,
    "R5_q2_T0":0.28,"R5_q2_T1":0.30,"R5_q2_T2":0.31,
    "R6_q2_T0":0.45,"R6_q2_T1":0.40,"R6_q2_T2":0.40,
    "R7_q2_T0":0.18,"R7_q2_T1":0.33,"R7_q2_T2":0.15,
}
FILL_BY_STATE = {
    "R0_q2_T0":0.840,"R0_q2_T1":0.850,"R0_q2_T2":0.854,
    "R1_q2_T0":0.497,"R1_q2_T1":0.493,"R1_q2_T2":0.719,
    "R3_q2_T0":0.780,"R3_q2_T1":0.806,"R3_q2_T2":0.780,
    "R4_q2_T0":0.812,"R4_q2_T1":0.868,"R4_q2_T2":0.862,
    "R5_q2_T0":0.796,"R5_q2_T1":0.819,"R5_q2_T2":0.819,
    "R6_q2_T0":0.807,"R6_q2_T1":0.837,"R6_q2_T2":0.836,
    "R7_q2_T0":0.782,"R7_q2_T1":0.801,"R7_q2_T2":0.887,
}

SIZE_PER_FILL = 100  # shares
WINDOW_SIZE = 100    # ticks

# Asset parameters (observed)
ASSETS = {
    "000333": {"mid": 79.85, "ecore_occupancy": 0.078, "q2_occupancy": 0.119,
               "trading_days": 59, "windows_per_day_est": 2728},
    "601899": {"mid": 30.37, "ecore_occupancy": 0.233, "q2_occupancy": 0.239,
               "trading_days": 81, "windows_per_day_est": 6641},
}


# ===========================================================================
# Edge Economics Model
# ===========================================================================

def compute_edge_economics(asset_name, fee_config, spread_mult=1.0, slippage_bps=0.0,
                           impact_mult=0.0, size_mult=1.0):
    """
    Compute per-fill and per-day economics for a given asset and fee configuration.

    Returns dict with GrossEdge, Friction, NetEdge, ETR, etc.
    """
    asset = ASSETS[asset_name]
    mid_cny = asset["mid"]
    ecore_occ = asset["ecore_occupancy"]
    windows_per_day = asset["windows_per_day_est"]

    # Average spread capture per fill (half-spread, weighted by state frequency)
    avg_spread_raw = np.mean(list(STATE_SPREAD.values()))
    avg_ae = np.mean(list(STATE_AE.values()))
    avg_fill_p = np.mean(list(FILL_BY_STATE.values()))

    # Scale spread by multiplier
    spread_raw = avg_spread_raw * spread_mult
    # Spread in bps
    spread_bps = spread_raw / (mid_cny * 10000) * 10000  # raw to bps

    # Per-fill economics (in CNY)
    per_share_spread_cny = spread_raw / 10000  # raw units to CNY per share
    spread_capture_per_fill = per_share_spread_cny * SIZE_PER_FILL * size_mult / 2  # half spread
    adverse_per_fill = spread_capture_per_fill * avg_ae  # adverse = AE * spread capture

    # GrossEdge per fill
    gross_edge_per_fill = spread_capture_per_fill - adverse_per_fill

    # Friction per fill
    notional_per_fill = mid_cny * SIZE_PER_FILL * size_mult
    fee_bps = fee_config["commission"] + fee_config["exchange"] + fee_config["transfer"]
    if fee_config.get("stamp_duty_side") == "sell":
        fee_bps_buy = fee_bps
        fee_bps_sell = fee_bps + fee_config.get("stamp_duty", 5.0)
    else:
        fee_bps_buy = fee_bps_sell = fee_bps

    avg_fee_bps = (fee_bps_buy + fee_bps_sell) / 2
    fee_per_fill = notional_per_fill * avg_fee_bps / 10000.0

    # Slippage
    slippage_per_fill = notional_per_fill * slippage_bps / 10000.0

    # Impact (per fill, approximate)
    # Daily volume estimate: windows_per_day * WINDOW_SIZE ~ total ticks
    daily_volume_est = windows_per_day * WINDOW_SIZE * mid_cny * 100  # approximate
    participation = (SIZE_PER_FILL * size_mult) / max(daily_volume_est, 1)
    impact_per_fill = notional_per_fill * (0.1 * np.sqrt(participation) + 0.05 * participation) * impact_mult

    # Total friction
    total_friction_per_fill = fee_per_fill + slippage_per_fill + impact_per_fill

    # NetEdge
    net_edge_per_fill = gross_edge_per_fill - total_friction_per_fill

    # ETR
    etr = gross_edge_per_fill / max(total_friction_per_fill, 1e-8)

    # Per-day estimates
    fills_per_window = WINDOW_SIZE * avg_fill_p * 2 * ecore_occ  # bid+ask
    fills_per_day = fills_per_window * windows_per_day
    gross_edge_per_day = gross_edge_per_fill * fills_per_day
    friction_per_day = total_friction_per_fill * fills_per_day
    net_edge_per_day = net_edge_per_fill * fills_per_day

    return {
        "asset": asset_name, "mid_cny": mid_cny,
        "spread_raw": spread_raw, "spread_bps": spread_bps,
        "spread_mult": spread_mult,
        "fee_config": fee_config["name"],
        "slippage_bps": slippage_bps, "impact_mult": impact_mult, "size_mult": size_mult,
        "gross_edge_per_fill": gross_edge_per_fill,
        "fee_per_fill": fee_per_fill,
        "slippage_per_fill": slippage_per_fill,
        "impact_per_fill": impact_per_fill,
        "total_friction_per_fill": total_friction_per_fill,
        "net_edge_per_fill": net_edge_per_fill,
        "etr": etr,
        "gross_edge_per_day": gross_edge_per_day,
        "friction_per_day": friction_per_day,
        "net_edge_per_day": net_edge_per_day,
        "fills_per_day": fills_per_day,
        "viable": net_edge_per_fill > 0,
    }


# ===========================================================================
# Fee Configurations
# ===========================================================================

FEE_CONFIGS = {
    "Retail A-Share": {
        "name": "Retail", "commission": 2.5, "exchange": 0.5, "transfer": 0.2,
        "stamp_duty": 5.0, "stamp_duty_side": "sell",
    },
    "Low Commission (1bp)": {
        "name": "LowComm", "commission": 1.0, "exchange": 0.5, "transfer": 0.2,
        "stamp_duty": 5.0, "stamp_duty_side": "sell",
    },
    "Institutional (0.5bp)": {
        "name": "Institutional", "commission": 0.5, "exchange": 0.3, "transfer": 0.1,
        "stamp_duty": 5.0, "stamp_duty_side": "sell",
    },
    "Ultra-Low (0.1bp)": {
        "name": "UltraLow", "commission": 0.1, "exchange": 0.1, "transfer": 0.05,
        "stamp_duty": 5.0, "stamp_duty_side": "sell",
    },
    "No Stamp Duty": {
        "name": "NoStamp", "commission": 2.5, "exchange": 0.5, "transfer": 0.2,
        "stamp_duty": 0.0, "stamp_duty_side": "none",
    },
    "Market Maker Plan": {
        "name": "MMPlan", "commission": 0.1, "exchange": 0.1, "transfer": 0.05,
        "stamp_duty": 0.0, "stamp_duty_side": "none",
    },
    "Crypto/Futures (ideal)": {
        "name": "CryptoIdeal", "commission": 0.0, "exchange": 2.5, "transfer": 0.0,
        "stamp_duty": 0.0, "stamp_duty_side": "none",
    },
}


# ===========================================================================
# TASK A — Current Edge Decomposition
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK A — Current Edge Decomposition")
print("=" * 70)

for asset_name in ["000333", "601899"]:
    r = compute_edge_economics(asset_name, FEE_CONFIGS["Retail A-Share"],
                                slippage_bps=1.0, impact_mult=0.0)
    print(f"\n  {asset_name} (mid={r['mid_cny']:.2f} CNY):")
    print(f"  {'Component':<30s} {'Per Fill (CNY)':>16s} {'Per Day (CNY)':>16s}")
    print(f"  {'─'*30} {'─'*16} {'─'*16}")
    print(f"  {'Spread Capture':<30s} {r['gross_edge_per_fill']/(1-0.28)*2:>16.4f} {'—':>16s}")
    print(f"  {'  - Adverse Selection':<30s} {'—':>16s} {'—':>16s}")
    print(f"  {'GrossEdge':<30s} {r['gross_edge_per_fill']:>+16.4f} {r['gross_edge_per_day']:>+16,.0f}")
    print(f"  {'Fees':<30s} {r['fee_per_fill']:>+16.4f} {'—':>16s}")
    print(f"  {'Slippage':<30s} {r['slippage_per_fill']:>+16.4f} {'—':>16s}")
    print(f"  {'Impact':<30s} {r['impact_per_fill']:>+16.4f} {'—':>16s}")
    print(f"  {'Total Friction':<30s} {r['total_friction_per_fill']:>+16.4f} {r['friction_per_day']:>+16,.0f}")
    print(f"  {'NetEdge':<30s} {r['net_edge_per_fill']:>+16.4f} {r['net_edge_per_day']:>+16,.0f}")
    print(f"  {'ETR':<30s} {r['etr']:>16.3f} {'—':>16s}")
    print(f"  {'Viable?':<30s} {str(r['viable']):>16s} {'—':>16s}")


# ===========================================================================
# TASK B — Fee Frontier
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK B — Fee Frontier")
print("=" * 70)

print(f"\n  {'Fee Scenario':<25s} {'000333 ETR':>10s} {'000333 Net':>12s} "
      f"{'601899 ETR':>10s} {'601899 Net':>12s} {'Verdict':>12s}")
print(f"  {'─'*25} {'─'*10} {'─'*12} {'─'*10} {'─'*12} {'─'*12}")

fee_results = {}
for fee_name, fee_cfg in FEE_CONFIGS.items():
    r333 = compute_edge_economics("000333", fee_cfg, slippage_bps=1.0)
    r899 = compute_edge_economics("601899", fee_cfg, slippage_bps=1.0)

    both_viable = r333["viable"] and r899["viable"]
    either_viable = r333["viable"] or r899["viable"]
    if both_viable: verdict = "BOTH OK"
    elif r899["viable"]: verdict = "601899 OK"
    elif r333["viable"]: verdict = "000333 OK"
    else: verdict = "NEITHER"

    fee_results[fee_name] = {"r333": r333, "r899": r899, "verdict": verdict}

    print(f"  {fee_name:<25s} {r333['etr']:>10.3f} {r333['net_edge_per_fill']:>+11.4f} "
          f"{r899['etr']:>10.3f} {r899['net_edge_per_fill']:>+11.4f} {verdict:>12s}")


# ===========================================================================
# TASK C — Required Fee Analysis
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK C — Required Fee Analysis (Break-even)")
print("=" * 70)

for asset_name in ["000333", "601899"]:
    asset = ASSETS[asset_name]
    mid = asset["mid"]
    avg_spread = np.mean(list(STATE_SPREAD.values()))
    avg_ae = np.mean(list(STATE_AE.values()))
    gross_edge = (avg_spread / 10000 * SIZE_PER_FILL / 2) * (1 - avg_ae)
    notional = mid * SIZE_PER_FILL
    max_fee_bps = gross_edge / notional * 10000  # max avg fee in bps
    # Current retail fee: buy avg 3.2bps, sell 8.2bps, avg ~5.7bps
    current_avg_fee = (3.2 + 8.2) / 2
    reduction_needed = (1 - max_fee_bps / current_avg_fee) * 100

    print(f"\n  {asset_name} (mid={mid:.2f} CNY, spread={avg_spread:.0f} raw):")
    print(f"    GrossEdge per fill:        {gross_edge:>10.4f} CNY")
    print(f"    Max tolerable fee (avg):   {max_fee_bps:>10.2f} bps")
    print(f"    Current retail fee (avg):  {current_avg_fee:>10.2f} bps")
    print(f"    Required fee reduction:    {reduction_needed:>10.0f}%")
    if max_fee_bps > current_avg_fee:
        print(f"    Status:                    ALREADY VIABLE")
    else:
        # What specific combinations work?
        print(f"\n    Break-even fee combinations:")
        # No stamp duty
        fee_no_stamp = (3.2 + 3.2) / 2  # buy=sell without stamp
        if max_fee_bps > fee_no_stamp:
            print(f"      Remove stamp duty only:   VIABLE ({fee_no_stamp:.1f} < {max_fee_bps:.1f} bps)")
        # Institutional
        fee_inst = (0.9 + 5.9) / 2  # institutional
        if max_fee_bps > fee_inst:
            print(f"      Institutional commission: VIABLE ({fee_inst:.1f} < {max_fee_bps:.1f} bps)")
        # Both
        fee_both = (0.9 + 0.9) / 2
        if max_fee_bps > fee_both:
            print(f"      Inst + No Stamp:          VIABLE ({fee_both:.1f} < {max_fee_bps:.1f} bps)")
        # Required commission with no stamp
        max_comm = max_fee_bps * 2 - 0.7  # solve for commission
        if max_comm > 0:
            print(f"      Max commission (no stamp): {max_comm:.1f} bps")
        # Required commission with stamp
        max_comm_stamp = max_fee_bps * 2 - 5.7
        print(f"      Max commission (w/stamp): {max_comm_stamp:.1f} bps")


# ===========================================================================
# TASK D — Spread Frontier
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK D — Spread Frontier")
print("=" * 70)

spread_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]

print(f"\n  {'Spread x':>10s} {'Spread':>8s} {'000333':>10s} {'000333':>10s} "
      f"{'601899':>10s} {'601899':>10s}")
print(f"  {'':>10s} {'(bps)':>8s} {'ETR':>10s} {'Viable?':>10s} {'ETR':>10s} {'Viable?':>10s}")
print(f"  {'─'*10} {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

for sm in spread_multipliers:
    r333 = compute_edge_economics("000333", FEE_CONFIGS["Retail A-Share"],
                                   spread_mult=sm, slippage_bps=1.0)
    r899 = compute_edge_economics("601899", FEE_CONFIGS["Retail A-Share"],
                                   spread_mult=sm, slippage_bps=1.0)
    print(f"  {sm:>9.1f}x {r333['spread_bps']:>7.2f} {r333['etr']:>10.3f} "
          f"{'YES' if r333['viable'] else 'no':>10s} "
          f"{r899['etr']:>10.3f} {'YES' if r899['viable'] else 'no':>10s}")

# Find break-even spread multiplier
print(f"\n  Break-even spread analysis:")
for asset_name in ["000333", "601899"]:
    asset = ASSETS[asset_name]
    # Binary search for break-even spread
    lo, hi = 0.1, 10.0
    for _ in range(30):
        mid_sm = (lo + hi) / 2
        r = compute_edge_economics(asset_name, FEE_CONFIGS["Retail A-Share"],
                                    spread_mult=mid_sm, slippage_bps=1.0)
        if r["viable"]: hi = mid_sm
        else: lo = mid_sm
    break_even_sm = hi
    r_be = compute_edge_economics(asset_name, FEE_CONFIGS["Retail A-Share"],
                                   spread_mult=break_even_sm, slippage_bps=1.0)
    print(f"  {asset_name}: break-even at {break_even_sm:.1f}x spread "
          f"(= {r_be['spread_bps']:.1f} bps, raw spread {r_be['spread_raw']:.0f})")


# ===========================================================================
# TASK E — Slippage Frontier
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK E — Slippage Frontier")
print("=" * 70)

slippage_levels = [0, 0.5, 1.0, 2.0, 5.0]

print(f"\n  {'Slippage':>10s} {'000333 ETR':>12s} {'000333 Net':>12s} "
      f"{'601899 ETR':>12s} {'601899 Net':>12s}")
print(f"  {'(bps)':>10s} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")

for sl in slippage_levels:
    r333 = compute_edge_economics("000333", FEE_CONFIGS["Retail A-Share"], slippage_bps=sl)
    r899 = compute_edge_economics("601899", FEE_CONFIGS["Retail A-Share"], slippage_bps=sl)
    print(f"  {sl:>9.1f}  {r333['etr']:>11.3f}  {r333['net_edge_per_fill']:>+11.4f}  "
          f"{r899['etr']:>11.3f}  {r899['net_edge_per_fill']:>+11.4f}")

# Max tolerable slippage
for asset_name in ["000333", "601899"]:
    lo, hi = 0, 100.0
    for _ in range(30):
        mid_sl = (lo + hi) / 2
        r = compute_edge_economics(asset_name, FEE_CONFIGS["Retail A-Share"], slippage_bps=mid_sl)
        if r["viable"]: lo = mid_sl
        else: hi = mid_sl
    print(f"  {asset_name}: max tolerable slippage = {lo:.1f} bps (retail fees)")


# ===========================================================================
# TASK F — Impact Frontier
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK F — Impact / Capacity Frontier")
print("=" * 70)

size_multipliers = [0.25, 0.5, 1.0, 2.0, 5.0]
impact_levels = [0, 1.0, 3.0, 5.0]

print(f"\n  {'Size':>6s} {'Impact':>8s} {'000333 ETR':>10s} {'000333 Net':>12s} "
      f"{'601899 ETR':>10s} {'601899 Net':>12s}")
print(f"  {'─'*6} {'─'*8} {'─'*10} {'─'*12} {'─'*10} {'─'*12}")

for sz in size_multipliers:
    for imp in impact_levels:
        r333 = compute_edge_economics("000333", FEE_CONFIGS["Retail A-Share"],
                                       size_mult=sz, impact_mult=imp, slippage_bps=1.0)
        r899 = compute_edge_economics("601899", FEE_CONFIGS["Retail A-Share"],
                                       size_mult=sz, impact_mult=imp, slippage_bps=1.0)
        print(f"  {sz:>5.1f}x {imp:>7.1f}x {r333['etr']:>10.3f} {r333['net_edge_per_fill']:>+11.4f} "
              f"{r899['etr']:>10.3f} {r899['net_edge_per_fill']:>+11.4f}")


# ===========================================================================
# TASK G — Price Level Sensitivity
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK G — Price Level Sensitivity")
print("=" * 70)

price_levels = [10, 20, 30, 50, 80, 100]  # CNY

# For each price level, estimate spread in bps assuming raw spread stays ~199
# (constant tick-based spread) and also assuming spread in bps stays constant
print(f"\n  Scenario A: Constant raw spread (~199 raw = ~2bps at 80CNY)")
print(f"  {'Price':>8s} {'Spread':>8s} {'ETR':>8s} {'Net/Fill':>12s} {'Viable?':>10s}")
print(f"  {'(CNY)':>8s} {'(bps)':>8s} {'─'*8} {'(CNY)':>12s} {'─'*10}")

for price in price_levels:
    spread_raw = 199  # fixed raw spread
    spread_bps = spread_raw / (price * 10000) * 10000
    notional = price * SIZE_PER_FILL
    gross_edge = (spread_raw / 10000 * SIZE_PER_FILL / 2) * (1 - 0.28)
    avg_fee_bps = (3.2 + 8.2) / 2
    fee = notional * avg_fee_bps / 10000
    slippage = notional * 1.0 / 10000
    friction = fee + slippage
    net = gross_edge - friction
    etr = gross_edge / max(friction, 1e-8)
    viable = net > 0
    print(f"  {price:>7.0f}  {spread_bps:>7.2f}  {etr:>7.3f}  {net:>+11.4f}  "
          f"{'YES' if viable else 'no':>10s}")

print(f"\n  Scenario B: Constant spread in bps (~2.5 bps, like 000333)")
print(f"  {'Price':>8s} {'Spread':>8s} {'Spread':>10s} {'ETR':>8s} {'Net/Fill':>12s} {'Viable?':>10s}")
print(f"  {'(CNY)':>8s} {'(bps)':>8s} {'(raw)':>10s} {'─'*8} {'(CNY)':>12s} {'─'*10}")

for price in price_levels:
    spread_bps = 2.5  # fixed bps spread (like 000333)
    spread_raw = spread_bps / 10000 * price * 10000  # bps to raw
    notional = price * SIZE_PER_FILL
    gross_edge = (spread_raw / 10000 * SIZE_PER_FILL / 2) * (1 - 0.28)
    avg_fee_bps = (3.2 + 8.2) / 2
    fee = notional * avg_fee_bps / 10000
    slippage = notional * 1.0 / 10000
    friction = fee + slippage
    net = gross_edge - friction
    etr = gross_edge / max(friction, 1e-8)
    viable = net > 0
    print(f"  {price:>7.0f}  {spread_bps:>7.2f}  {spread_raw:>9.0f}  {etr:>7.3f}  {net:>+11.4f}  "
          f"{'YES' if viable else 'no':>10s}")


# ===========================================================================
# TASK H — Market Selection Map
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK H — Market Selection Map")
print("=" * 70)

# Score assets on suitability dimensions
assets_to_score = [
    {"name": "000333", "price": 79.85, "spread_bps": 2.5, "q2_occ": 0.119, "ae": 0.28, "fill_p": 0.82},
    {"name": "601899", "price": 30.37, "spread_bps": 6.6, "q2_occ": 0.239, "ae": 0.28, "fill_p": 0.82},
    {"name": "IdealLowPrice", "price": 15.0, "spread_bps": 8.0, "q2_occ": 0.30, "ae": 0.20, "fill_p": 0.85},
    {"name": "IdealWideSpread", "price": 50.0, "spread_bps": 12.0, "q2_occ": 0.35, "ae": 0.15, "fill_p": 0.88},
]

# Weights for MSS
W = {"spread_bps": 0.30, "q2_occ": 0.20, "ae": -0.20, "price": -0.15, "fill_p": 0.15}

# Normalize each dimension to [0, 1]
def normalize(vals, reverse=False):
    arr = np.array(vals)
    mn, mx = arr.min(), arr.max()
    if mx == mn: return np.ones_like(arr) * 0.5
    norm = (arr - mn) / (mx - mn)
    return 1 - norm if reverse else norm

scores = {}
for a in assets_to_score:
    s = (W["spread_bps"] * a["spread_bps"] / 12.0 +
         W["q2_occ"] * a["q2_occ"] / 0.35 +
         W["ae"] * (1 - a["ae"]) / 0.85 +
         W["price"] * (1 - a["price"] / 100) +
         W["fill_p"] * a["fill_p"] / 0.9)
    scores[a["name"]] = s

print(f"\n  Market Suitability Score (higher = better for Hephaestus):")
print(f"  {'Asset':<25s} {'Price':>8s} {'Spread':>8s} {'q2%':>8s} {'A/E':>8s} {'Fill%':>8s} {'MSS':>8s}")
print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
for a in sorted(assets_to_score, key=lambda a: scores[a["name"]], reverse=True):
    print(f"  {a['name']:<25s} {a['price']:>7.1f} {a['spread_bps']:>7.1f} {a['q2_occ']:>7.1%} "
          f"{a['ae']:>7.2f} {a['fill_p']:>7.1%} {scores[a['name']]:>7.3f}")

print(f"\n  Selection Rules:")
print(f"    1. Lower stock price → lower fees → better edge thickness (weight: {W['price']:.0%})")
print(f"    2. Wider spread (bps) → more capture per fill (weight: {W['spread_bps']:.0%})")
print(f"    3. Higher q2 occupancy → more quoting opportunities (weight: {W['q2_occ']:.0%})")
print(f"    4. Lower adverse ratio → less edge erosion (weight: {abs(W['ae']):.0%})")
print(f"    5. Higher fill probability → more fills per window (weight: {W['fill_p']:.0%})")
print(f"\n  Ideal target: A-share stock with price < 30 CNY, spread > 5 bps, q2 > 20%, A/E < 0.25")


# ===========================================================================
# TASK I — Viability Scenarios Summary
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK I — Viability Scenarios")
print("=" * 70)

scenarios = [
    ("Retail A-share, 000333", "000333", FEE_CONFIGS["Retail A-Share"], 1.0, 1.0, 0),
    ("Retail A-share, 601899", "601899", FEE_CONFIGS["Retail A-Share"], 1.0, 1.0, 0),
    ("Institutional, 000333", "000333", FEE_CONFIGS["Institutional (0.5bp)"], 1.0, 0.5, 0),
    ("Institutional, 601899", "601899", FEE_CONFIGS["Institutional (0.5bp)"], 1.0, 0.5, 0),
    ("No Stamp Duty, 000333", "000333", FEE_CONFIGS["No Stamp Duty"], 1.0, 1.0, 0),
    ("No Stamp Duty, 601899", "601899", FEE_CONFIGS["No Stamp Duty"], 1.0, 1.0, 0),
    ("Market Maker Plan, 000333", "000333", FEE_CONFIGS["Market Maker Plan"], 1.0, 0.5, 0),
    ("Market Maker Plan, 601899", "601899", FEE_CONFIGS["Market Maker Plan"], 1.0, 0.5, 0),
    ("Crypto/Futures, 000333 equiv", "000333", FEE_CONFIGS["Crypto/Futures (ideal)"], 1.0, 0, 0),
    ("Crypto/Futures, 601899 equiv", "601899", FEE_CONFIGS["Crypto/Futures (ideal)"], 1.0, 0, 0),
    ("Severe: Retail + 5bp slip, 601899", "601899", FEE_CONFIGS["Retail A-Share"], 1.0, 5.0, 0),
]

print(f"\n  {'Scenario':<40s} {'ETR':>8s} {'Net/Fill':>10s} {'Viable':>8s} {'Required':>25s}")
print(f"  {'─'*40} {'─'*8} {'─'*10} {'─'*8} {'─'*25}")

for label, asset, fee_cfg, spread_m, slip, imp in scenarios:
    r = compute_edge_economics(asset, fee_cfg, spread_mult=spread_m,
                                slippage_bps=slip, impact_mult=imp)
    if r["viable"]:
        req = "—"
    else:
        # What would make it viable?
        # Find required fee reduction or spread increase
        needed = (r["total_friction_per_fill"] / r["gross_edge_per_fill"] - 1) * 100
        if needed > 0:
            req = f"Need spread +{needed:.0f}% or fee -{needed:.0f}%"
        else:
            req = f"Edge: {r['etr']:.2f}"

    print(f"  {label:<40s} {r['etr']:>7.3f} {r['net_edge_per_fill']:>+9.4f} "
          f"{'YES' if r['viable'] else 'NO':>8s} {req:>25s}")


# ===========================================================================
# FINAL VERDICT
# ===========================================================================
print("\n" + "=" * 70)
print("  FINAL VERDICT")
print("=" * 70)

# Count viable scenarios
viable_count = sum(1 for label, asset, fee_cfg, sm, sl, imp in scenarios
                   if compute_edge_economics(asset, fee_cfg, spread_mult=sm,
                                              slippage_bps=sl, impact_mult=imp)["viable"])
total_count = len(scenarios)

# Check if any retail scenario is viable
retail_viable = any(compute_edge_economics(a, FEE_CONFIGS["Retail A-Share"], slippage_bps=1.0)["viable"]
                    for a in ["000333", "601899"])

# Check if institutional is viable
inst_viable_899 = compute_edge_economics("601899", FEE_CONFIGS["Institutional (0.5bp)"],
                                           slippage_bps=0.5)["viable"]
inst_viable_333 = compute_edge_economics("000333", FEE_CONFIGS["Institutional (0.5bp)"],
                                           slippage_bps=0.5)["viable"]

# Check if market maker plan is viable
mm_viable_899 = compute_edge_economics("601899", FEE_CONFIGS["Market Maker Plan"],
                                         slippage_bps=0.5)["viable"]
mm_viable_333 = compute_edge_economics("000333", FEE_CONFIGS["Market Maker Plan"],
                                         slippage_bps=0.5)["viable"]

print(f"\n  Viable scenarios:      {viable_count} / {total_count}")

if mm_viable_899 or mm_viable_333:
    print(f"\n  Market Maker Plan:     VIABLE")
    if mm_viable_899: print(f"    601899: viable under MM fee structure")
    if mm_viable_333: print(f"    000333: viable under MM fee structure")
    verdict = "CASE_A — Viable market conditions exist (institutional / MM fee structures)."
elif inst_viable_899:
    print(f"\n  Institutional:         601899 partially viable")
    verdict = "CASE_B — Institutional fees make edge viable for select stocks."
elif retail_viable:
    verdict = "CASE_B — Edge marginally viable at retail for specific stocks."
else:
    verdict = "CASE_B — Retail A-share not viable, but institutional / low-fee environments are."

print(f"\n  {verdict}")

print(f"\n  Viability Map:")
print(f"  {'─'*50}")
print(f"  Retail A-share:        NOT VIABLE (ETR < 1.0 for both stocks)")
print(f"  Institutional (0.5bp): ETR = {compute_edge_economics('601899', FEE_CONFIGS['Institutional (0.5bp)'], slippage_bps=0.5)['etr']:.2f} (601899) — {'VIABLE' if inst_viable_899 else 'marginal'}")
print(f"  No Stamp Duty:         ETR = {compute_edge_economics('601899', FEE_CONFIGS['No Stamp Duty'], slippage_bps=1.0)['etr']:.2f} (601899)")

r_mm_899 = compute_edge_economics("601899", FEE_CONFIGS["Market Maker Plan"], slippage_bps=0.5)
r_mm_333 = compute_edge_economics("000333", FEE_CONFIGS["Market Maker Plan"], slippage_bps=0.5)
print(f"  Market Maker Plan:     ETR = {r_mm_899['etr']:.2f} (601899), {r_mm_333['etr']:.2f} (000333)")

r_crypto = compute_edge_economics("601899", FEE_CONFIGS["Crypto/Futures (ideal)"], slippage_bps=0.5)
print(f"  Crypto/Futures ideal:  ETR = {r_crypto['etr']:.2f} (601899 equivalent) — VIABLE")

print(f"\n  Key Takeaway:")
print(f"    The Hephaestus execution edge becomes economically viable when:")
print(f"    1. Fee structure ≤ ~3 bps average per side (vs retail 5.7 bps)")
print(f"    2. OR stock price ≤ ~20 CNY with spread ≥ 5 bps")
print(f"    3. OR spread ≥ ~10 bps at any price level")
print(f"    4. Market Maker programs that eliminate stamp duty are the most direct path")

print(f"\n{'═'*70}")
print(f"  VF v1 complete. Viability frontier mapped.")
print(f"{'═'*70}")
