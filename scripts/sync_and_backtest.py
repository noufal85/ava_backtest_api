#!/usr/bin/env python3
"""Sync 1yr of market data to local cache, then backtest ALL strategies."""
import json
import sys
import time
from datetime import date, timedelta

import requests

API = "http://localhost:8201/api/v2"


def sync_data(days: int = 365):
    """Trigger data sync and wait for completion."""
    print(f"\n{'='*60}")
    print(f"  STEP 1: Syncing {days} days of market data to local cache")
    print(f"{'='*60}\n")

    resp = requests.post(f"{API}/data/sync/all", params={"days": days})
    resp.raise_for_status()
    info = resp.json()
    print(f"Started: {info['total']} symbols to sync")

    while True:
        time.sleep(10)
        status = requests.get(f"{API}/data/sync/status").json()
        done, total = status["done"], status["total"]
        current = status.get("current", "")
        errors = len(status.get("errors", []))
        print(f"  [{done}/{total}] {current}  (errors: {errors})", flush=True)

        if status["status"] != "running":
            break

    print(f"\nSync complete! {status['done']}/{status['total']} symbols, {len(status.get('errors',[]))} errors")
    if status.get("errors"):
        for e in status["errors"][:10]:
            print(f"  ⚠ {e['symbol']}: {e['error']}")
    return status


def get_strategies():
    """Fetch all registered strategies."""
    resp = requests.get(f"{API}/strategies")
    resp.raise_for_status()
    return resp.json().get("items", resp.json() if isinstance(resp.json(), list) else [])


def run_backtest(strategy_name: str, universe: str = "sp500_liquid", days: int = 365):
    """Launch one backtest and return the run_id."""
    end = date.today()
    start = end - timedelta(days=days)
    body = {
        "strategy_name": strategy_name,
        "universe": universe,
        "market": "US",
        "initial_capital": 100000,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "parameters": {},
    }
    resp = requests.post(f"{API}/backtests", json=body)
    resp.raise_for_status()
    return resp.json()["id"]


def wait_for_backtest(run_id: str, strategy_name: str, timeout: int = 300):
    """Poll until backtest completes or fails."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(3)
        resp = requests.get(f"{API}/backtests/{run_id}")
        run = resp.json()
        status = run.get("status", "unknown")
        if status == "completed":
            metrics = run.get("metrics", run.get("results", {}))
            ret = run.get("total_return_pct", metrics.get("total_return_pct", "?"))
            sharpe = run.get("sharpe_ratio", metrics.get("sharpe_ratio", "?"))
            trades = run.get("total_trades", "?")
            return {
                "strategy": strategy_name,
                "status": "completed",
                "return_pct": ret,
                "sharpe": sharpe,
                "trades": trades,
                "duration": run.get("duration_seconds", "?"),
            }
        elif status == "failed":
            return {
                "strategy": strategy_name,
                "status": "failed",
                "error": run.get("error", "unknown"),
            }
    return {"strategy": strategy_name, "status": "timeout"}


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 365
    universe = sys.argv[2] if len(sys.argv) > 2 else "sp500_liquid"

    # Step 1: Sync
    sync_data(days)

    # Step 2: Get strategies
    strategies = get_strategies()
    strat_names = [s["name"] for s in strategies]
    print(f"\n{'='*60}")
    print(f"  STEP 2: Backtesting {len(strat_names)} strategies on {universe}")
    print(f"{'='*60}\n")

    # Step 3: Run backtests (sequential to avoid overload)
    results = []
    for i, name in enumerate(strat_names):
        print(f"[{i+1}/{len(strat_names)}] {name}...", end=" ", flush=True)
        try:
            run_id = run_backtest(name, universe, days)
            result = wait_for_backtest(run_id, name)
            if result["status"] == "completed":
                print(f"✓ return={result['return_pct']}% sharpe={result['sharpe']} trades={result['trades']}")
            else:
                print(f"✗ {result.get('error', result['status'])}")
            results.append(result)
        except Exception as e:
            print(f"✗ {e}")
            results.append({"strategy": name, "status": "error", "error": str(e)})

    # Summary
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}\n")

    completed = [r for r in results if r["status"] == "completed"]
    completed.sort(key=lambda r: float(r.get("return_pct", 0) or 0), reverse=True)

    print(f"{'Strategy':<35} {'Return%':>10} {'Sharpe':>8} {'Trades':>8}")
    print("-" * 65)
    for r in completed:
        print(f"{r['strategy']:<35} {r['return_pct']:>10} {r['sharpe']:>8} {r['trades']:>8}")

    failed = [r for r in results if r["status"] != "completed"]
    if failed:
        print(f"\n{len(failed)} strategies failed/timed out:")
        for r in failed:
            print(f"  - {r['strategy']}: {r.get('error', r['status'])}")

    # Save results
    with open("backtest_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to backtest_results.json")


if __name__ == "__main__":
    main()
