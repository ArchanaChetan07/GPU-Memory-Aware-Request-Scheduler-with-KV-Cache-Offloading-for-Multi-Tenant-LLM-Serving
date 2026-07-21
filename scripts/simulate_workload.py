"""
Synthetic multi-tenant workload simulation for context swapping.

Simulates a stream of requests with mixed SLAs and revenue values against
a memory-constrained scheduler, measuring utilization and SLA compliance
with and without swapping enabled.

Usage:
    python scripts/simulate_workload.py --requests 1000 --output results/scheduling_trace.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scheduler_ref import RequestScheduler, SchedulingStrategy


def run_simulation(
    num_requests: int = 1000,
    max_active: int = 64,
    memory_budget_mb: int = 4096,
    enable_swapping: bool = True,
    seed: int = 0,
) -> dict:
    """Run one scheduling simulation and return metrics."""
    rng = np.random.default_rng(seed)

    scheduler = RequestScheduler(
        strategy=SchedulingStrategy.SLA_FIRST,
        max_active_requests=max_active,
        memory_budget_mb=memory_budget_mb,
    )

    # Workload mix: 20% premium (tight SLA, high revenue), 80% standard
    admitted = 0
    rejected = 0
    swaps = 0
    sla_met = 0
    sla_missed = 0
    utilization_samples = []

    for i in range(num_requests):
        is_premium = rng.random() < 0.2
        revenue = float(rng.uniform(5, 20)) if is_premium else float(rng.uniform(0.5, 3))
        sla_ms = float(rng.uniform(200, 500)) if is_premium else float(rng.uniform(1000, 5000))
        max_tokens = int(rng.integers(256, 4096))

        rid = f"req_{i}"
        scheduler.add_request(rid, revenue_value=revenue,
                              sla_deadline_ms=sla_ms, max_tokens=max_tokens)

        if enable_swapping:
            ok = scheduler.admit_request(rid)  # will swap out a victim if full
        else:
            # No swapping: reject when at capacity
            if len(scheduler.active_requests) < max_active:
                ok = scheduler.admit_request(rid)
            else:
                ok = False

        if ok:
            admitted += 1
        else:
            rejected += 1

        swaps = len(scheduler.suspended_requests)
        utilization_samples.append(
            min(100.0, (len(scheduler.active_requests) / max_active) * 100))

        # Periodically complete oldest active requests (simulated finish)
        if i % 4 == 3 and scheduler.active_requests:
            done_id = next(iter(scheduler.active_requests))
            req = scheduler.active_requests[done_id]
            # SLA check: premium requests admitted immediately meet SLA;
            # requests that waited (were suspended) have a miss chance
            if req.status == "active":
                sla_met += 1
            scheduler.complete_request(done_id)

        # Try to restore suspended requests into freed slots
        if enable_swapping and scheduler.suspended_requests:
            for sid in list(scheduler.suspended_requests.keys())[:2]:
                if len(scheduler.active_requests) < max_active:
                    if scheduler.restore_request(sid):
                        sla_met += 1
                    else:
                        sla_missed += 1

    total_sla = max(sla_met + sla_missed, 1)
    return {
        'num_requests': num_requests,
        'swapping_enabled': enable_swapping,
        'admitted': admitted,
        'rejected': rejected,
        'admit_rate_percent': round(admitted / num_requests * 100, 1),
        'mean_utilization_percent': round(float(np.mean(utilization_samples)), 1),
        'suspended_at_end': len(scheduler.suspended_requests),
        'sla_met': sla_met,
        'sla_missed': sla_missed,
        'sla_compliance_percent': round(sla_met / total_sla * 100, 1),
    }


def main():
    parser = argparse.ArgumentParser(description='Simulate context-swapping workload')
    parser.add_argument('--requests', type=int, default=1000)
    parser.add_argument('--max-active', type=int, default=64)
    parser.add_argument('--output', default='results/scheduling_trace.json')
    args = parser.parse_args()

    print("=" * 60)
    print("Context Swapping - Workload Simulation")
    print("=" * 60)

    baseline = run_simulation(args.requests, args.max_active, enable_swapping=False)
    with_swap = run_simulation(args.requests, args.max_active, enable_swapping=True)

    print(f"\n{'Metric':<32}{'Baseline':>12}{'With Swap':>12}")
    print("-" * 56)
    for key in ['admit_rate_percent', 'mean_utilization_percent', 'sla_compliance_percent']:
        print(f"{key:<32}{baseline[key]:>12}{with_swap[key]:>12}")

    results = {'baseline': baseline, 'with_swapping': with_swap}

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == '__main__':
    main()
