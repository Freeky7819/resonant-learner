#!/usr/bin/env python3
"""
Quick Comparison Script

Runs baseline and RCA side-by-side on MNIST for quick verification.

Usage:
    python compare_baseline_vs_rca.py
"""

import subprocess
import time
import sys


def run_command(cmd, label):
    """Run a command and time it."""
    print("\n" + "=" * 70)
    print(f"🚀 Running: {label}")
    print("=" * 70)
    
    start = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"❌ {label} failed!")
        return None
    
    return elapsed


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║       🌊 RESONANT LEARNER - BASELINE vs RCA COMPARISON 🌊        ║
║                                                                   ║
║  This script will run MNIST training twice:                       ║
║    1. Baseline (traditional training)                             ║
║    2. With RCA (intelligent early stopping)                       ║
║                                                                   ║
║  Watch the difference! 💙                                        ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Test 1: Baseline
    baseline_time = run_command(
        "python examples/mnist_rca.py --baseline --epochs 20",
        "BASELINE (Traditional Training)"
    )
    
    if baseline_time is None:
        print("\n❌ Baseline test failed. Stopping.")
        sys.exit(1)
    
    print(f"\n⏱️  Baseline completed in {baseline_time:.1f}s")
    
    # Test 2: RCA
    rca_time = run_command(
        "python examples/mnist_rca.py --epochs 20",
        "RCA (Intelligent Early Stopping)"
    )
    
    if rca_time is None:
        print("\n❌ RCA test failed. Stopping.")
        sys.exit(1)
    
    print(f"\n⏱️  RCA completed in {rca_time:.1f}s")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Baseline time: {baseline_time:.1f}s")
    print(f"RCA time:      {rca_time:.1f}s")
    
    if rca_time < baseline_time:
        savings = (1 - rca_time / baseline_time) * 100
        print(f"\n🎉 RCA was {savings:.1f}% faster!")
        print(f"   Time saved: {baseline_time - rca_time:.1f}s")
    else:
        print(f"\n⚠️  RCA took longer (this can happen with small datasets)")
    
    print("\n💡 Key Takeaway:")
    print("   Check the epoch counts in the logs above!")
    print("   RCA should have stopped earlier while maintaining accuracy.")
    print("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Comparison interrupted by user.")
        sys.exit(1)
