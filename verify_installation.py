#!/usr/bin/env python3
"""
Installation Verification Script

Checks if Resonant Learner is installed correctly and all dependencies work.

Usage:
    python verify_installation.py
"""

import sys
import importlib


def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    display_name = package_name or module_name
    try:
        importlib.import_module(module_name)
        print(f"✅ {display_name:20s} - OK")
        return True
    except ImportError as e:
        print(f"❌ {display_name:20s} - MISSING")
        print(f"   Error: {e}")
        return False


def check_version():
    """Check Python version."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - TOO OLD")
        print(f"   Required: Python 3.8+")
        return False


def check_resonant_learner():
    """Check if resonant_learner can be imported and used."""
    try:
        from resonant_learner import ResonantCallback
        
        # Try to instantiate
        rca = ResonantCallback(verbose=False)
        
        # Try to call
        rca(val_loss=0.5)
        
        # Try to get stats
        stats = rca.get_statistics()
        
        print(f"✅ resonant_learner:20s - OK")
        print(f"   Version: {importlib.import_module('resonant_learner').__version__}")
        return True
    except Exception as e:
        print(f"❌ resonant_learner:20s - ERROR")
        print(f"   Error: {e}")
        return False


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║          🌊 RESONANT LEARNER - INSTALLATION CHECK 🌊             ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    print("Checking installation...\n")
    
    all_ok = True
    
    # Check Python version
    print("1. Python Version:")
    all_ok &= check_version()
    print()
    
    # Check core dependencies
    print("2. Core Dependencies:")
    all_ok &= check_import('torch', 'PyTorch')
    all_ok &= check_import('numpy', 'NumPy')
    print()
    
    # Check optional dependencies
    print("3. Optional Dependencies (for examples):")
    has_torchvision = check_import('torchvision', 'torchvision')
    print()
    
    # Check resonant_learner
    print("4. Resonant Learner:")
    all_ok &= check_resonant_learner()
    print()
    
    # Summary
    print("=" * 70)
    if all_ok:
        print("✅ ALL CHECKS PASSED!")
        print("\nYou're ready to use Resonant Learner! 🎉")
        print("\nNext steps:")
        print("  1. Read the Quick Start guide: docs/QUICKSTART.md")
        print("  2. Run an example: python examples/mnist_rca.py")
        print("  3. Compare with baseline: python compare_baseline_vs_rca.py")
    else:
        print("❌ SOME CHECKS FAILED!")
        print("\nPlease fix the errors above and try again.")
        print("\nCommon fixes:")
        print("  - Update Python: https://www.python.org/downloads/")
        print("  - Install PyTorch: pip install torch")
        print("  - Install package: pip install -e .")
        if not has_torchvision:
            print("  - For examples: pip install torchvision")
    
    print("=" * 70)
    
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
