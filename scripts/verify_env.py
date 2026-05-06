"""
verify_env.py — Quick environment check before running the pipeline.

Run this in your Kaggle notebook or local environment to confirm
all required packages are installed and compatible.

Usage:
    python scripts/verify_env.py
"""

import sys
import subprocess


def check(name, import_name=None, min_version=None):
    imp = import_name or name
    try:
        mod = __import__(imp)
        version = getattr(mod, "__version__", "unknown")
        ok = True
        if min_version:
            from packaging.version import Version
            ok = Version(version) >= Version(min_version)
        status = "✅" if ok else "⚠️ "
        print(f"  {status} {name:<30} {version}")
    except ImportError:
        print(f"  ❌ {name:<30} NOT INSTALLED")


def main():
    print("=" * 55)
    print("  Darija Pipeline — Environment Check")
    print("=" * 55)

    print("\n📦 Core scientific stack:")
    check("numpy", min_version="1.26.0")
    check("pandas", min_version="2.2.0")
    check("scipy", min_version="1.14.0")
    check("scikit-learn", import_name="sklearn", min_version="1.5.0")
    check("pyarrow", min_version="20.0.0")

    print("\n🎧 Audio packages:")
    check("librosa", min_version="0.10.0")
    check("soundfile")
    check("pydub")
    check("pyloudnorm")

    print("\n🤖 NLP / ML packages:")
    check("torch")
    check("transformers", min_version="4.44.0")
    check("datasets", min_version="3.0.0")
    check("accelerate")
    check("seqeval")

    print("\n📊 Data tools:")
    check("openpyxl")
    check("tqdm")

    print("\n🎙️ NVIDIA NeMo:")
    try:
        import nemo
        print(f"  ✅ {'nemo_toolkit':<30} {nemo.__version__}")
    except ImportError:
        print(f"  ❌ {'nemo_toolkit':<30} NOT INSTALLED")
    except Exception as e:
        print(f"  ⚠️  nemo_toolkit — import error: {e}")

    print("\n💻 GPU status:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available — {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"     VRAM: {vram:.1f} GB")
        else:
            print("  ⚠️  No GPU detected — enable GPU in Kaggle Settings > Accelerator")
    except Exception as e:
        print(f"  ❌ Could not check GPU: {e}")

    print("\n" + "=" * 55)
    print("  If any ❌ appear, run STEP 0 in the notebook first.")
    print("=" * 55)


if __name__ == "__main__":
    main()
