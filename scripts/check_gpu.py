"""Diagnostic — verify torch sees the GPU and the CUDA build matches the card.

Run after install:

    python scripts/check_gpu.py

For RTX 50-series (Blackwell, sm_120) you MUST have torch built against
CUDA 12.8 or newer. Older builds will print "no kernel image is available"
when you actually try to use the device.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("torch is not installed. Run:")
        print("  pip install --index-url https://download.pytorch.org/whl/cu128 "
              "torch==2.7.0 torchvision==0.22.0")
        return 1

    print(f"torch         : {torch.__version__}")
    print(f"cuda built    : {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("\nGPU NOT visible. Either no NVIDIA driver, or this torch is a CPU build.")
        print("Forensic AI will still run via heuristic fallback / CPU path.")
        return 2

    n = torch.cuda.device_count()
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        cap = f"sm_{props.major}{props.minor}"
        mem_gb = props.total_memory / (1024 ** 3)
        print(f"\nGPU {i}: {props.name}")
        print(f"  compute cap : {cap}")
        print(f"  total VRAM  : {mem_gb:.1f} GB")
        print(f"  multiproc   : {props.multi_processor_count}")

    # Live kernel test — catches "no kernel image" on Blackwell-with-old-torch.
    print("\nRunning a tiny matmul on cuda:0 ...")
    try:
        a = torch.randn(256, 256, device="cuda:0", dtype=torch.float16)
        b = torch.randn(256, 256, device="cuda:0", dtype=torch.float16)
        c = (a @ b).sum().item()
        print(f"  ok — result sum = {c:.2f}")
    except RuntimeError as e:
        print(f"  FAILED: {e}")
        print("\nIf this says 'no kernel image is available for execution on the device',")
        print("your torch CUDA build is too old for this GPU. Reinstall:")
        print("  pip install --index-url https://download.pytorch.org/whl/cu128 "
              "torch==2.7.0 torchvision==0.22.0 --force-reinstall")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
