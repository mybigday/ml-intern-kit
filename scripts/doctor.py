"""Print env summary: python / torch / accelerator / hf stack / hub login."""
from __future__ import annotations

import os
import platform
import sys


def main() -> None:
    print(f"python   {sys.version.split()[0]}  ({platform.platform()})")

    try:
        import torch

        backend = "cpu"
        if torch.cuda.is_available():
            backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            backend = "mps"
        print(
            f"torch    {torch.__version__}  backend={backend}  "
            f"cuda_avail={torch.cuda.is_available()}  devices={torch.cuda.device_count()}"
        )
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                arch = getattr(p, "gcnArchName", None)
                mem = p.total_memory // (1024**3)
                extra = f" arch={arch}" if arch else ""
                print(f"  gpu[{i}] {p.name}{extra}  mem={mem}GiB")
        if getattr(torch.version, "hip", None):
            print(f"  rocm/hip {torch.version.hip}")
            aotriton = os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")
            print(f"  TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL={aotriton or '<unset>'}")
    except Exception as e:
        print(f"torch    NOT INSTALLED  ({e})")

    try:
        import mlx.core as mx

        ver = getattr(mx, "__version__", "installed")
        print(f"mlx      {ver}  default_device={mx.default_device()}")
    except Exception:
        pass

    try:
        import accelerate
        import datasets
        import peft
        import transformers
        import trl

        print(f"transformers {transformers.__version__}")
        print(f"trl          {trl.__version__}")
        print(f"peft         {peft.__version__}")
        print(f"accelerate   {accelerate.__version__}")
        print(f"datasets     {datasets.__version__}")
    except Exception as e:
        print(f"hf stack   NOT FULLY INSTALLED  ({e})")

    try:
        from huggingface_hub import whoami

        print(f"hf user    {whoami().get('name', '<not logged in>')}")
    except Exception as e:
        print(f"hf user    <not logged in>  ({type(e).__name__})")


if __name__ == "__main__":
    main()
