from __future__ import annotations

import json
import platform
import sys

import torch


def collect_hardware_info() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; expected the connected RTX 5090.")

    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    compiled_arches = torch.cuda.get_arch_list()
    required_arch = f"sm_{capability[0]}{capability[1]}"

    vector = torch.tensor([1.0, 2.0, 3.0], device=device)
    result = torch.sum(vector * vector)
    torch.cuda.synchronize(device)
    if float(result.item()) != 14.0:
        raise RuntimeError(f"unexpected CUDA smoke-test result: {result.item()}")

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(device),
        "compute_capability": list(capability),
        "compiled_arches": compiled_arches,
        "reported_device_arch": required_arch,
        "cuda_smoke_result": float(result.item()),
    }


def main() -> None:
    print(json.dumps(collect_hardware_info(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
