from __future__ import annotations

import hashlib
import math
import os
import random
import struct

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")


def _encode_key(value: object) -> bytes:
    if value is None:
        tag = b"N"
        body = b""
    elif isinstance(value, bool):
        tag = b"B"
        body = b"1" if value else b"0"
    elif isinstance(value, int):
        tag = b"I"
        body = str(value).encode("ascii")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("floating-point random-stream keys must be finite")
        tag = b"F"
        body = value.hex().encode("ascii")
    elif isinstance(value, str):
        tag = b"S"
        body = value.encode("utf-8")
    else:
        raise TypeError(
            "random-stream keys must be None, bool, int, finite float, or str; "
            f"got {type(value).__name__}"
        )
    return tag + struct.pack(">Q", len(body)) + body


def keyed_uniform(base_seed: int, *keys: object) -> float:
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    payload = b"".join(_encode_key(value) for value in (base_seed, *keys))
    digest = hashlib.blake2b(payload, digest_size=8, person=b"smc-crn1").digest()
    integer = int.from_bytes(digest, byteorder="big", signed=False)
    return integer / float(1 << 64)
