# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Builds and loads the target's CUDA kernels (the sources under ``csrc/``) at first use.

The sources compile into one module with torch's extension builder, for the arch-specific target
of the device (sm_100a on B200, sm_103a on B300 / GB300). The build is cached under
``$TRTLLM_MODELING_V2_CACHE`` (default ``~/.cache/tensorrt_llm/modeling_v2``) in a directory keyed
by the sources, the flags and the torch / CUDA versions, so an unchanged tree loads in a second and
a changed one never loads a stale build. The ranks of one model share it: the builder locks the
directory, one rank compiles and the others wait. Nothing compiles at import; the first read of an
attribute of ``ops`` does.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from torch.utils import cpp_extension

from tensorrt_llm.logger import logger

_HERE = Path(__file__).resolve().parent
_CSRC = _HERE / "csrc"

_CXX_FLAGS = ["-O3", "-std=c++20"]
_CUDA_FLAGS = ["-O3", "-std=c++20", "--use_fast_math", "-lineinfo"]


def _arch() -> str:
    major, minor = torch.cuda.get_device_capability()
    cc = 10 * major + minor
    if cc not in (100, 103):
        raise RuntimeError(f"the GLM-5.3-Flash FP8 kernels run on sm_100 / sm_103, not on sm_{cc}")
    return f"{cc}a"


def _cuda_flags() -> list[str]:
    a = _arch()
    return _CUDA_FLAGS + [f"-gencode=arch=compute_{a},code=sm_{a}"]


def _sources() -> list[Path]:
    return sorted(_CSRC.glob("*.cpp")) + sorted((_CSRC / "kernels").glob("*.cu"))


def _build_key() -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in _CSRC.rglob("*") if p.is_file()) + [Path(__file__)]:
        digest.update(str(path.relative_to(_HERE)).encode() + b"\0")
        digest.update(path.read_bytes())
    digest.update(" ".join(_CXX_FLAGS + _cuda_flags()).encode())
    digest.update(torch.__version__.encode())
    digest.update(str(torch.version.cuda).encode())
    return digest.hexdigest()[:16]


def _load():
    key = _build_key()
    cache = Path(
        os.environ.get("TRTLLM_MODELING_V2_CACHE", "~/.cache/tensorrt_llm/modeling_v2")
    ).expanduser()
    build_dir = cache / "glm53_fp8" / key
    build_dir.mkdir(parents=True, exist_ok=True)
    name = f"glm53_fp8_kernels_{key}"
    if not any(build_dir.glob(f"{name}*.so")):
        logger.info(
            f"GLM-5.3-Flash FP8 kernels: compiling {len(_sources())} sources for sm_{_arch()} "
            f"into {build_dir} (first use of this tree)"
        )
    return cpp_extension.load(
        name=name,
        sources=[str(p) for p in _sources()],
        extra_include_paths=[str(_CSRC)],
        extra_cflags=_CXX_FLAGS,
        extra_cuda_cflags=_cuda_flags(),
        build_directory=str(build_dir),
        verbose=os.environ.get("TRTLLM_MODELING_V2_JIT_VERBOSE", "0") == "1",
    )


class _Ops:
    """The kernel module's launchers and constants, compiled and loaded at the first read."""

    _module = None

    def __getattr__(self, name: str):
        if _Ops._module is None:
            _Ops._module = _load()
        return getattr(_Ops._module, name)


ops = _Ops()
