# Copyright (c) NXAI GmbH and its affiliates 2023
# Korbinian Poeppel

import os
from typing import Sequence, Union
import logging

import time
import random

import torch
from torch.utils.cpp_extension import load as _load

LOGGER = logging.getLogger(__name__)


def defines_to_cflags(defines=Union[dict[str, Union[int, str]], Sequence[tuple[str, Union[str, int]]]]):
    cflags = []
    if isinstance(defines, dict):
        defines = defines.items()
    for key, val in defines:
        cflags.append(f"-D{key}={str(val)}")
    return cflags


curdir = os.path.dirname(__file__)

if torch.cuda.is_available():
    from packaging import version

    if version.parse(torch.__version__) >= version.parse("2.6.0"):
        os.environ["CUDA_LIB"] = os.path.join(
            os.path.split(torch.utils.cpp_extension.include_paths(device_type="cuda")[-1])[0], "lib"
        )
    else:
        os.environ["CUDA_LIB"] = os.path.join(
            os.path.split(torch.utils.cpp_extension.include_paths(cuda=True)[-1])[0], "lib"
        )


EXTRA_INCLUDE_PATHS = (
    tuple(os.environ["XLSTM_EXTRA_INCLUDE_PATHS"].split(":")) if "XLSTM_EXTRA_INCLUDE_PATHS" in os.environ else ()
)
if "CONDA_PREFIX" in os.environ:
    # This enforces adding the correct include directory from the CUDA installation via torch. If you use the system
    # installation, you might have to add the cflags yourself.
    from pathlib import Path
    from packaging import version
    import glob

    if version.parse(torch.__version__) >= version.parse("2.6.0"):
        matching_dirs = glob.glob(f"{os.environ['CONDA_PREFIX']}/targets/**", recursive=True)
        EXTRA_INCLUDE_PATHS = (
            EXTRA_INCLUDE_PATHS
            + tuple(map(str, (Path(os.environ["CONDA_PREFIX"]) / "targets").glob("**/include/")))[:1]
        )


def get_slstm_cuda_sources() -> list[str]:
    """Returns absolute paths to all sLSTM C++/CUDA extension sources."""
    src_dir = os.path.abspath(os.path.dirname(__file__))
    return [
        os.path.join(src_dir, "cuda", "slstm.cc"),
        os.path.join(src_dir, "cuda", "slstm_forward.cu"),
        os.path.join(src_dir, "cuda", "slstm_backward.cu"),
        os.path.join(src_dir, "cuda", "slstm_backward_cut.cu"),
        os.path.join(src_dir, "cuda", "slstm_pointwise.cu"),
        os.path.join(src_dir, "util", "blas.cu"),
        os.path.join(src_dir, "util", "cuda_error.cu"),
    ]


def load(*, name, sources, extra_cflags=(), extra_cuda_cflags=(), **kwargs):
    suffix = ""
    for flag in extra_cflags:
        pref = [st[0] for st in flag[2:].split("=")[0].split("_")]
        if len(pref) > 1:
            pref = pref[1:]
        suffix += "".join(pref)
        value = flag[2:].split("=")[1].replace("-", "m").replace(".", "d")
        value_map = {"float": "f", "__half": "h", "__nv_bfloat16": "b", "true": "1", "false": "0"}
        if value in value_map:
            value = value_map[value]
        suffix += value
    if suffix:
        suffix = "_" + suffix
    suffix = suffix[:64]

    extra_cflags = list(extra_cflags) + [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        # *(f"-I{path}" for path in EXTRA_INCLUDE_PATHS)
    ]
    for eip in EXTRA_INCLUDE_PATHS:
        extra_cflags.append("-isystem")
        extra_cflags.append(eip)

    cuda_arch_flags = []
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        arch_code = f"compute_{major}{minor}"
        cuda_arch_flags = ["-gencode", f"arch={arch_code},code={arch_code}"]

    myargs = {
        "verbose": False,
        "with_cuda": True,
        "extra_ldflags": [f"-L{os.environ.get('CUDA_LIB', '/usr/local/cuda/lib64')}", "-lcublas"],
        "extra_cflags": [*extra_cflags],
        "extra_cuda_cflags": [
            *cuda_arch_flags,
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas", "-O3",
            "--extra-device-vectorization",
            *extra_cflags,
            *extra_cuda_cflags,
        ],
    }
    myargs.update(**kwargs)
    # add random waiting time to minimize deadlocks because of badly managed multicompile of pytorch ext
    time.sleep(random.random() * 10)
    LOGGER.info(f"Before compilation and loading of {name}.")
    mod = _load(name + suffix, sources, **myargs)
    LOGGER.info(f"After compilation and loading of {name}.")
    return mod
