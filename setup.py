import glob
import os
import platform
import re
import subprocess
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# from primus_turbo.utils.hip_extension import HIPExtension

PROJECT_ROOT = Path(os.path.dirname(__file__)).resolve()
DEFAULT_HIPCC = "/opt/rocm/bin/hipcc"


def all_files_in_dir(path, name_extensions=[]):
    all_files = []
    for dirname, _, names in os.walk(path):
        for name in names:
            skip = True
            for name_extension in name_extensions:
                if name_extension in name:
                    skip = False
                    break
            if skip:
                continue
            all_files.append(Path(dirname, name))

    return all_files


def setup_cxx_env():
    user_cxx = os.environ.get("CXX")
    if user_cxx:
        print(f"[Primus-Turbo Setup] Using user-provided CXX: {user_cxx}")
    else:
        os.environ["CXX"] = DEFAULT_HIPCC
        print(f"[Primus-Turbo Setup] No CXX provided. Defaulting to: {DEFAULT_HIPCC}")

    os.environ.setdefault("CMAKE_CXX_COMPILER", os.environ["CXX"])
    os.environ.setdefault("CMAKE_HIP_COMPILER", os.environ["CXX"])
    print(f"[Primus-Turbo Setup] CMAKE_CXX_COMPILER set to: {os.environ['CMAKE_CXX_COMPILER']}")
    print(f"[Primus-Turbo Setup] CMAKE_HIP_COMPILER set to: {os.environ['CMAKE_HIP_COMPILER']}")


def read_version():
    with open(os.path.join("primus_turbo", "__init__.py")) as f:
        for line in f:
            match = re.match(r"^__version__\s*=\s*[\"'](.+?)[\"']", line)
            if match:
                return match.group(1)
    raise RuntimeError("Cannot find version.")


def find_shared_lib(name: str) -> str:
    so_files = glob.glob(str(PROJECT_ROOT / "build" / "lib.*" / f"{name}*.so"))
    if not so_files:
        raise FileNotFoundError(f"{name}.so not found after build")
    print(f"[Primus-Turbo Setup] Found shared lib: {so_files[0]}")
    return so_files[0]


def get_common_flags():
    arch = platform.machine().lower()
    extra_link_args = [
        "-Wl,-rpath,/opt/rocm/lib",
        f"-L/usr/lib/{arch}-linux-gnu",
    ]

    cxx_flags = [
        "-O3",
        "-fvisibility=hidden",
    ]

    nvcc_flags = [
        "-O3",
        "-U__HIP_NO_HALF_OPERATORS__",
        "-U__HIP_NO_HALF_CONVERSIONS__",
        "-U__HIP_NO_BFLOAT16_OPERATORS__",
        "-U__HIP_NO_BFLOAT16_CONVERSIONS__",
        "-U__HIP_NO_BFLOAT162_OPERATORS__",
        "-U__HIP_NO_BFLOAT162_CONVERSIONS__",
        "-fno-offload-uniform-block",
        "-mllvm",
        "--lsr-drop-solution=1",
        "-mllvm",
        "-enable-post-misched=0",
        "-mllvm",
        "-amdgpu-coerce-illegal-types=1",
        "-mllvm",
        "-amdgpu-early-inline-all=true",
        "-mllvm",
        "-amdgpu-function-calls=false",
    ]

    # Device Arch
    # TODO: Add ENV Setting
    # TODO: ROCM Version support
    # nvcc_flags += [
    #     "--offload-arch=gfx942",
    #     "--offload-arch=gfx950",
    # ]

    max_jobs = int(os.getenv("MAX_JOBS", "4"))
    nvcc_flags.append(f"-parallel-jobs={max_jobs}")

    return {
        "extra_link_args": extra_link_args,
        "extra_compile_args": {
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
    }


# def build_csrc_kernels_extension():
#     flags = get_common_flags()
#     return HIPExtension(
#     )


def build_torch_extension():
    # Link and Compile flags
    extra_flags = get_common_flags()
    extra_flags["extra_link_args"] = [
        "-Wl,-rpath,$ORIGIN/../../torch/lib",
        *extra_flags.get("extra_link_args", []),
    ]

    # CPP
    kernels_source_files = Path(PROJECT_ROOT / "csrc" / "kernels")
    pytorch_csrc_source_files = Path(PROJECT_ROOT / "csrc" / "pytorch")
    sources = (
        [pytorch_csrc_source_files / "bindings_pytorch.cpp"]
        + all_files_in_dir(pytorch_csrc_source_files, name_extensions=["cpp", "cc", "cu"])
        + all_files_in_dir(kernels_source_files, name_extensions=["cpp", "cc", "cu"])
    )

    return CUDAExtension(
        name="primus_turbo.pytorch._C",
        sources=sources,
        include_dirs=[
            Path(PROJECT_ROOT / "csrc" / "include"),
            Path(PROJECT_ROOT / "3rdparty" / "composable_kernel" / "include"),
            Path(PROJECT_ROOT / "csrc"),
        ],
        libraries=["hipblas"],
        **extra_flags,
    )


def compile_aiter():
    aiter_dir = os.path.join("3rdparty", "aiter")
    subprocess.run(["python3", "setup.py", "develop"], cwd=aiter_dir, check=True)


if __name__ == "__main__":
    # set cxx
    setup_cxx_env()
    # Compile aiter before setting up the main package
    compile_aiter()

    setup(
        name="primus_turbo",
        version=read_version(),
        packages=find_packages(),
        ext_modules=[build_torch_extension()],
        cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    )
