import os
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .patch import patch_torch_extension

patch_torch_extension()


TURBO_FALLBACK_LIBRARY_HOME = "/opt/rocm"


@dataclass
class Library:
    name: str
    include_dirs: List[str]
    libraries: List[str]
    library_dirs: List[str]
    extra_link_args: List[str]


def _guess_library_home(exec_names: List[str]) -> Optional[str]:
    all_exec_paths = []
    for execute_name in exec_names:
        exec_home = shutil.which(execute_name)
        if exec_home is not None:
            exec_home = os.path.dirname(os.path.dirname(os.path.realpath(exec_home)))
        else:
            fallback_path = TURBO_FALLBACK_LIBRARY_HOME
            if os.path.exists(fallback_path):
                exec_home = fallback_path
        if exec_home is not None:
            all_exec_paths.append(exec_home)

    return all_exec_paths[0] if len(all_exec_paths) > 0 else None


def _find_library_home(env_names: List[str], exec_names: Optional[List[str]] = None) -> Optional[str]:
    lib_home = None
    for env in env_names:
        lib_home = lib_home or os.getenv(env)
        if lib_home:
            break

    if lib_home is None and exec_names is not None:
        lib_home = _guess_library_home(exec_names)

    if not os.path.exists(lib_home):
        return None
    return lib_home


def find_mpi_home():
    return _find_library_home(
        env_names=["MPI_HOME", "MPI_PATH", "MPI_DIR"], exec_names=["mpirun", "mpicc", "ompi_info"]
    )


def find_rocshmem_home():
    return _find_library_home(env_names=["ROCSHMEM_HOME", "ROCSHMEM_PATH", "ROCSHMEM_DIR"])


def find_rocshmem_library() -> Library:
    rocshmem_home = find_rocshmem_home()
    mpi_home = find_mpi_home()
    if rocshmem_home is None or mpi_home is None:
        warnings.warn(
            "rocSHMEM or MPI library is not found, internode of DeepEP will be disabled. Please set 'MPI_HOME' and 'ROCSHMEM_HOME' env variables and reinstall."
        )
        return None

    rocshmem_home = Path(rocshmem_home).resolve()
    mpi_home = Path(mpi_home).resolve()

    rocshmem_library_dir = str(rocshmem_home / "lib")
    rocshmem_include_dir = str(rocshmem_home / "include")
    mpi_library_dir = str(mpi_home / "lib")
    mpi_include_dir = str(mpi_home / "include")
    rocshmem_library = Library(
        name="rocshmem",
        libraries=[],
        library_dirs=[rocshmem_library_dir, mpi_library_dir],
        include_dirs=[rocshmem_include_dir, mpi_include_dir],
        extra_link_args=[
            f"-Wl,-rpath,{rocshmem_library_dir}",
            "-l:librocshmem.a",
            "-fgpu-rdc",
            "--hip-link",
            "-lamdhip64",
            "-lhsa-runtime64",
            "-l:libmpi.so",
            f"-Wl,-rpath,{mpi_library_dir}",
            "-libverbs",
            "-lmlx5",
        ],
    )
    return rocshmem_library
