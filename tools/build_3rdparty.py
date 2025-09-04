import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from packaging.version import Version
from .patch import patch_torch_extension

patch_torch_extension()


PROJECT_ROOT = Path(os.path.dirname(__file__)).resolve().parent
THIRD_PARTY_DIR = PROJECT_ROOT / "3rdparty"

DEFAULT_BUILD_PATH = os.getenv("PRIMUS_TURBO_BUILD_DIR", PROJECT_ROOT / "build" / "3rdparty")
DEFAULT_INSTALL_PREFIX = os.getenv("PRIMUS_TURBO_INSTALL_PREFIX", PROJECT_ROOT / "install")
SKIP_DEPS_VERSION_CHECK = int(os.getenv("PRIMUS_TURBO_SKIP_DEPS_VERSION_CHECK", 1))

ROCSHMEM_HOME: Optional[Path] = None
ROCSHMEM_LIBRARY_PATH: Optional[Path] = None
ROCSHMEM_INCLUDE_PATH: Optional[Path] = None

MPI_HOME: Optional[Path] = None
UCX_HOME: Optional[Path] = None
MPI_LIBRARY_PATH: Optional[Path] = None
MPI_INCLUDE_PATH: Optional[Path] = None
UCX_LIBRARY_PATH: Optional[Path] = None
UCX_INCLUDE_PATH: Optional[Path] = None

# minimum versions for ROCSHMEM, OpenMPI, and UCX
ROCSHMEM_MINIMUM_MPI_VERSION = Version("5.0.6")
ROCSHMEM_MINIMUM_UCX_VERSION = Version("1.17.0")


@dataclass
class Library:
    name: str
    include_dirs: List[str]
    libraries: List[str]
    library_dirs: List[str]
    extra_link_args: List[str]


def build_3rdparty(
    build_dir: Path = DEFAULT_BUILD_PATH, install_prefix: Path = DEFAULT_INSTALL_PREFIX
) -> List[Library]:
    global ROCSHMEM_HOME, ROCSHMEM_LIBRARY_PATH, ROCSHMEM_INCLUDE_PATH
    global MPI_HOME, UCX_HOME, MPI_LIBRARY_PATH, MPI_INCLUDE_PATH, UCX_LIBRARY_PATH, UCX_INCLUDE_PATH
    global SKIP_DEPS_VERSION_CHECK

    _check_submodules()

    if not install_prefix.exists():
        install_prefix.mkdir(parents=True, exist_ok=True)

    if not build_dir.exists():
        build_dir.mkdir(parents=True, exist_ok=True)

    # 1. try to find existing ompi and ucx from environment variables
    # if not found, try to find them in default locations /opt/rocm/rocshmem
    ompi_home, ucx_home = _try_find_ompi_ucx_home()

    # 2. check if the found ompi and ucx versions are compatible
    # if not compatible, build them from source
    satisfied = ompi_home and ucx_home
    if satisfied:
        satisfied = _check_ompi_ucx(ompi_home, ucx_home)

    if not satisfied and not SKIP_DEPS_VERSION_CHECK:
        print(
            "[Primus-Turbo Setup] ompi and ucx not exist or not satisfy minimum version, building from source..."
        )
        ompi_home, ucx_home = _build_ompi_ucx(build_dir, install_prefix)

    # 3. set global variables for ompi and ucx
    print(f"[Primus-Turbo Setup] set MPI_HOME: {ompi_home}")
    print(f"[Primus-Turbo Setup] set UCX_HOME: {ucx_home}")

    MPI_HOME, UCX_HOME = ompi_home, ucx_home
    MPI_LIBRARY_PATH = MPI_HOME / "lib"
    MPI_INCLUDE_PATH = MPI_HOME / "include"
    UCX_LIBRARY_PATH = UCX_HOME / "lib"
    UCX_INCLUDE_PATH = UCX_HOME / "include"

    # 4. check if rocSHMEM is already installed
    # if not found, try to find it in environment variables or default locations
    # if not found, build it from source
    rocshmem_home = _try_find_rocshmem_home()
    if rocshmem_home is not None:
        print(f"[Primus-Turbo Setup] Found rocshmem at {rocshmem_home}")
        ROCSHMEM_HOME = Path(rocshmem_home)

    else:
        print("[Primus-Turbo Setup] rocshmem not found, building from source...")
        rocshmem_home = _build_rocshmem(build_dir, install_prefix)

    # 5. set global variables for rocSHMEM
    print(f"[Primus-Turbo Setup] set ROCSHMEM_HOME: {rocshmem_home}")
    ROCSHMEM_HOME = Path(rocshmem_home)
    ROCSHMEM_LIBRARY_PATH = ROCSHMEM_HOME / "lib"
    ROCSHMEM_INCLUDE_PATH = ROCSHMEM_HOME / "include"

    rocshmem_library = Library(
        name="rocshmem",
        libraries=[],
        library_dirs=[str(ROCSHMEM_LIBRARY_PATH), str(MPI_LIBRARY_PATH), str(UCX_LIBRARY_PATH)],
        include_dirs=[str(MPI_INCLUDE_PATH), str(UCX_INCLUDE_PATH), str(ROCSHMEM_INCLUDE_PATH)],
        extra_link_args=[
            f"-Wl,-rpath,{str(ROCSHMEM_LIBRARY_PATH)}",
            "-l:librocshmem.a",
            "-fgpu-rdc",
            "--hip-link",
            "-lamdhip64",
            "-lhsa-runtime64",
            "-l:libmpi.so",
            f"-Wl,-rpath,{str(MPI_LIBRARY_PATH)}",
            "-libverbs",
            "-lmlx5",
        ],
    )

    return [rocshmem_library]


def _build_rocshmem(build_dir, install_prefix) -> None:
    global MPI_LIBRARY_PATH, UCX_LIBRARY_PATH, UCX_INCLUDE_PATH, ROCSHMEM_HOME, ROCSHMEM_LIBRARY_PATH, ROCSHMEM_INCLUDE_PATH
    rocshmem_build_dir = build_dir / "rocSHMEM"
    rocshmem_dirs = install_prefix / "rocshmem"

    folders = [rocshmem_dirs, rocshmem_dirs / "include", rocshmem_dirs / "lib"]

    if all([folder.exists() for folder in folders]):
        print(f"[Primus-Turbo Setup] Using existing installed rocshmem : {rocshmem_dirs}")
    else:
        print(f"[Primus-Turbo Setup] Install rocshmem to {rocshmem_dirs}")
        rocshmem_src_dirs = THIRD_PARTY_DIR / "rocSHMEM"
        try:
            if not rocshmem_dirs.exists():
                rocshmem_dirs.mkdir(parents=True, exist_ok=True)

            if not rocshmem_build_dir.exists():
                rocshmem_build_dir.mkdir(parents=True, exist_ok=True)
            start = time.time()
            libraries = os.getenv("LD_LIBRARY_PATH", "")
            path = os.getenv("PATH", "")
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = f"{MPI_LIBRARY_PATH}:{UCX_LIBRARY_PATH}:{libraries}"
            env["PATH"] = f"{MPI_HOME}/bin:{UCX_HOME}/bin:{path}"
            cmake_prefix = os.getenv("CMAKE_PREFIX_PATH", "")
            env["CMAKE_PREFIX_PATH"] = f"{MPI_LIBRARY_PATH}:{UCX_LIBRARY_PATH}:/opt/rocm/lib:{cmake_prefix}"
            subprocess.check_call(
                ["bash", f"{rocshmem_src_dirs}/scripts/build_configs/rc", rocshmem_dirs],
                cwd=rocshmem_build_dir,
                env=env,
            )
            end = time.time()
            print(f"[Primus-Turbo Setup] Build rocshmem took {end - start:.2f} sec")
        except Exception as e:
            print(f"[Primus-Turbo Setup] Build rocshmem failed --- {e}")
            print(f"Please run:\n\t  bash {rocshmem_src_dirs}/scripts/build_configs/rc {rocshmem_dirs}")
            sys.exit(1)

    return rocshmem_dirs


def _try_find_rocshmem_home() -> Path:
    rocshmem_home = os.getenv("ROCSHMEM_HOME")
    if rocshmem_home is None:
        fallback_path = "/opt/rocm/rocshmem"
        if os.path.exists(fallback_path):
            rocshmem_home = fallback_path

    default_folders = ["lib", "include"]
    if rocshmem_home is None or not all(
        [(Path(rocshmem_home) / folder).exists() for folder in default_folders]
    ):
        return None

    return rocshmem_home


def _check_submodules() -> None:
    def _get_submodule_folders() -> list[Path]:
        git_modules_file = PROJECT_ROOT / ".gitmodules"
        default_modules_path = [
            THIRD_PARTY_DIR / name
            for name in [
                "composable_kernel",
                "rocSHMEM",
            ]
        ]
        if not git_modules_file.exists():
            return default_modules_path
        with git_modules_file.open(encoding="utf-8") as f:
            return [
                PROJECT_ROOT / line.partition("=")[-1].strip()
                for line in f
                if line.strip().startswith("path")
            ]

    def _check_for_files(folder: Path, files: list[str]) -> None:
        if not any((folder / f).exists() for f in files):
            print("Could not find any of {} in {}".format(", ".join(files), folder))
            print("Did you run 'git submodule update --init --recursive'?")
            sys.exit(1)

    def _not_exists_or_empty(folder: Path) -> bool:
        return not folder.exists() or (folder.is_dir() and next(folder.iterdir(), None) is None)

    folders = _get_submodule_folders()
    if all(_not_exists_or_empty(folder) for folder in folders):
        try:
            print("[Primus-Turbo Setup] Trying to initialize submodules")
            start = time.time()
            subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=PROJECT_ROOT)
            end = time.time()
            print(f"[Primus-Turbo Setup] Submodule initialization took {end - start:.2f} sec")
        except Exception as e:
            print(f"[Primus-Turbo Setup] Submodule initialization failed -- {e}")
            print("Please run:\n\tgit submodule update --init --recursive")
            sys.exit(1)
    for folder in folders:
        _check_for_files(
            folder,
            ["CMakeLists.txt", "Makefile", "setup.py", "LICENSE", "LICENSE.md", "LICENSE.txt", "README.md"],
        )


def _try_find_ompi_ucx_home() -> Tuple[Optional[Path], Optional[Path]]:
    ompi_home, ucx_home = os.environ.get("MPI_HOME"), os.environ.get("UCX_HOME")

    def _guess_path(execute_names: List[str]) -> Optional[str]:
        all_bin_paths = []
        for execute_name in execute_names:
            bin_home = shutil.which(execute_name)
            if bin_home is not None:
                bin_home = os.path.dirname(os.path.dirname(os.path.realpath(bin_home)))
            else:
                fallback_path = "/opt/rocm"
                if os.path.exists(fallback_path):
                    bin_home = fallback_path
            if bin_home is not None:
                all_bin_paths.append(bin_home)

        return all_bin_paths[0] if len(all_bin_paths) > 0 else None

    if ompi_home is None:
        ompi_home = _guess_path(["mpirun", "mpiexec", "ompi_info"])

    if ucx_home is None:
        ucx_home = _guess_path(["ucx_info"])

    def _warp_path(path: Optional[str]) -> Optional[Path]:
        if path is None:
            return None
        path = Path(path)
        if not path.exists():
            return None
        return path

    return _warp_path(ompi_home), _warp_path(ucx_home)


def _check_ompi_ucx(ompi_home: Path, ucx_home: Path) -> bool:
    global ROCSHMEM_MINIMUM_MPI_VERSION, ROCSHMEM_MINIMUM_UCX_VERSION

    def _get_version(commands: list[str], version_regex):
        bin_version = Version("0.0.0")
        try:
            library_dir = os.getenv("LD_LIBRARY_PATH", "")
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = f"{ompi_home}/lib:{ucx_home}/lib:{library_dir}"
            ret = subprocess.check_output(commands, text=True, env=env)
            version_match = re.search(version_regex, ret)
            if version_match:
                bin_version = Version(version_match.group(1))
            else:
                print(f"[Primus-Turbo Setup] Version not found in output: {ret}")
        except subprocess.CalledProcessError as e:
            print(f"[Primus-Turbo Setup] Error checking version: {e}")
        return bin_version

    ompi_version_commands = [f"{ompi_home}/bin/ompi_info", "--version"]
    ucx_version_commands = [f"{ucx_home}/bin/ucx_info", "-v"]

    ompi_version_regex = r"v?(\d+\.\d+\.\d+)"
    ucx_version_regex = r"(\d+\.\d+\.\d+)"

    ompi_version = _get_version(ompi_version_commands, ompi_version_regex)
    ucx_version = _get_version(ucx_version_commands, ucx_version_regex)

    ompi_satisfied = ompi_version >= ROCSHMEM_MINIMUM_MPI_VERSION
    ucx_satisfied = ucx_version >= ROCSHMEM_MINIMUM_UCX_VERSION
    print(
        f"[Primus-Turbo Setup] Found ompi version {ompi_version} {'>=' if ompi_satisfied else '<'} {ROCSHMEM_MINIMUM_MPI_VERSION}"
    )

    print(
        f"[Primus-Turbo Setup] Found ucx version {ucx_version} {'>=' if ucx_satisfied else '<'} {ROCSHMEM_MINIMUM_UCX_VERSION}"
    )

    return ompi_satisfied and ucx_satisfied


def _build_ompi_ucx(build_dir: Path, install_prefix: Path) -> Tuple[Path, Path]:
    global THIRD_PARTY_DIR
    rocshmem_src_dirs = THIRD_PARTY_DIR / "rocSHMEM"
    ompi_ucx_build_dir = build_dir / "ompi_ucx"
    install_folders = [install_prefix / "ompi", install_prefix / "ucx"]

    if not all([folder.exists() for folder in install_folders]):
        try:
            print("[Primus-Turbo Setup] Trying to install ompi and ucx")
            if not install_prefix.exists():
                os.makedirs(install_prefix)
            env = os.environ.copy()
            env["BUILD_DIR"] = ompi_ucx_build_dir
            start = time.time()
            subprocess.check_call(
                ["bash", f"{rocshmem_src_dirs}/scripts/install_dependencies.sh"],
                cwd=PROJECT_ROOT,
                env=env,
            )

            shutil.copytree(
                ompi_ucx_build_dir / "install" / "ompi", install_prefix / "ompi", dirs_exist_ok=True
            )
            shutil.copytree(
                ompi_ucx_build_dir / "install" / "ucx", install_prefix / "ucx", dirs_exist_ok=True
            )
            end = time.time()
            print(f"[Primus-Turbo Setup] Install ompi and ucx took {end - start:.2f} sec")
        except Exception as e:
            print(f"[Primus-Turbo Setup] Install ompi and ucx failed --- {e}")
            print(
                f"Please run:\n\t BUILD_DIR={ompi_ucx_build_dir} bash {rocshmem_src_dirs}/scripts/install_dependencies.sh "
            )
            sys.exit(1)
    else:
        print(f"[Primus-Turbo Setup] Found installed ompi and ucx in {install_prefix}")

    return install_prefix / "ompi", install_prefix / "ucx"
