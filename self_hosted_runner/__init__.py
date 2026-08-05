"""Assets and installer for the GitHub Actions runner tray launcher."""

from pathlib import Path
from typing import Union
import importlib.resources
import shutil


PathLike = Union[str, Path]
ASSET_NAMES = ("GitHub-Runner-Tray.ps1", "GitHub-Runner.ico")


def install_assets(runner_directory: PathLike) -> tuple[Path, Path]:
    """Copy the tray launcher script and icon into a runner installation directory."""
    destination = Path(runner_directory).expanduser().resolve()
    if not (destination / "run.cmd").is_file():
        raise FileNotFoundError(f"No GitHub Actions runner found at {destination}")

    asset_directory = importlib.resources.files(__package__).joinpath("assets")
    installed = []
    for asset_name in ASSET_NAMES:
        target = destination / asset_name
        with importlib.resources.as_file(asset_directory.joinpath(asset_name)) as source:
            shutil.copyfile(source, target)
        installed.append(target)
    return tuple(installed)  # type: ignore[return-value]