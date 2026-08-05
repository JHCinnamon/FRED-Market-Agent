"""Install the packaged runner tray assets into an existing runner directory."""

import argparse
from pathlib import Path

from . import install_assets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install GitHub Actions runner tray assets into a runner directory."
    )
    parser.add_argument("runner_directory", type=Path)
    arguments = parser.parse_args()
    script_path, icon_path = install_assets(arguments.runner_directory)
    print(f"Installed {script_path.name} and {icon_path.name} in {script_path.parent}")


if __name__ == "__main__":
    main()