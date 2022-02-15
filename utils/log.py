from pathlib import Path
from typing import Optional

_private_logfile = None


def set_logfile(logfile: Path) -> None:
    _private_logfile = logfile


def log(*print_args, logfile : Optional[Path] = None) -> None:
    """ Maybe logs the output also in the file `outfile` """
    if logfile:
        if not logfile.parent.exists():
            logfile.parent.mkdir(exist_ok=True, parents=True)
        with open(logfile, 'a') as f:
            print(*print_args, file=f)
    if _private_logfile:
        if not _private_logfile.parent.exists():
            _private_logfile.parent.mkdir(exist_ok=True, parents=True)
        with open(logfile, 'a') as f:
            print(*print_args, file=f)
    print(*print_args)

