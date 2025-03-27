# encoding: utf-8

from pathlib import Path
import datetime as dt


def now():
    return dt.datetime.now().strftime("%Y%m%d%H%M%S")


def check_directory(file_p):
    file_p = Path(file_p)
    if file_p.suffix == "":
        dir_p = file_p
    else:
        dir_p = file_p.parent
    if not dir_p.is_dir():
        dir_p.mkdir(parents=True)
    return file_p.resolve()


def set_log(log_file: str, log_level: str):
    from logging import basicConfig

    if log_level.lower() == "error":
        from logging import ERROR

        basicConfig(
            filename=check_directory(log_file),
            level=ERROR,
            format="%(asctime)s - %(levelname)s: %(message)s",
        )
    elif log_level.lower() == "info":
        from logging import INFO

        basicConfig(
            filename=check_directory(log_file),
            level=INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
        )
    elif log_level.lower() == "debug":
        from logging import DEBUG

        basicConfig(
            filename=check_directory(log_file),
            level=DEBUG,
            format="%(asctime)s - %(levelname)s: %(message)s",
        )
    else:
        raise ValueError(f"Log level was not invalid [{log_level}]")
