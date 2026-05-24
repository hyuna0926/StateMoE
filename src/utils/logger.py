# coding: utf-8

import logging
import os

from utils.utils import get_local_time


def init_logger(config):
    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    log_root = config["log_dir"] or os.path.join(src_root, "log", config["model"])
    os.makedirs(log_root, exist_ok=True)

    logfilename = "{}-{}-{}-{}-{}.log".format(
        config["model"],
        config["dataset"],
        get_local_time(),
        f"new_items{config['new_items']}",
        f"missing_modal{config['missing_modal']}",
    )
    if config["missing_modal"]:
        logfilename = "{}-{}-{}-{}-{}-{}.log".format(
            config["model"],
            config["dataset"],
            get_local_time(),
            f"new_items{config['new_items']}",
            f"missing_modal{config['missing_modal']}",
            f"missing_imputation{config['missing_imputation']}",
        )
    logfilepath = os.path.join(log_root, logfilename)

    model_save_root = os.path.join(log_root, "models")
    os.makedirs(model_save_root, exist_ok=True)
    config["save_name"] = os.path.join(model_save_root, logfilename)

    filefmt = "%(asctime)-15s %(levelname)s %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(asctime)-15s %(levelname)s %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = logging.Formatter(sfmt, sdatefmt)

    state = config["state"]
    if state is None or state.lower() == "info":
        level = logging.INFO
    elif state.lower() == "debug":
        level = logging.DEBUG
    elif state.lower() == "error":
        level = logging.ERROR
    elif state.lower() == "warning":
        level = logging.WARNING
    elif state.lower() == "critical":
        level = logging.CRITICAL
    else:
        level = logging.INFO

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    disable_file_log = (
        bool(config["disable_file_log"]) if config["disable_file_log"] is not None else False
    )
    handlers = [sh]
    if not disable_file_log:
        fh = logging.FileHandler(logfilepath, "w", "utf-8")
        fh.setLevel(level)
        fh.setFormatter(fileformatter)
        handlers.append(fh)

    logging.basicConfig(level=level, handlers=handlers)
