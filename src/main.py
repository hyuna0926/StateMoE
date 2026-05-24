# coding: utf-8

import argparse

from utils.quick_start import quick_start


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="StateMoE", help="name of model")
    parser.add_argument("--dataset", "-d", type=str, default="baby", help="dataset name")
    parser.add_argument(
        "--model_config_name",
        type=str,
        default=None,
        help="config file name under src/configs/model without .yaml",
    )
    parser.add_argument("--gpu_id", "-g", type=str, default="0", help="gpu id")
    parser.add_argument("--missing_modal", type=int, default=0, help="missing modal flag")
    parser.add_argument("--missing_ratio", type=str, default="0.0", help="missing ratio")
    parser.add_argument("--split_mode", type=str, default=None, help="x_label or recbole_ls")
    parser.add_argument("--split_order", type=str, default=None, help="split order for recbole_ls")

    args, _ = parser.parse_known_args()
    config_dict = {
        "gpu_id": args.gpu_id,
        "missing_modal": args.missing_modal,
        "missing_ratio": eval(args.missing_ratio),
    }
    if args.split_mode is not None:
        config_dict["split_mode"] = args.split_mode
    if args.split_order is not None:
        config_dict["split_order"] = args.split_order
    if args.model_config_name:
        config_dict["model_config_name"] = args.model_config_name
    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=False)
