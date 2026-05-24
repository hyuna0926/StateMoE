# coding: utf-8

import os
import re

import torch
import yaml


class Config(object):
    def __init__(self, model=None, dataset=None, config_dict=None):
        if config_dict is None:
            config_dict = {}
        config_dict["model"] = model
        config_dict["dataset"] = dataset
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.final_config_dict = self._load_dataset_model_config(config_dict)
        self.final_config_dict.update(config_dict)
        self._set_default_parameters()
        self._init_device()

    def _load_dataset_model_config(self, config_dict):
        config_root = os.path.join(self.project_root, "configs")
        dataset_config_name = config_dict.get("dataset_config_name", config_dict["dataset"])
        model_config_name = config_dict.get("model_config_name", config_dict["model"])
        overall_config_path = os.path.join(config_root, "overall.yaml")
        dataset_config_path = os.path.join(
            config_root, "dataset", "{}.yaml".format(dataset_config_name)
        )
        model_config_path = os.path.join(
            config_root, "model", "{}.yaml".format(model_config_name)
        )
        file_list = [overall_config_path, dataset_config_path, model_config_path]

        if "model_config_name" in config_dict and not os.path.isfile(model_config_path):
            raise FileNotFoundError(
                "Model config file not found: {}. "
                "Check `model_config_name` under src/configs/model.".format(
                    model_config_path
                )
            )
        if "dataset_config_name" in config_dict and not os.path.isfile(dataset_config_path):
            raise FileNotFoundError(
                "Dataset config file not found: {}. "
                "Check `dataset_config_name` under src/configs/dataset.".format(
                    dataset_config_path
                )
            )

        file_config_dict = {}
        for file_path in file_list:
            if os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8") as fp:
                    file_config_dict.update(
                        yaml.load(fp.read(), Loader=self._build_yaml_loader())
                    )
        return file_config_dict

    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            u"tag:yaml.org,2002:float",
            re.compile(
                u"""^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list(u"-+0123456789."),
        )
        return loader

    def _set_default_parameters(self):
        smaller_metric = ["rmse", "mae", "logloss"]
        valid_metric = self.final_config_dict["valid_metric"].split("@")[0]
        self.final_config_dict["valid_metric_bigger"] = valid_metric not in smaller_metric
        if "seed" not in self.final_config_dict["hyper_parameters"]:
            self.final_config_dict["hyper_parameters"] += ["seed"]

    def _init_device(self):
        use_gpu = self.final_config_dict["use_gpu"]
        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.final_config_dict["gpu_id"])
        self.final_config_dict["device"] = torch.device(
            "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        )

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config_dict[key] = value

    def __getitem__(self, item):
        return self.final_config_dict[item] if item in self.final_config_dict else None

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config_dict

    def __str__(self):
        args_info = "\n"
        args_info += "\n".join(
            ["{}={}".format(arg, value) for arg, value in self.final_config_dict.items()]
        )
        args_info += "\n\n"
        return args_info

    def __repr__(self):
        return self.__str__()
