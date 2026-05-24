# coding: utf-8

from itertools import product
from logging import getLogger
import os
import platform

from utils.configurator import Config
from utils.dataloader import EvalDataLoader, TrainDataLoader
from utils.dataset import RecDataset
from utils.logger import init_logger
from utils.utils import dict2str, get_model, get_trainer, init_seed


def quick_start(model, dataset, config_dict, save_model=True):
    config = Config(model, dataset, config_dict)
    init_logger(config)
    logger = getLogger()
    logger.info("██Server: \t" + platform.node())
    logger.info("██Dir: \t" + os.getcwd() + "\n")
    logger.info(config)

    dataset = RecDataset(config)
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info("\n====Training====\n" + str(train_dataset))
    logger.info("\n====Validation====\n" + str(valid_dataset))
    logger.info("\n====Testing====\n" + str(test_dataset))

    train_data = TrainDataLoader(
        config, train_dataset, batch_size=config["train_batch_size"], shuffle=True
    )
    valid_data = EvalDataLoader(
        config,
        valid_dataset,
        additional_dataset=train_dataset,
        batch_size=config["eval_batch_size"],
    )
    test_data = EvalDataLoader(
        config,
        test_dataset,
        additional_dataset=train_dataset,
        batch_size=config["eval_batch_size"],
    )

    hyper_ret = []
    val_metric = config["valid_metric"].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info("\n\n=================================\n\n")

    hyper_ls = []
    if "seed" not in config["hyper_parameters"]:
        config["hyper_parameters"] = ["seed"] + config["hyper_parameters"]
    for param in config["hyper_parameters"]:
        hyper_ls.append(config[param] or [None])
    combinators = list(product(*hyper_ls))

    for i, hyper_tuple in enumerate(combinators):
        for param_name, param_value in zip(config["hyper_parameters"], hyper_tuple):
            config[param_name] = param_value
        init_seed(config["seed"])

        logger.info(
            "========={}/{}: Parameters:{}={}=======".format(
                idx + 1, len(combinators), config["hyper_parameters"], hyper_tuple
            )
        )

        train_data.pretrain_setup()
        model_inst = get_model(config["model"])(config, train_data).to(config["device"])
        model_inst.logger = logger
        logger.info(model_inst)

        trainer = get_trainer()(config, model_inst)
        save_dir = config["save_name"][:-4] + "-" + str(hyper_tuple)
        model_inst.save_dir = save_dir

        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(
            train_data,
            valid_data=valid_data,
            test_data=test_data,
            saved=save_model,
            save_dir=save_dir,
        )

        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))
        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        logger.info("best valid result: {}".format(dict2str(best_valid_result)))
        logger.info("test result: {}".format(dict2str(best_test_upon_valid)))
        logger.info(
            "████Current BEST████:\nParameters: {}={},\nValid: {},\nTest: {}\n\n\n".format(
                config["hyper_parameters"],
                hyper_ret[best_test_idx][0],
                dict2str(hyper_ret[best_test_idx][1]),
                dict2str(hyper_ret[best_test_idx][2]),
            )
        )

    logger.info("\n============All Over=====================")
    for p, k, v in hyper_ret:
        logger.info(
            "Parameters: {}={},\n best valid: {},\n best test: {}".format(
                config["hyper_parameters"], p, dict2str(k), dict2str(v)
            )
        )

    logger.info("\n\n█████████████ BEST ████████████████")
    logger.info(
        "\tParameters: {}={},\nValid: {},\nTest: {}\n\n".format(
            config["hyper_parameters"],
            hyper_ret[best_test_idx][0],
            dict2str(hyper_ret[best_test_idx][1]),
            dict2str(hyper_ret[best_test_idx][2]),
        )
    )
