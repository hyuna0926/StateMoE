# coding: utf-8

import os

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from utils.metrics import metrics_dict
from utils.utils import get_local_time


topk_metrics = {
    metric.lower(): metric for metric in ["Recall", "Recall2", "MRR", "Precision", "NDCG", "MAP"]
}


class TopKEvaluator(object):
    def __init__(self, config):
        self.config = config
        self.metrics = config["metrics"]
        self.topk = config["topk"]
        self.save_recom_result = config["save_recommended_topk"]
        self._check_args()

    def collect(self, interaction, scores_tensor, full=False):
        user_len_list = interaction.user_len_list
        if full:
            scores_matrix = scores_tensor.view(len(user_len_list), -1)
        else:
            scores_list = torch.split(scores_tensor, user_len_list, dim=0)
            scores_matrix = pad_sequence(scores_list, batch_first=True, padding_value=-np.inf)
        _, topk_index = torch.topk(scores_matrix, max(self.topk), dim=-1)
        return topk_index

    def evaluate(self, batch_matrix_list, eval_data, is_test=False, idx=0):
        pos_items = eval_data.get_eval_items()
        pos_len_list = eval_data.get_eval_len_list()
        topk_index = torch.cat(batch_matrix_list, dim=0).cpu().numpy()

        if self.save_recom_result and is_test:
            dataset_name = self.config["dataset"]
            model_name = self.config["model"]
            max_k = max(self.topk)
            dir_name = os.path.abspath(self.config["recommend_topk"])
            os.makedirs(dir_name, exist_ok=True)
            file_path = os.path.join(
                dir_name,
                "{}-{}-idx{}-top{}-{}.csv".format(
                    model_name, dataset_name, idx, max_k, get_local_time()
                ),
            )
            x_df = pd.DataFrame(topk_index)
            x_df.insert(0, "id", eval_data.get_eval_users())
            x_df.columns = ["id"] + ["top_" + str(i) for i in range(max_k)]
            x_df = x_df.astype(int)
            x_df.to_csv(file_path, sep="\t", index=False)

        assert len(pos_len_list) == len(topk_index)
        bool_rec_matrix = []
        for gt_items, rec_items in zip(pos_items, topk_index):
            bool_rec_matrix.append([item in gt_items for item in rec_items])
        bool_rec_matrix = np.asarray(bool_rec_matrix)

        metric_dict = {}
        result_list = self._calculate_metrics(pos_len_list, bool_rec_matrix)
        for metric, value in zip(self.metrics, result_list):
            for k in self.topk:
                metric_dict["{}@{}".format(metric, k)] = round(value[k - 1], 4)
        return metric_dict

    def _check_args(self):
        if isinstance(self.metrics, str):
            self.metrics = [self.metrics]
        elif not isinstance(self.metrics, list):
            raise TypeError("metrics must be str or list")

        for metric in self.metrics:
            if metric.lower() not in topk_metrics:
                raise ValueError(
                    "There is no user grouped topk metric named {}!".format(metric)
                )
        self.metrics = [metric.lower() for metric in self.metrics]

        if isinstance(self.topk, int):
            self.topk = [self.topk]
        elif not isinstance(self.topk, list):
            raise TypeError("The topk must be a integer, list")
        for topk in self.topk:
            if topk <= 0:
                raise ValueError("topk must be positive, got `{}`".format(topk))

    def _calculate_metrics(self, pos_len_list, topk_index):
        result_list = []
        for metric in self.metrics:
            result_list.append(metrics_dict[metric.lower()](topk_index, pos_len_list))
        return np.stack(result_list, axis=0)

    def __str__(self):
        return (
            "The TopK Evaluator Info:\n\tMetrics:[{}], TopK:[{}]".format(
                ", ".join(topk_metrics[metric.lower()] for metric in self.metrics),
                ", ".join(map(str, self.topk)),
            )
        )
