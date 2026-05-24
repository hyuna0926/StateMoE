# coding: utf-8

import os

import numpy as np
import torch
import torch.nn as nn


class AbstractRecommender(nn.Module):
    def pre_epoch_processing(self):
        pass

    def post_epoch_processing(self):
        pass

    def calculate_loss(self, interaction):
        raise NotImplementedError

    def predict(self, interaction):
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        raise NotImplementedError

    def __str__(self):
        params = sum(np.prod(p.size()) for p in self.parameters())
        return super().__str__() + "\nTrainable parameters: {}".format(params)


class GeneralRecommender(AbstractRecommender):
    def __init__(self, config, dataloader):
        super().__init__()

        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID
        self.n_users = dataloader.dataset.get_user_num()
        self.n_items = dataloader.dataset.get_item_num()

        self.batch_size = config["train_batch_size"]
        self.device = config["device"]

        self.v_feat, self.t_feat = None, None
        if not config["end2end"] and config["is_multimodal_model"]:
            dataset_path = os.path.abspath(config["data_path"] + config["dataset"])
            v_feat_path = os.path.join(dataset_path, config["vision_feature_file"])
            t_feat_path = os.path.join(dataset_path, config["text_feature_file"])
            if os.path.isfile(v_feat_path):
                self.v_feat = torch.from_numpy(np.load(v_feat_path, allow_pickle=True)).float().to(
                    self.device
                )
            if os.path.isfile(t_feat_path):
                self.t_feat = torch.from_numpy(np.load(t_feat_path, allow_pickle=True)).float().to(
                    self.device
                )

            assert self.v_feat is not None or self.t_feat is not None, "Features all NONE"
