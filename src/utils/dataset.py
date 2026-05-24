# coding: utf-8

from logging import getLogger
import os

import pandas as pd


class RecDataset(object):
    def __init__(self, config, df=None):
        self.config = config
        self.logger = getLogger()

        self.dataset_name = config["dataset"]
        self.dataset_path = os.path.abspath(config["data_path"] + self.dataset_name)

        self.uid_field = self.config["USER_ID_FIELD"]
        self.iid_field = self.config["ITEM_ID_FIELD"]
        self.splitting_label = self.config["inter_splitting_label"]

        if df is not None:
            self.df = df
            self.inter_num = len(self.df)
            return

        inter_file_name = self.config["inter_file_name"]
        if not inter_file_name:
            raise ValueError(
                "Missing required config `inter_file_name` for dataset `{}`.".format(
                    self.dataset_name
                )
            )

        inter_file = os.path.join(self.dataset_path, inter_file_name)
        if not os.path.isfile(inter_file):
            raise ValueError("File {} not exist".format(inter_file))

        self.load_inter_graph(inter_file_name)
        self.item_num = int(max(self.df[self.iid_field].values)) + 1
        self.user_num = int(max(self.df[self.uid_field].values)) + 1
        self.inter_num = len(self.df)

    def load_inter_graph(self, file_name):
        inter_file = os.path.join(self.dataset_path, file_name)
        split_mode = str(self.config["split_mode"] or "x_label").lower()
        cols = [self.uid_field, self.iid_field]
        if split_mode == "x_label":
            cols.append(self.splitting_label)
        elif split_mode == "recbole_ls":
            cols.append(self.config["TIME_FIELD"])
        else:
            raise ValueError("Unsupported split_mode: {}".format(split_mode))

        self.df = pd.read_csv(inter_file, usecols=cols, sep=self.config["field_separator"])
        if not self.df.columns.isin(cols).all():
            raise ValueError("File {} lost some required columns.".format(inter_file))

    def split(self):
        split_mode = str(self.config["split_mode"] or "x_label").lower()
        dfs = []

        if split_mode == "x_label":
            for i in range(3):
                temp_df = self.df[self.df[self.splitting_label] == i].copy()
                temp_df.drop(self.splitting_label, inplace=True, axis=1)
                dfs.append(temp_df)
            if self.config["filter_out_cod_start_users"]:
                train_u = set(dfs[0][self.uid_field].values)
                for i in [1, 2]:
                    dropped_inter = pd.Series(True, index=dfs[i].index)
                    dropped_inter ^= dfs[i][self.uid_field].isin(train_u)
                    dfs[i].drop(dfs[i].index[dropped_inter], inplace=True)
        elif split_mode == "recbole_ls":
            time_field = self.config["TIME_FIELD"]
            if time_field not in self.df.columns:
                raise ValueError("TIME_FIELD {} not found for recbole_ls split".format(time_field))

            split_order = str(self.config["split_order"] or "TO").upper()
            if split_order == "TO":
                ordered_df = self.df.sort_values(by=time_field, kind="mergesort")
            elif split_order == "RO":
                ordered_df = (
                    self.df.sample(frac=1, replace=False)
                    .reset_index(drop=False)
                    .set_index("index")
                )
            else:
                raise ValueError(
                    "Unsupported split_order {} for recbole_ls split".format(split_order)
                )

            grouped_indices = {}
            for idx, uid in zip(ordered_df.index.values, ordered_df[self.uid_field].values):
                grouped_indices.setdefault(uid, []).append(idx)

            train_index, valid_index, test_index = [], [], []
            for idx_list in grouped_indices.values():
                total = len(idx_list)
                legal_leave_one_num = min(2, total - 1)
                pr = total - legal_leave_one_num
                train_index.extend(idx_list[:pr])
                for i in range(legal_leave_one_num):
                    if i == 0:
                        valid_index.append(idx_list[pr])
                    elif i == 1:
                        test_index.append(idx_list[pr])
                    pr += 1

            dfs = [
                ordered_df.loc[train_index].copy(),
                ordered_df.loc[valid_index].copy(),
                ordered_df.loc[test_index].copy(),
            ]
            for dframe in dfs:
                if self.splitting_label in dframe.columns:
                    dframe.drop(self.splitting_label, inplace=True, axis=1)
        else:
            raise ValueError("Unsupported split_mode: {}".format(split_mode))

        return [self.copy(_) for _ in dfs]

    def copy(self, new_df):
        nxt = RecDataset(self.config, new_df)
        nxt.item_num = self.item_num
        nxt.user_num = self.user_num
        return nxt

    def get_user_num(self):
        return self.user_num

    def get_item_num(self):
        return self.item_num

    def shuffle(self):
        self.df = self.df.sample(frac=1, replace=False).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [self.dataset_name]
        self.inter_num = len(self.df)
        uni_u = pd.unique(self.df[self.uid_field])
        uni_i = pd.unique(self.df[self.iid_field])
        tmp_user_num, tmp_item_num = 0, 0
        if self.uid_field:
            tmp_user_num = len(uni_u)
            avg_actions_of_users = self.inter_num / tmp_user_num
            info.extend(
                [
                    "The number of users: {}".format(tmp_user_num),
                    "Average actions of users: {}".format(avg_actions_of_users),
                ]
            )
        if self.iid_field:
            tmp_item_num = len(uni_i)
            avg_actions_of_items = self.inter_num / tmp_item_num
            info.extend(
                [
                    "The number of items: {}".format(tmp_item_num),
                    "Average actions of items: {}".format(avg_actions_of_items),
                ]
            )
        info.append("The number of inters: {}".format(self.inter_num))
        if self.uid_field and self.iid_field:
            sparsity = 1 - self.inter_num / tmp_user_num / tmp_item_num
            info.append("The sparsity of the dataset: {}%".format(sparsity * 100))
        return "\n".join(info)
