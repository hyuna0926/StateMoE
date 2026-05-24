# coding: utf-8

import math
import random
from logging import getLogger

import numpy as np
import torch
from scipy.sparse import coo_matrix


class AbstractDataLoader(object):
    def __init__(self, config, dataset, additional_dataset=None, batch_size=1, neg_sampling=False, shuffle=False):
        self.config = config
        self.logger = getLogger()
        self.dataset = dataset
        self.dataset_bk = self.dataset.copy(self.dataset.df)
        self.additional_dataset = additional_dataset
        self.batch_size = batch_size
        self.step = batch_size
        self.shuffle = shuffle
        self.neg_sampling = neg_sampling
        self.device = config["device"]
        self.sparsity = 1 - self.dataset.inter_num / self.dataset.user_num / self.dataset.item_num
        self.pr = 0
        self.inter_pr = 0

    def pretrain_setup(self):
        pass

    def data_preprocess(self):
        pass

    def __len__(self):
        return math.ceil(self.pr_end / self.step)

    def __iter__(self):
        if self.shuffle:
            self._shuffle()
        return self

    def __next__(self):
        if self.pr >= self.pr_end:
            self.pr = 0
            self.inter_pr = 0
            raise StopIteration()
        return self._next_batch_data()

    @property
    def pr_end(self):
        raise NotImplementedError

    def _shuffle(self):
        raise NotImplementedError

    def _next_batch_data(self):
        raise NotImplementedError


class TrainDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset, batch_size=1, shuffle=False):
        super().__init__(
            config,
            dataset,
            additional_dataset=None,
            batch_size=batch_size,
            neg_sampling=True,
            shuffle=shuffle,
        )

        self.history_items_per_u = {}
        self.all_items = self.dataset.df[self.dataset.iid_field].unique().tolist()
        self.all_uids = self.dataset.df[self.dataset.uid_field].unique()
        self.all_items_set = set(self.all_items)
        self.all_users_set = set(self.all_uids)
        self.use_full_sampling = config["use_full_sampling"]

        if config["use_neg_sampling"]:
            self.sample_func = (
                self._get_full_uids_sample if self.use_full_sampling else self._get_neg_sample
            )
        else:
            self.sample_func = self._get_non_neg_sample

        self._get_history_items_u()
        self.neighborhood_loss_required = config["use_neighborhood_loss"]
        if self.neighborhood_loss_required:
            self.history_users_per_i = {}
            self._get_history_users_i()
            self.user_user_dict = self._get_my_neighbors(self.config["USER_ID_FIELD"])
            self.item_item_dict = self._get_my_neighbors(self.config["ITEM_ID_FIELD"])

    def pretrain_setup(self):
        if self.shuffle:
            self.dataset = self.dataset_bk.copy(self.dataset_bk.df)
        self.all_items.sort()
        if self.use_full_sampling:
            self.all_uids.sort()
        random.shuffle(self.all_items)

    def inter_matrix(self, form="coo", value_field=None):
        if not self.dataset.uid_field or not self.dataset.iid_field:
            raise ValueError("dataset doesn't exist uid/iid, thus can not converted to sparse matrix")
        return self._create_sparse_matrix(
            self.dataset.df, self.dataset.uid_field, self.dataset.iid_field, form, value_field
        )

    def _create_sparse_matrix(self, df_feat, source_field, target_field, form="coo", value_field=None):
        src = df_feat[source_field].values
        tgt = df_feat[target_field].values
        if value_field is None:
            data = np.ones(len(df_feat))
        else:
            if value_field not in df_feat.columns:
                raise ValueError(
                    "value_field [{}] should be one of df_feat's features.".format(value_field)
                )
            data = df_feat[value_field].values
        mat = coo_matrix((data, (src, tgt)), shape=(self.dataset.user_num, self.dataset.item_num))

        if form == "coo":
            return mat
        if form == "csr":
            return mat.tocsr()
        raise NotImplementedError("sparse matrix format [{}] has not been implemented.".format(form))

    @property
    def pr_end(self):
        return len(self.all_uids) if self.use_full_sampling else len(self.dataset)

    def _shuffle(self):
        self.dataset.shuffle()
        if self.use_full_sampling:
            np.random.shuffle(self.all_uids)

    def _next_batch_data(self):
        return self.sample_func()

    def _get_neg_sample(self):
        cur_data = self.dataset[self.pr : self.pr + self.step]
        self.pr += self.step
        user_tensor = torch.tensor(cur_data[self.config["USER_ID_FIELD"]].values).long().to(self.device)
        item_tensor = torch.tensor(cur_data[self.config["ITEM_ID_FIELD"]].values).long().to(self.device)
        batch_tensor = torch.cat((user_tensor.unsqueeze(0), item_tensor.unsqueeze(0)))
        u_ids = cur_data[self.config["USER_ID_FIELD"]]
        neg_ids = self._sample_neg_ids(u_ids).to(self.device)

        if self.neighborhood_loss_required:
            i_ids = cur_data[self.config["ITEM_ID_FIELD"]]
            pos_neighbors, neg_neighbors = self._get_neighborhood_samples(
                i_ids, self.config["ITEM_ID_FIELD"]
            )
            pos_neighbors = pos_neighbors.to(self.device)
            neg_neighbors = neg_neighbors.to(self.device)
            batch_tensor = torch.cat(
                (
                    batch_tensor,
                    neg_ids.unsqueeze(0),
                    pos_neighbors.unsqueeze(0),
                    neg_neighbors.unsqueeze(0),
                )
            )
        else:
            batch_tensor = torch.cat((batch_tensor, neg_ids.unsqueeze(0)))

        return batch_tensor

    def _get_non_neg_sample(self):
        cur_data = self.dataset[self.pr : self.pr + self.step]
        self.pr += self.step
        user_tensor = torch.tensor(cur_data[self.config["USER_ID_FIELD"]].values).long().to(self.device)
        item_tensor = torch.tensor(cur_data[self.config["ITEM_ID_FIELD"]].values).long().to(self.device)
        return torch.cat((user_tensor.unsqueeze(0), item_tensor.unsqueeze(0)))

    def _get_full_uids_sample(self):
        user_tensor = torch.tensor(self.all_uids[self.pr : self.pr + self.step]).long().to(self.device)
        self.pr += self.step
        return user_tensor

    def _sample_neg_ids(self, u_ids):
        neg_ids = []
        for u in u_ids:
            iid = self._random()
            while iid in self.history_items_per_u[u]:
                iid = self._random()
            neg_ids.append(iid)
        return torch.tensor(neg_ids).long()

    def _get_my_neighbors(self, id_str):
        ret_dict = {}
        a2b_dict = (
            self.history_items_per_u
            if id_str == self.config["USER_ID_FIELD"]
            else self.history_users_per_i
        )
        b2a_dict = (
            self.history_users_per_i
            if id_str == self.config["USER_ID_FIELD"]
            else self.history_items_per_u
        )
        for i, j in a2b_dict.items():
            k = set()
            for m in j:
                k |= b2a_dict.get(m, set()).copy()
            k.discard(i)
            ret_dict[i] = k
        return ret_dict

    def _get_neighborhood_samples(self, ids, id_str):
        a2a_dict = self.user_user_dict if id_str == self.config["USER_ID_FIELD"] else self.item_item_dict
        all_set = self.all_users_set if id_str == self.config["USER_ID_FIELD"] else self.all_items_set
        pos_ids, neg_ids = [], []
        for i in ids:
            pos_ids_my = a2a_dict[i]
            if len(pos_ids_my) <= 0 or len(pos_ids_my) / len(all_set) > 0.8:
                pos_ids.append(0)
                neg_ids.append(0)
                continue
            pos_id = random.sample(list(pos_ids_my), 1)[0]
            pos_ids.append(pos_id)
            neg_id = random.sample(list(all_set), 1)[0]
            while neg_id in pos_ids_my:
                neg_id = random.sample(list(all_set), 1)[0]
            neg_ids.append(neg_id)
        return torch.tensor(pos_ids).long(), torch.tensor(neg_ids).long()

    def _random(self):
        return random.sample(self.all_items, 1)[0]

    def _get_history_items_u(self):
        uid_freq = self.dataset.df.groupby(self.dataset.uid_field)[self.dataset.iid_field]
        for u, u_ls in uid_freq:
            self.history_items_per_u[u] = set(u_ls.values)
        return self.history_items_per_u

    def _get_history_users_i(self):
        iid_freq = self.dataset.df.groupby(self.dataset.iid_field)[self.dataset.uid_field]
        for i, u_ls in iid_freq:
            self.history_users_per_i[i] = set(u_ls.values)
        return self.history_users_per_i


class EvalDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset, additional_dataset=None, batch_size=1, shuffle=False):
        super().__init__(
            config,
            dataset,
            additional_dataset=additional_dataset,
            batch_size=batch_size,
            neg_sampling=False,
            shuffle=shuffle,
        )

        if additional_dataset is None:
            raise ValueError("Training datasets is nan")
        self.eval_items_per_u = []
        self.eval_len_list = []
        self.train_pos_len_list = []

        self.eval_u = self.dataset.df[self.dataset.uid_field].unique()
        self.pos_items_per_u = self._get_pos_items_per_u(self.eval_u).to(self.device)
        self._get_eval_items_per_u(self.eval_u)
        self.eval_u = torch.tensor(self.eval_u).long().to(self.device)

    @property
    def pr_end(self):
        return self.eval_u.shape[0]

    def _shuffle(self):
        self.dataset.shuffle()

    def _next_batch_data(self):
        inter_cnt = sum(self.train_pos_len_list[self.pr : self.pr + self.step])
        batch_users = self.eval_u[self.pr : self.pr + self.step]
        batch_mask_matrix = self.pos_items_per_u[:, self.inter_pr : self.inter_pr + inter_cnt].clone()
        batch_mask_matrix[0] -= self.pr
        self.inter_pr += inter_cnt
        self.pr += self.step
        return [batch_users, batch_mask_matrix]

    def _get_pos_items_per_u(self, eval_users):
        uid_freq = self.additional_dataset.df.groupby(self.additional_dataset.uid_field)[
            self.additional_dataset.iid_field
        ]
        u_ids = []
        i_ids = []
        missing_train_users = []
        for i, u in enumerate(eval_users):
            if u in uid_freq.groups:
                u_ls = uid_freq.get_group(u).values
            else:
                # Keep cold-start eval users when the split explicitly contains them.
                # They contribute an empty train-history mask for evaluation.
                u_ls = np.empty(0, dtype=np.int64)
                missing_train_users.append(int(u))
            i_len = len(u_ls)
            self.train_pos_len_list.append(i_len)
            u_ids.extend([i] * i_len)
            i_ids.extend(u_ls)
        if missing_train_users:
            preview = ", ".join(map(str, missing_train_users[:10]))
            self.logger.warning(
                "Eval split contains %d user(s) with no train interactions; "
                "keeping them with empty history masks. Sample user ids: %s",
                len(missing_train_users),
                preview,
            )
        return torch.tensor([u_ids, i_ids]).long()

    def _get_eval_items_per_u(self, eval_users):
        uid_freq = self.dataset.df.groupby(self.dataset.uid_field)[self.dataset.iid_field]
        for u in eval_users:
            u_ls = uid_freq.get_group(u).values
            self.eval_len_list.append(len(u_ls))
            self.eval_items_per_u.append(u_ls)
        self.eval_len_list = np.asarray(self.eval_len_list)

    def get_eval_items(self):
        return self.eval_items_per_u

    def get_eval_len_list(self):
        return self.eval_len_list

    def get_eval_users(self):
        return self.eval_u.cpu()
