# coding: utf-8

import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import (
    build_chunked_sparse_knn_graph,
    build_knn_neighbourhood,
    build_sim,
    compute_normalized_laplacian,
)


class ZeroMIEstimator(nn.Module):
    def forward(self, *args, **kwargs):
        if args and torch.is_tensor(args[0]):
            return args[0].new_tensor(0.0)
        return torch.tensor(0.0)

    def learning_loss(self, *args, **kwargs):
        if args and torch.is_tensor(args[0]):
            return args[0].new_tensor(0.0)
        return torch.tensor(0.0)


class QualityEstimator(nn.Module):
    def __init__(self, input_dim, hidden_dim, state_emb_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.state_proj = nn.Sequential(
            nn.Linear(3, state_emb_dim),
            nn.ReLU(),
        )
        self.a_noisy = nn.Parameter(torch.zeros(1))
        self.delta = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        logits = self.classifier(x)
        prob = F.softmax(logits, dim=-1)

        p_clean = prob[:, 0:1]
        p_noisy = prob[:, 1:2]
        p_missing = prob[:, 2:3]
        state_emb = self.state_proj(prob)

        alpha_noisy = F.softplus(self.a_noisy)
        alpha_missing = alpha_noisy + F.softplus(self.delta)
        severity = torch.sigmoid(
            alpha_noisy * p_noisy + alpha_missing * p_missing + self.b
        )
        reliability = 1.0 - severity
        uncertainty = -(
            prob.clamp_min(1e-8) * prob.clamp_min(1e-8).log()
        ).sum(dim=-1, keepdim=True) / np.log(3.0)

        return {
            "logits": logits,
            "prob": prob,
            "p_clean": p_clean,
            "p_noisy": p_noisy,
            "p_missing": p_missing,
            "state_emb": state_emb,
            "severity": severity,
            "reliability": reliability,
            "uncertainty": uncertainty,
        }


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5

    def forward(self, query, kv):
        q = self.w_q(query).unsqueeze(1)
        k = self.w_k(kv)
        v = self.w_v(kv)
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        return torch.bmm(attn, v).squeeze(1)


class KeepExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, target_emb):
        return target_emb


class NoisyRepairExpert(nn.Module):
    def __init__(self, dim, state_emb_dim):
        super().__init__()
        self.cross_attn = CrossAttention(dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(state_emb_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        target_emb,
        other_emb,
        item_emb,
        target_hist,
        id_hist,
        target_state_emb,
        other_state_emb,
        target_severity,
        other_severity,
    ):
        gate_input = torch.cat([target_state_emb, other_state_emb], dim=-1)
        rho = torch.sigmoid(self.gate_mlp(gate_input))
        kv = torch.stack([other_emb, item_emb, target_hist, id_hist], dim=1)
        repair = self.cross_attn(target_emb, kv)
        return self.norm(rho * target_emb + (1.0 - rho) * repair)


class MissingRepairExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cross_attn = CrossAttention(dim)
        self.query_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.base_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.blend_gate = nn.Sequential(
            nn.Linear(dim * 2, 1),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, target_emb, other_emb, item_emb, target_hist, id_hist):
        query = self.query_fusion(torch.cat([other_emb, item_emb], dim=-1))
        kv = torch.stack([target_emb, other_emb, item_emb, id_hist], dim=1)
        repair = self.cross_attn(query, kv)
        base = self.base_fusion(torch.cat([id_hist, other_emb], dim=-1))
        alpha = self.blend_gate(torch.cat([repair, base], dim=-1))
        return self.norm(alpha * base + (1.0 - alpha) * repair)


class StateMoE(GeneralRecommender):
    """Flattened DGMRec_QE_NEW5 model with no baseline/DGMRec model imports."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.embedding_dim = int(config["embedding_size"])
        self.n_ui_layers = int(config["n_ui_layers"])
        self.n_mm_layers = int(config["n_mm_layers"])
        self.knn_k = int(config["knn_k"])
        self.graph_build_device = config["graph_build_device"]
        self.graph_build_chunk_size = int(config["graph_build_chunk_size"] or 1024)

        self.quality_hidden_size = int(
            config["quality_hidden_size"]
            if "quality_hidden_size" in config
            else (config["qe_hidden_dim"] if "qe_hidden_dim" in config else 256)
        )
        self.state_emb_size = (
            int(config["state_emb_size"]) if "state_emb_size" in config else 64
        )
        self.quality_ckpt_dir = (
            config["quality_ckpt_dir"] if "quality_ckpt_dir" in config else None
        )
        self.freeze_qe = bool(config["freeze_qe"]) if "freeze_qe" in config else False
        self.ablation_force_uniform_qe = (
            bool(config["ablation_force_uniform_qe"])
            if "ablation_force_uniform_qe" in config
            else False
        )
        self.ablation_disable_moe_repair = (
            bool(config["ablation_disable_moe_repair"])
            if "ablation_disable_moe_repair" in config
            else False
        )
        self.ablation_use_full_state_gate = (
            bool(config["ablation_use_full_state_gate"])
            if "ablation_use_full_state_gate" in config
            else self.ablation_force_uniform_qe
        )
        self.repair_clean_only_context = (
            bool(config["repair_clean_only_context"])
            if "repair_clean_only_context" in config
            else False
        )
        self.ablation_all_aux_context = (
            bool(config["ablation_all_aux_context"])
            if "ablation_all_aux_context" in config
            else False
        )

        self.repair_bottleneck = (
            int(config["repair_bottleneck"]) if "repair_bottleneck" in config else 256
        )
        self.repair_hist_alpha = (
            float(config["repair_hist_alpha"]) if "repair_hist_alpha" in config else 0.5
        )
        self.loss_weight_repair = (
            float(config["loss_weight_repair"]) if "loss_weight_repair" in config else 0.15
        )
        self.loss_weight_mm = (
            float(config["loss_weight_mm"]) if "loss_weight_mm" in config else 0.01
        )
        self.loss_weight_align = (
            float(config["loss_weight_align"]) if "loss_weight_align" in config else 0.01
        )
        self.use_router_prior = (
            bool(config["use_router_prior"]) if "use_router_prior" in config else False
        )
        self.loss_weight_router = (
            float(config["loss_weight_router"]) if "loss_weight_router" in config else 0.0
        )
        self.log_qe_prob = (
            bool(config["log_qe_prob"]) if "log_qe_prob" in config else True
        )
        self.log_qe_state_stats = (
            bool(config["log_qe_state_stats"]) if "log_qe_state_stats" in config else True
        )
        self.log_repair_feature_stats = (
            bool(config["log_repair_feature_stats"])
            if "log_repair_feature_stats" in config
            else True
        )
        self.diag_log_interval = (
            int(config["diag_log_interval"]) if "diag_log_interval" in config else 20
        )
        self.repair_clean_bypass = (
            bool(config["repair_clean_bypass"])
            if "repair_clean_bypass" in config
            else True
        )
        self.repair_clean_bypass_power = (
            float(config["repair_clean_bypass_power"])
            if "repair_clean_bypass_power" in config
            else 1.0
        )
        self.repair_img_cosine_weight = (
            float(config["repair_img_cosine_weight"])
            if "repair_img_cosine_weight" in config
            else 0.01
        )
        self.repair_txt_cosine_weight = (
            float(config["repair_txt_cosine_weight"])
            if "repair_txt_cosine_weight" in config
            else 0.02
        )
        self.repair_clean_keep_weight = (
            float(config["repair_clean_keep_weight"])
            if "repair_clean_keep_weight" in config
            else 0.1
        )
        self.mm_adj_refresh_interval = (
            int(config["mm_adj_refresh_interval"])
            if "mm_adj_refresh_interval" in config
            else 5
        )
        self.mm_adj_refresh_skip_qe_threshold = 0.05
        self.img_hard_clean_corrupt_threshold = (
            float(config["img_hard_clean_corrupt_threshold"])
            if "img_hard_clean_corrupt_threshold" in config
            else 0.05
        )
        self.txt_hard_clean_corrupt_threshold = (
            float(config["txt_hard_clean_corrupt_threshold"])
            if "txt_hard_clean_corrupt_threshold" in config
            else 0.10
        )
        self.img_clean_route_margin = (
            float(config["img_clean_route_margin"])
            if "img_clean_route_margin" in config
            else 0.05
        )
        self.txt_clean_route_margin = (
            float(config["txt_clean_route_margin"])
            if "txt_clean_route_margin" in config
            else 0.05
        )
        self.repair_img_clean_keep_weight = (
            float(config["repair_img_clean_keep_weight"])
            if "repair_img_clean_keep_weight" in config
            else 0.10
        )
        self.repair_txt_clean_keep_weight = (
            float(config["repair_txt_clean_keep_weight"])
            if "repair_txt_clean_keep_weight" in config
            else 0.10
        )
        self.repair_txt_residual_alpha = (
            float(config["repair_txt_residual_alpha"])
            if "repair_txt_residual_alpha" in config
            else 0.75
        )
        self.txt_clean_adapter_scale = (
            float(config["txt_clean_adapter_scale"])
            if "txt_clean_adapter_scale" in config
            else 0.35
        )

        self.infoNCETemp = float(config["infoNCETemp"])
        self.alignBMTemp = float(config["alignBMTemp"])
        self.alignUITemp = float(config["alignUITemp"])

        if self.v_feat is None or self.t_feat is None:
            raise ValueError("StateMoE expects both image and text features.")

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.n_nodes = self.n_users + self.n_items
        self.adj = self.scipy_matrix_to_sparse_tenser(
            self.interaction_matrix, torch.Size((self.n_users, self.n_items))
        )
        self.num_inters, self.norm_adj = self.get_norm_adj_mat()
        self.norm_adj = self.norm_adj.to(self.device)
        self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).to(
            self.device
        )

        self.all_items = np.arange(self.n_items)
        self.complete_items = np.arange(self.n_items)
        self.missing_modal = config["missing_modal"]
        if self.missing_modal:
            self.preprocess_missing_modal(config)
        else:
            self.missing_items = {"all": np.array([]), "t": np.array([]), "v": np.array([])}
            self.missing_items_t = np.array([], dtype=np.int64)
            self.missing_items_v = np.array([], dtype=np.int64)

        self.image_feat_dim = self.v_feat.shape[1]
        self.text_feat_dim = self.t_feat.shape[1]

        self.register_buffer(
            "_img_clean_target",
            self._load_optional_repair_target(config, "repair_target_vision_feature_file"),
        )
        self.register_buffer(
            "_txt_clean_target",
            self._load_optional_repair_target(config, "repair_target_text_feature_file"),
        )

        self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False).to(
            self.device
        )
        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False).to(
            self.device
        )
        self.image_adj = self._build_mm_adj(
            self.image_embedding.weight.detach(),
            self.missing_items_v if self.missing_modal else None,
        )
        self.text_adj = self._build_mm_adj(
            self.text_embedding.weight.detach(),
            self.missing_items_t if self.missing_modal else None,
        )

        self.register_buffer(
            "_img_raw_snapshot", self.image_embedding.weight.detach().clone()
        )
        self.register_buffer(
            "_txt_raw_snapshot", self.text_embedding.weight.detach().clone()
        )
        self._gt_text_state, self._gt_image_state = self._load_gt_states(config)

        self.image_encoder = nn.Linear(self.image_feat_dim, self.embedding_dim).to(self.device)
        self.text_encoder = nn.Linear(self.text_feat_dim, self.embedding_dim).to(self.device)
        self.shared_encoder = nn.Linear(self.embedding_dim, self.embedding_dim).to(
            self.device
        )
        self.image_preference_ = nn.Linear(
            self.embedding_dim, self.embedding_dim, bias=False
        )
        self.text_preference_ = nn.Linear(
            self.embedding_dim, self.embedding_dim, bias=False
        )
        self.image_preference_.to(self.device)
        self.text_preference_.to(self.device)

        self.image_repair_proj = nn.Sequential(
            nn.Linear(self.image_feat_dim, self.repair_bottleneck),
            nn.ReLU(),
            nn.LayerNorm(self.repair_bottleneck),
        ).to(self.device)
        self.text_repair_proj = nn.Sequential(
            nn.Linear(self.text_feat_dim, self.repair_bottleneck),
            nn.ReLU(),
            nn.LayerNorm(self.repair_bottleneck),
        ).to(self.device)
        self.item_repair_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, self.repair_bottleneck),
            nn.ReLU(),
            nn.LayerNorm(self.repair_bottleneck),
        ).to(self.device)
        self.image_repair_decode = nn.Linear(
            self.repair_bottleneck, self.image_feat_dim
        ).to(self.device)
        self.text_repair_decode = nn.Linear(
            self.repair_bottleneck, self.text_feat_dim
        ).to(self.device)

        self.image_encoder.apply(self.init_weight)
        self.text_encoder.apply(self.init_weight)
        self.shared_encoder.apply(self.init_weight)
        self.image_preference_.apply(self.init_weight)
        self.text_preference_.apply(self.init_weight)
        self.image_repair_proj.apply(self.init_weight)
        self.text_repair_proj.apply(self.init_weight)
        self.item_repair_proj.apply(self.init_weight)
        self.image_repair_decode.apply(self.init_weight)
        self.text_repair_decode.apply(self.init_weight)

        self.img_qe = self._build_raw_qe("image", self.image_feat_dim)
        self.txt_qe = self._build_raw_qe("text", self.text_feat_dim)
        if self.freeze_qe:
            for qe in (self.img_qe, self.txt_qe):
                qe.eval()
                for param in qe.parameters():
                    param.requires_grad_(False)

        qe_state_dim = self._infer_qe_state_dim(self.img_qe)
        if qe_state_dim != self._infer_qe_state_dim(self.txt_qe):
            raise ValueError("Image/Text QE state embedding dimensions must match.")

        self.img_keep_expert = KeepExpert(self.repair_bottleneck).to(self.device)
        self.img_noisy_expert = NoisyRepairExpert(
            self.repair_bottleneck, qe_state_dim
        ).to(self.device)
        self.img_missing_expert = MissingRepairExpert(self.repair_bottleneck).to(
            self.device
        )
        self.txt_keep_expert = KeepExpert(self.repair_bottleneck).to(self.device)
        self.txt_noisy_expert = NoisyRepairExpert(
            self.repair_bottleneck, qe_state_dim
        ).to(self.device)
        self.txt_missing_expert = MissingRepairExpert(self.repair_bottleneck).to(
            self.device
        )

        self.act_g = nn.Tanh()
        self._last_qe_ctx = None
        self._qe_log_epoch = 0
        self._cached_qe_ctx = None
        self._eval_repr_cache = None
        self._last_txt_repair_core_feat = None

        self.v_feat = None
        self.t_feat = None

    def init_weight(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def init_mi_estimator(self):
        self.item_image_estimator = ZeroMIEstimator().to(self.device)
        self.user_image_estimator = ZeroMIEstimator().to(self.device)
        self.item_text_estimator = ZeroMIEstimator().to(self.device)
        self.user_text_estimator = ZeroMIEstimator().to(self.device)

    def _load_optional_repair_target(self, config, key):
        file_name = config[key] if key in config else None
        if not file_name:
            return None
        dataset_path = os.path.abspath(config["data_path"] + config["dataset"])
        file_path = (
            file_name if os.path.isabs(file_name) else os.path.join(dataset_path, file_name)
        )
        if not os.path.isfile(file_path):
            raise FileNotFoundError("Repair target feature file not found: {}".format(file_path))
        return torch.from_numpy(np.load(file_path, allow_pickle=True)).float().to(self.device)

    def _infer_gt_state_file(self, config):
        dataset_path = os.path.abspath(config["data_path"] + config["dataset"])
        gt_state_file = config["gt_state_file"] if "gt_state_file" in config else None
        if gt_state_file:
            path = (
                gt_state_file
                if os.path.isabs(gt_state_file)
                else os.path.join(dataset_path, gt_state_file)
            )
            return path if os.path.isfile(path) else None

        for key in ("vision_feature_file", "text_feature_file"):
            feat_file = config[key] if key in config else None
            if not feat_file:
                continue
            dirname = os.path.dirname(feat_file)
            basename = os.path.basename(feat_file)
            candidate = None
            if basename.startswith("image_feat_"):
                candidate = basename.replace("image_feat_", "gt_state_", 1)
            elif basename.startswith("text_feat_"):
                candidate = basename.replace("text_feat_", "gt_state_", 1)
            if candidate is None:
                continue
            gt_path = os.path.join(dataset_path, dirname, candidate)
            if os.path.isfile(gt_path):
                return gt_path
        return None

    def _load_gt_states(self, config):
        gt_path = self._infer_gt_state_file(config)
        if gt_path and os.path.isfile(gt_path):
            payload = np.load(gt_path, allow_pickle=True).item()
            txt_state = torch.as_tensor(payload["t"], device=self.device, dtype=torch.long)
            img_state = torch.as_tensor(payload["v"], device=self.device, dtype=torch.long)
            return txt_state, img_state

        txt_state = torch.zeros(self.n_items, device=self.device, dtype=torch.long)
        img_state = torch.zeros(self.n_items, device=self.device, dtype=torch.long)
        if self.missing_modal:
            if len(self.missing_items_t) > 0:
                txt_state[
                    torch.as_tensor(self.missing_items_t, device=self.device, dtype=torch.long)
                ] = 2
            if len(self.missing_items_v) > 0:
                img_state[
                    torch.as_tensor(self.missing_items_v, device=self.device, dtype=torch.long)
                ] = 2
        return txt_state, img_state

    def _build_raw_qe(self, modality_name, input_dim):
        if self.quality_ckpt_dir:
            ckpt_path = os.path.join(
                self.quality_ckpt_dir, f"{modality_name}_quality_estimator_best.pt"
            )
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    "QE checkpoint not found for {}: {}".format(modality_name, ckpt_path)
                )
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            classifier_weight = state_dict.get("classifier.0.weight")
            state_proj_weight = state_dict.get("state_proj.0.weight")
            if classifier_weight is None or state_proj_weight is None:
                raise KeyError(
                    "Incompatible QE checkpoint for {}: missing classifier/state_proj".format(
                        modality_name
                    )
                )
            if classifier_weight.shape[1] != input_dim:
                raise ValueError(
                    "QE input dim mismatch for {}: ckpt expects {}, model has {}".format(
                        modality_name, classifier_weight.shape[1], input_dim
                    )
                )
            qe = QualityEstimator(
                input_dim=input_dim,
                hidden_dim=classifier_weight.shape[0],
                state_emb_dim=state_proj_weight.shape[0],
            ).to(self.device)
            qe.load_state_dict(state_dict)
            return qe

        return QualityEstimator(
            input_dim=input_dim,
            hidden_dim=self.quality_hidden_size,
            state_emb_dim=self.state_emb_size,
        ).to(self.device)

    def _infer_qe_state_dim(self, qe):
        first = qe.state_proj[0]
        if isinstance(first, nn.Linear):
            return int(first.out_features)
        raise ValueError("Unable to infer QE state embedding dimension.")

    def _build_mm_adj(self, feat_weight, missing_indices=None):
        if self.graph_build_device:
            adj = build_chunked_sparse_knn_graph(
                feat_weight,
                topk=self.knn_k,
                chunk_size=self.graph_build_chunk_size,
                build_device=self.graph_build_device,
                mask_indices=missing_indices,
            )
            return adj.to(self.device)

        adj = build_sim(feat_weight)
        adj = build_knn_neighbourhood(adj, topk=self.knn_k)
        if missing_indices is not None and len(missing_indices) > 0:
            adj[missing_indices, :] = 0.0
            adj[:, missing_indices] = 0.0
            adj[missing_indices, missing_indices] = 1.0
        return compute_normalized_laplacian(adj).to_sparse_coo().to(self.device)

    def preprocess_missing_modal(self, config):
        dataset_path = os.path.abspath(config["data_path"] + config["dataset"])
        self.missing_ratio = config["missing_ratio"]
        self.missing_items = np.load(
            os.path.join(dataset_path, f"missing_items_{self.missing_ratio}.npy"),
            allow_pickle=True,
        ).item()

        self.missing_items_t = np.concatenate(
            (self.missing_items["all"], self.missing_items["t"])
        )
        self.missing_items_v = np.concatenate(
            (self.missing_items["all"], self.missing_items["v"])
        )
        self.complete_items = np.setdiff1d(
            np.arange(self.n_items), np.union1d(self.missing_items_v, self.missing_items_t)
        )

        non_missing_item_t = np.setdiff1d(self.all_items, self.missing_items_t)
        non_missing_item_v = np.setdiff1d(self.all_items, self.missing_items_v)
        image_mean = self.v_feat[non_missing_item_v].mean(dim=0)
        text_mean = self.t_feat[non_missing_item_t].mean(dim=0)
        self.v_feat[self.missing_items_v] = image_mean
        self.t_feat[self.missing_items_t] = text_mean

    def scipy_matrix_to_sparse_tenser(self, matrix, shape):
        indices = torch.LongTensor(np.array([matrix.row, matrix.col]))
        data = torch.FloatTensor(matrix.data)
        return torch.sparse_coo_tensor(indices, data, shape).to(self.device)

    def get_norm_adj_mat(self):
        adj = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_m = self.interaction_matrix
        inter_m_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_m.row, inter_m.col + self.n_users), [1] * inter_m.nnz)
        )
        data_dict.update(
            dict(zip(zip(inter_m_t.row + self.n_users, inter_m_t.col), [1] * inter_m_t.nnz))
        )
        for key, value in data_dict.items():
            adj[key] = value

        sum_arr = (adj > 0).sum(axis=1)
        diag = np.array(sum_arr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        d_mat = sp.diags(diag)
        lap = sp.coo_matrix(d_mat * adj * d_mat)
        indices = torch.LongTensor(np.array([lap.row, lap.col]))
        data = torch.FloatTensor(lap.data)
        return sum_arr, torch.sparse_coo_tensor(
            indices, data, torch.Size((self.n_nodes, self.n_nodes))
        )

    def cge(self, user_emb, item_emb, adj):
        ego_embeddings = torch.cat((user_emb, item_emb), dim=0)
        all_embeddings = [ego_embeddings]
        for _ in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_embeddings, item_embedding = torch.split(
            all_embeddings, [self.n_users, self.n_items], dim=0
        )
        return user_embeddings, item_embedding

    def _invalidate_qe_cache(self):
        self._cached_qe_ctx = None

    def _invalidate_eval_cache(self):
        self._eval_repr_cache = None

    def _run_qe(self, qe, snapshot):
        if self.freeze_qe:
            with torch.no_grad():
                return qe(snapshot)
        return qe(snapshot)

    def _force_uniform_qe_output(self, q, qe):
        prob = torch.full_like(q["prob"], 1.0 / 3.0)
        if self.freeze_qe:
            with torch.no_grad():
                state_emb = qe.state_proj(prob)
        else:
            state_emb = qe.state_proj(prob)

        p_clean = prob[:, 0:1]
        p_noisy = prob[:, 1:2]
        p_missing = prob[:, 2:3]
        severity = p_noisy + p_missing
        return {
            "logits": torch.zeros_like(q["logits"]),
            "prob": prob,
            "p_clean": p_clean,
            "p_noisy": p_noisy,
            "p_missing": p_missing,
            "state_emb": state_emb,
            "severity": severity,
            "reliability": 1.0 - severity,
            "uncertainty": torch.ones_like(q["uncertainty"]),
        }

    def _compute_qe_ctx(self):
        img_q = self._run_qe(self.img_qe, self._img_raw_snapshot)
        txt_q = self._run_qe(self.txt_qe, self._txt_raw_snapshot)
        if self.ablation_force_uniform_qe:
            img_q = self._force_uniform_qe_output(img_q, self.img_qe)
            txt_q = self._force_uniform_qe_output(txt_q, self.txt_qe)
        else:
            img_q = self._apply_missing_override(img_q, self.missing_items_v, self.img_qe)
            txt_q = self._apply_missing_override(txt_q, self.missing_items_t, self.txt_qe)
        return {"img_q": img_q, "txt_q": txt_q}

    def _get_qe_ctx(self):
        cacheable = self.freeze_qe or (not self.training)
        if cacheable and self._cached_qe_ctx is not None:
            return self._cached_qe_ctx

        qe_ctx = self._compute_qe_ctx()
        if cacheable:
            self._cached_qe_ctx = qe_ctx
        return qe_ctx

    def _apply_missing_override(self, q, missing_idx, qe):
        if len(missing_idx) == 0:
            return q
        idx = torch.from_numpy(missing_idx).long().to(self.device)
        out = {key: value.clone() for key, value in q.items()}
        missing_prob = torch.zeros(idx.numel(), 3, device=self.device)
        missing_prob[:, 2] = 1.0

        if self.freeze_qe:
            with torch.no_grad():
                missing_state_emb = qe.state_proj(missing_prob)
        else:
            missing_state_emb = qe.state_proj(missing_prob)

        out["prob"][idx] = missing_prob
        out["p_clean"][idx] = missing_prob[:, 0:1]
        out["p_noisy"][idx] = missing_prob[:, 1:2]
        out["p_missing"][idx] = missing_prob[:, 2:3]
        out["logits"][idx] = torch.log(missing_prob + 1e-8)
        out["state_emb"][idx] = missing_state_emb
        out["severity"][idx] = 1.0
        out["reliability"][idx] = 0.0
        out["uncertainty"][idx] = 0.0
        return out

    def _qe_reliability(self, q):
        if self.ablation_force_uniform_qe:
            return torch.full_like(q["p_clean"], 1.0 / 3.0)
        return (q["p_clean"] + self.repair_hist_alpha * q["p_noisy"]).clamp(0.0, 1.0)

    def _context_reliability(self, q, modality):
        reliability = self._qe_reliability(q)
        if not self.repair_clean_only_context or self.ablation_all_aux_context:
            return reliability
        clean_mask = self._get_clean_route_mask(q, modality).to(reliability.dtype)
        return reliability * clean_mask.view_as(reliability)

    def _build_history_context(self, img_latent, txt_latent, item_anchor, img_q, txt_q):
        img_rel = self._qe_reliability(img_q)
        txt_rel = self._qe_reliability(txt_q)
        img_ctx_rel = self._context_reliability(img_q, "image")
        txt_ctx_rel = self._context_reliability(txt_q, "text")

        img_hist = (txt_ctx_rel * txt_latent + item_anchor) / (
            txt_ctx_rel + 1.0
        ).clamp_min(1e-8)
        txt_hist = (img_ctx_rel * img_latent + item_anchor) / (
            img_ctx_rel + 1.0
        ).clamp_min(1e-8)
        id_hist = (
            img_rel * img_latent + txt_rel * txt_latent + item_anchor
        ) / (img_rel + txt_rel + 1.0).clamp_min(1e-8)
        return img_hist, txt_hist, id_hist

    def _get_modal_repair_raw_feat(self, modality):
        if modality == "image":
            return self._img_raw_snapshot
        return self._txt_raw_snapshot

    def _build_clean_text_adapter_feat(self):
        base_txt = self._txt_raw_snapshot
        txt_delta = self.text_embedding.weight - base_txt
        return base_txt + self.txt_clean_adapter_scale * txt_delta

    def _get_clean_route_mask(self, q, modality):
        if modality == "image":
            margin = self.img_clean_route_margin
            threshold = self.img_hard_clean_corrupt_threshold
        else:
            margin = self.txt_clean_route_margin
            threshold = self.txt_hard_clean_corrupt_threshold

        clean_prob = q["p_clean"].detach()
        top_corrupt_prob = torch.maximum(
            q["p_noisy"].detach(), q["p_missing"].detach()
        )
        corrupt_prob = (q["p_noisy"] + q["p_missing"]).detach()
        return (clean_prob >= (top_corrupt_prob + margin)) | (corrupt_prob <= threshold)

    def _mix_state_experts(
        self,
        target_emb,
        other_emb,
        item_emb,
        target_hist,
        id_hist,
        target_q,
        other_q,
        keep_expert,
        noisy_expert,
        missing_expert,
        modality=None,
    ):
        keep_out = keep_expert(target_emb)
        noisy_out = noisy_expert(
            target_emb=target_emb,
            other_emb=other_emb,
            item_emb=item_emb,
            target_hist=target_hist,
            id_hist=id_hist,
            target_state_emb=target_q["state_emb"],
            other_state_emb=other_q["state_emb"],
            target_severity=target_q["severity"],
            other_severity=other_q["severity"],
        )
        missing_out = missing_expert(
            target_emb=target_emb,
            other_emb=other_emb,
            item_emb=item_emb,
            target_hist=target_hist,
            id_hist=id_hist,
        )

        if modality is None or self.ablation_use_full_state_gate:
            gate = target_q["prob"]
            return (
                gate[:, 0:1] * keep_out
                + gate[:, 1:2] * noisy_out
                + gate[:, 2:3] * missing_out
            )

        clean_mask = self._get_clean_route_mask(target_q, modality)
        corrupt_gate = target_q["prob"][:, 1:3]
        corrupt_gate = corrupt_gate / corrupt_gate.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        corrupt_mix = corrupt_gate[:, 0:1] * noisy_out + corrupt_gate[:, 1:2] * missing_out
        return torch.where(clean_mask.expand_as(keep_out), keep_out, corrupt_mix)

    def _encode_modalities(self, img_feat, txt_feat):
        item_image = torch.sigmoid(
            self.shared_encoder(self.act_g(self.image_encoder(img_feat)))
        )
        item_text = torch.sigmoid(
            self.shared_encoder(self.act_g(self.text_encoder(txt_feat)))
        )
        return item_image, item_text

    def _apply_clean_bypass(self, raw_feat, repaired_feat, q):
        if not self.repair_clean_bypass:
            return repaired_feat
        keep_weight = q["p_clean"].detach().clamp(0.0, 1.0)
        if self.repair_clean_bypass_power != 1.0:
            keep_weight = keep_weight.pow(self.repair_clean_bypass_power)
        return keep_weight * raw_feat + (1.0 - keep_weight) * repaired_feat

    def _apply_modality_hard_clean_bypass(self, raw_feat, repaired_feat, q, modality):
        keep_mask = self._get_clean_route_mask(q, modality).expand_as(repaired_feat)
        return torch.where(keep_mask, raw_feat, repaired_feat)

    def _apply_text_residual_repair(self, raw_feat, decoded_feat, q):
        corrupt_weight = (q["p_noisy"] + q["p_missing"]).detach()
        residual_scale = self.repair_txt_residual_alpha * corrupt_weight
        repaired_feat = raw_feat + residual_scale * (decoded_feat - raw_feat)
        return self._apply_modality_hard_clean_bypass(raw_feat, repaired_feat, q, "text")

    def mge(self):
        raw_img = self._get_modal_repair_raw_feat("image")
        raw_txt = self._get_modal_repair_raw_feat("text")

        qe_ctx = self._get_qe_ctx()
        img_q = qe_ctx["img_q"]
        txt_q = qe_ctx["txt_q"]
        self._last_qe_ctx = qe_ctx

        if self.ablation_disable_moe_repair:
            self._last_txt_repair_core_feat = raw_txt
            item_image, item_text = self._encode_modalities(raw_img, raw_txt)
            return item_image, item_text, raw_img, raw_txt, img_q, txt_q

        img_latent = self.image_repair_proj(raw_img)
        txt_latent = self.text_repair_proj(raw_txt)
        item_anchor = self.item_repair_proj(self.item_id_embedding.weight)
        img_hist, txt_hist, id_hist = self._build_history_context(
            img_latent, txt_latent, item_anchor, img_q, txt_q
        )

        repaired_img_latent = self._mix_state_experts(
            target_emb=img_latent,
            other_emb=txt_latent,
            item_emb=item_anchor,
            target_hist=img_hist,
            id_hist=id_hist,
            target_q=img_q,
            other_q=txt_q,
            keep_expert=self.img_keep_expert,
            noisy_expert=self.img_noisy_expert,
            missing_expert=self.img_missing_expert,
            modality="image",
        )
        repaired_txt_latent = self._mix_state_experts(
            target_emb=txt_latent,
            other_emb=img_latent,
            item_emb=item_anchor,
            target_hist=txt_hist,
            id_hist=id_hist,
            target_q=txt_q,
            other_q=img_q,
            keep_expert=self.txt_keep_expert,
            noisy_expert=self.txt_noisy_expert,
            missing_expert=self.txt_missing_expert,
            modality="text",
        )

        decoded_img_feat = self.image_repair_decode(repaired_img_latent)
        decoded_txt_feat = self.text_repair_decode(repaired_txt_latent)

        soft_img_feat = self._apply_clean_bypass(raw_img, decoded_img_feat, img_q)
        repaired_img_feat = self._apply_modality_hard_clean_bypass(
            raw_img, soft_img_feat, img_q, "image"
        )
        core_repaired_txt_feat = self._apply_text_residual_repair(
            raw_txt, decoded_txt_feat, txt_q
        )
        clean_txt_mask = self._get_clean_route_mask(txt_q, "text")
        clean_txt_feat = self._build_clean_text_adapter_feat()
        repaired_txt_feat = torch.where(
            clean_txt_mask.expand_as(core_repaired_txt_feat),
            clean_txt_feat,
            core_repaired_txt_feat,
        )
        self._last_txt_repair_core_feat = core_repaired_txt_feat

        item_image, item_text = self._encode_modalities(
            repaired_img_feat, repaired_txt_feat
        )
        return item_image, item_text, repaired_img_feat, repaired_txt_feat, img_q, txt_q

    def _propagate_modalities(self, item_image, item_text):
        item_image_filter = torch.sparse.mm(
            self.adj.t(), torch.tanh(self.image_preference_(self.user_embedding.weight))
        ) * self.num_inters[self.n_users:]
        item_text_filter = torch.sparse.mm(
            self.adj.t(), torch.tanh(self.text_preference_(self.user_embedding.weight))
        ) * self.num_inters[self.n_users:]

        item_image = torch.einsum("ij,ij->ij", item_image_filter, item_image)
        item_text = torch.einsum("ij,ij->ij", item_text_filter, item_text)
        for _ in range(self.n_mm_layers):
            item_image = torch.sparse.mm(self.image_adj, item_image)
            item_text = torch.sparse.mm(self.text_adj, item_text)

        user_image = torch.sparse.mm(self.adj, item_image) * self.num_inters[: self.n_users]
        user_text = torch.sparse.mm(self.adj, item_text) * self.num_inters[: self.n_users]
        return item_image, item_text, user_image, user_text

    def _build_valid_mask(self, size, device, exclude_idx=None):
        valid_mask = torch.ones(size, dtype=torch.bool, device=device)
        if exclude_idx is None or len(exclude_idx) == 0:
            return valid_mask
        valid_mask[torch.from_numpy(exclude_idx).long().to(device)] = False
        return valid_mask

    def _weighted_repair_loss(self, pred, target, exclude_idx=None, weight=None):
        valid_mask = self._build_valid_mask(pred.size(0), pred.device, exclude_idx)
        pred = pred[valid_mask]
        target = target[valid_mask]
        if pred.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        base_loss = F.mse_loss(pred, target, reduction="none")
        if weight is None:
            return 0.05 * base_loss.mean()

        weight = weight[valid_mask].clamp_min(1e-6)
        return 0.05 * (base_loss * weight).sum() / (weight.sum() * base_loss.size(-1))

    def _weighted_cosine_repair_loss(self, pred, target, exclude_idx=None, weight=None):
        valid_mask = self._build_valid_mask(pred.size(0), pred.device, exclude_idx)
        pred = pred[valid_mask]
        target = target[valid_mask]
        if pred.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        base_loss = 1.0 - F.cosine_similarity(pred, target, dim=-1, eps=1e-8)
        if weight is None:
            return base_loss.mean()

        weight = weight[valid_mask].view(-1).clamp_min(1e-6)
        return (base_loss * weight).sum() / weight.sum()

    def _repair_reconstruction_loss(
        self,
        repaired_img_feat,
        repaired_txt_feat,
        img_q,
        txt_q,
    ):
        core_repaired_txt_feat = self._last_txt_repair_core_feat
        if core_repaired_txt_feat is None:
            core_repaired_txt_feat = repaired_txt_feat

        img_weight = (img_q["p_noisy"] + img_q["p_missing"]).detach()
        txt_weight = (txt_q["p_noisy"] + txt_q["p_missing"]).detach()

        if self._img_clean_target is not None:
            img_target = self._img_clean_target
            img_exclude_idx = None
        else:
            img_target = self._img_raw_snapshot
            img_exclude_idx = self.missing_items_v

        if self._txt_clean_target is not None:
            txt_target = self._txt_clean_target
            txt_exclude_idx = None
        else:
            txt_target = self._txt_raw_snapshot
            txt_exclude_idx = self.missing_items_t

        loss = self._weighted_repair_loss(
            repaired_img_feat, img_target, img_exclude_idx, img_weight
        )
        loss = loss + self._weighted_repair_loss(
            core_repaired_txt_feat, txt_target, txt_exclude_idx, txt_weight
        )
        if self.repair_img_cosine_weight > 0.0:
            loss = loss + self.repair_img_cosine_weight * self._weighted_cosine_repair_loss(
                repaired_img_feat, img_target, img_exclude_idx, img_weight
            )
        if self.repair_txt_cosine_weight > 0.0:
            loss = loss + self.repair_txt_cosine_weight * self._weighted_cosine_repair_loss(
                core_repaired_txt_feat, txt_target, txt_exclude_idx, txt_weight
            )

        img_clean_keep_target = self._get_modal_repair_raw_feat("image").detach()
        txt_clean_keep_target = self._get_modal_repair_raw_feat("text").detach()
        if self.repair_img_clean_keep_weight > 0.0:
            loss = loss + self.repair_img_clean_keep_weight * self._weighted_repair_loss(
                repaired_img_feat,
                img_clean_keep_target,
                None,
                img_q["p_clean"].detach(),
            )
        if self.repair_txt_clean_keep_weight > 0.0:
            loss = loss + self.repair_txt_clean_keep_weight * self._weighted_repair_loss(
                core_repaired_txt_feat,
                txt_clean_keep_target,
                None,
                txt_q["p_clean"].detach(),
            )
        return loss

    def _router_prior_loss(self):
        if (
            not self.use_router_prior
            or self.loss_weight_router <= 0.0
            or self.freeze_qe
            or self._last_qe_ctx is None
        ):
            return torch.tensor(0.0, device=self.device)

        total = torch.tensor(0.0, device=self.device)
        count = 0
        for q_key, missing_idx in (
            ("img_q", self.missing_items_v),
            ("txt_q", self.missing_items_t),
        ):
            if len(missing_idx) == 0:
                continue
            idx = torch.from_numpy(missing_idx).long().to(self.device)
            q = self._last_qe_ctx[q_key]
            gate = q["prob"][idx]
            prior = torch.zeros_like(gate)
            prior[:, 2] = 1.0
            total = total + F.kl_div(
                gate.clamp_min(1e-8).log(), prior, reduction="batchmean"
            )
            count += 1
        if count == 0:
            return torch.tensor(0.0, device=self.device)
        return total / count

    def reg_loss(self, *embs):
        reg = 0.0
        for emb in embs:
            reg = reg + torch.norm(emb, p=2)
        return reg / embs[-1].shape[0]

    def calculate_reg_loss(self, user_emb, pos_items_emb, neg_item_emb, image_emb, text_emb):
        loss_reg = self.reg_loss(user_emb, pos_items_emb, neg_item_emb) * 1e-5
        loss_reg += self.reg_loss(image_emb) * 0.1
        loss_reg += self.reg_loss(text_emb) * 0.1
        return loss_reg

    def info_nce(self, view1, view2, temperature=0.4):
        if view1.size(0) <= 1 or view2.size(0) <= 1:
            return torch.tensor(0.0, device=self.device)
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        pos_score = torch.exp((view1 * view2).sum(dim=-1) / temperature)
        ttl_score = torch.exp(torch.matmul(view1, view2.transpose(0, 1)) / temperature).sum(
            dim=1
        )
        return (-torch.log(pos_score / ttl_score.clamp_min(1e-8))).mean()

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)
        return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))

    def pre_epoch_processing(self):
        self._qe_log_epoch += 1
        self._invalidate_eval_cache()
        if not self.freeze_qe:
            self._invalidate_qe_cache()
        if (
            self.mm_adj_refresh_interval > 0
            and self._qe_log_epoch % self.mm_adj_refresh_interval == 0
        ):
            self._refresh_repaired_mm_adj()

    def _format_qe_prob_line(self, label, q):
        prob = 100.0 * q["prob"].detach().mean(dim=0)
        reliability = 100.0 * q["reliability"].detach().mean().item()
        severity = 100.0 * q["severity"].detach().mean().item()
        return (
            f"[QEProb][train@epoch{self._qe_log_epoch}][{label}] "
            f"clean={prob[0].item():.2f} noisy={prob[1].item():.2f} "
            f"missing={prob[2].item():.2f} rel={reliability:.2f} sev={severity:.2f}"
        )

    def _should_log_extra_diag(self):
        if self._qe_log_epoch <= 3:
            return True
        return self.diag_log_interval > 0 and self._qe_log_epoch % self.diag_log_interval == 0

    def _state_percent(self, state):
        counts = torch.bincount(state.detach().view(-1), minlength=3).float()
        total = counts.sum().clamp_min(1.0)
        return 100.0 * counts / total

    def _format_state_triplet(self, values):
        return (
            f"clean={values[0].item():.2f} "
            f"noisy={values[1].item():.2f} "
            f"missing={values[2].item():.2f}"
        )

    def _build_qe_state_diag_lines(self, label, q, gt_state):
        pred = q["prob"].detach().argmax(dim=-1)
        acc = 100.0 * (pred == gt_state).float().mean().item()
        gt_dist = self._state_percent(gt_state)
        pred_dist = self._state_percent(pred)
        recalls = []
        for state_id in range(3):
            mask = gt_state == state_id
            recalls.append(
                100.0 * (pred[mask] == state_id).float().mean().item() if mask.any() else 0.0
            )

        return [
            (
                f"[QEDiag][train@epoch{self._qe_log_epoch}][{label}] "
                f"acc={acc:.2f} gt({self._format_state_triplet(gt_dist)}) "
                f"pred({self._format_state_triplet(pred_dist)})"
            ),
            (
                f"[QEDiag][train@epoch{self._qe_log_epoch}][{label}][recall] "
                f"clean={recalls[0]:.2f} noisy={recalls[1]:.2f} missing={recalls[2]:.2f}"
            ),
        ]

    def _metric_against_target(self, pred, target):
        cos = F.cosine_similarity(pred, target, dim=-1, eps=1e-8)
        mse = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
        return cos, mse

    def _format_repair_metric_line(self, label, state_name, state_count, raw_cos, rep_cos, raw_mse, rep_mse):
        return (
            f"[RepairDiag][train@epoch{self._qe_log_epoch}][{label}][{state_name}][n={state_count}] "
            f"cos raw={raw_cos:.4f} rep={rep_cos:.4f} delta={rep_cos - raw_cos:+.4f} "
            f"mse raw={raw_mse:.6f} rep={rep_mse:.6f} delta={rep_mse - raw_mse:+.6f}"
        )

    def _build_repair_diag_lines(self, label, raw_feat, repaired_feat, clean_target, gt_state):
        raw_cos_all, raw_mse_all = self._metric_against_target(raw_feat, clean_target)
        rep_cos_all, rep_mse_all = self._metric_against_target(repaired_feat, clean_target)

        lines = [
            self._format_repair_metric_line(
                label=label,
                state_name="all",
                state_count=int(raw_feat.size(0)),
                raw_cos=raw_cos_all.mean().item(),
                rep_cos=rep_cos_all.mean().item(),
                raw_mse=raw_mse_all.mean().item(),
                rep_mse=rep_mse_all.mean().item(),
            )
        ]

        for state_id, state_name in enumerate(("clean", "noisy", "missing")):
            mask = gt_state == state_id
            if not mask.any():
                continue
            lines.append(
                self._format_repair_metric_line(
                    label=label,
                    state_name=state_name,
                    state_count=int(mask.sum().item()),
                    raw_cos=raw_cos_all[mask].mean().item(),
                    rep_cos=rep_cos_all[mask].mean().item(),
                    raw_mse=raw_mse_all[mask].mean().item(),
                    rep_mse=rep_mse_all[mask].mean().item(),
                )
            )
        return lines

    def _compute_extra_diag_lines(self):
        if not self._should_log_extra_diag():
            return []

        was_training = self.training
        try:
            self.eval()
            with torch.no_grad():
                _, _, repaired_img_feat, repaired_txt_feat, img_q, txt_q = self.mge()
        finally:
            if was_training:
                self.train()

        lines = []
        if self.log_qe_state_stats:
            lines.extend(self._build_qe_state_diag_lines("Img", img_q, self._gt_image_state))
            lines.extend(self._build_qe_state_diag_lines("Txt", txt_q, self._gt_text_state))

        if self.log_repair_feature_stats:
            if self._img_clean_target is not None:
                lines.extend(
                    self._build_repair_diag_lines(
                        "Img",
                        self._img_raw_snapshot.detach(),
                        repaired_img_feat.detach(),
                        self._img_clean_target.detach(),
                        self._gt_image_state,
                    )
                )
            if self._txt_clean_target is not None:
                lines.extend(
                    self._build_repair_diag_lines(
                        "Txt",
                        self._txt_raw_snapshot.detach(),
                        repaired_txt_feat.detach(),
                        self._txt_clean_target.detach(),
                        self._gt_text_state,
                    )
                )
        return lines

    def post_epoch_processing(self):
        if not self.log_qe_prob or self._last_qe_ctx is None:
            base_lines = []
        else:
            base_lines = [
                self._format_qe_prob_line("Img", self._last_qe_ctx["img_q"]),
                self._format_qe_prob_line("Txt", self._last_qe_ctx["txt_q"]),
            ]

        extra_lines = self._compute_extra_diag_lines()
        lines = base_lines + extra_lines
        if not lines:
            return None
        return "\n".join(lines)

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction
        self._invalidate_eval_cache()
        if not self.freeze_qe:
            self._invalidate_qe_cache()

        user_embeddings, item_embedding = self.cge(
            self.user_embedding.weight, self.item_id_embedding.weight, self.norm_adj
        )
        item_image, item_text, repaired_img_feat, repaired_txt_feat, img_q, txt_q = self.mge()

        all_items, _ = torch.unique(
            torch.cat((pos_items, neg_items)), return_inverse=True, sorted=False
        )
        valid_both_idx = np.setdiff1d(
            all_items.detach().cpu().numpy(),
            np.union1d(self.missing_items_t, self.missing_items_v),
        )

        loss_mm = self.info_nce(
            item_image[valid_both_idx], item_text[valid_both_idx], temperature=self.infoNCETemp
        )

        item_image, item_text, user_image, user_text = self._propagate_modalities(
            item_image, item_text
        )
        user_mm = 0.5 * (user_image + user_text)
        item_mm = 0.5 * (item_image + item_text)

        loss_align_ui = self.info_nce(
            user_embeddings[users], item_embedding[pos_items], temperature=self.alignUITemp
        )
        loss_align_ui += self.info_nce(
            user_mm[users], item_mm[pos_items], temperature=self.infoNCETemp
        )

        loss_align_id = self.info_nce(
            item_embedding[pos_items], item_mm[pos_items], temperature=self.alignBMTemp
        )
        loss_align_id += self.info_nce(
            user_embeddings[users], user_mm[users], temperature=self.alignBMTemp
        )

        user_emb = user_embeddings + user_mm
        item_emb = item_embedding + item_mm

        loss_main_bpr = self.bpr_loss(user_emb[users], item_emb[pos_items], item_emb[neg_items])
        loss_reg = self.calculate_reg_loss(
            user_embeddings[users],
            item_embedding[pos_items],
            item_embedding[neg_items],
            item_image[pos_items],
            item_text[pos_items],
        )
        loss_repair = self._repair_reconstruction_loss(
            repaired_img_feat, repaired_txt_feat, img_q, txt_q
        )
        loss_router = self.loss_weight_router * self._router_prior_loss()

        return (
            loss_main_bpr
            + self.loss_weight_repair * loss_repair
            + self.loss_weight_mm * loss_mm
            + self.loss_weight_align * (loss_align_ui + loss_align_id)
            + loss_reg
            + loss_router
        )

    def _refresh_repaired_mm_adj(self):
        was_training = self.training
        try:
            self.eval()
            with torch.no_grad():
                _, _, repaired_img_feat, repaired_txt_feat, img_q, txt_q = self.mge()
                img_corrupt_mask = ~self._get_clean_route_mask(img_q, "image").view(-1)
                txt_corrupt_mask = ~self._get_clean_route_mask(txt_q, "text").view(-1)

                img_corrupt_mean = float(img_corrupt_mask.float().mean().item())
                txt_corrupt_mean = float(txt_corrupt_mask.float().mean().item())
                if (
                    img_corrupt_mean <= self.mm_adj_refresh_skip_qe_threshold
                    and txt_corrupt_mean <= self.mm_adj_refresh_skip_qe_threshold
                ):
                    if hasattr(self, "logger") and self.logger is not None:
                        self.logger.info(
                            f"[MMAdj][train@epoch{self._qe_log_epoch}] "
                            f"skip QE-clean graph refresh "
                            f"(img_corrupt={img_corrupt_mean:.4f}, txt_corrupt={txt_corrupt_mean:.4f})"
                        )
                    return

                raw_img = self._get_modal_repair_raw_feat("image").detach()
                raw_txt = self._get_modal_repair_raw_feat("text").detach()
                graph_img_feat = raw_img.clone()
                graph_txt_feat = raw_txt.clone()
                graph_img_feat[img_corrupt_mask] = repaired_img_feat.detach()[img_corrupt_mask]
                graph_txt_feat[txt_corrupt_mask] = repaired_txt_feat.detach()[txt_corrupt_mask]

                self.image_adj = self._build_mm_adj(graph_img_feat, None)
                self.text_adj = self._build_mm_adj(graph_txt_feat, None)
        finally:
            if was_training:
                self.train()

        self._invalidate_eval_cache()
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.info(
                f"[MMAdj][train@epoch{self._qe_log_epoch}] refreshed from QE corrupt-only graph features "
                f"(img_items={int(img_corrupt_mask.sum().item())}, txt_items={int(txt_corrupt_mask.sum().item())})"
            )

    def _compute_eval_representations(self):
        user_embeddings, item_embedding = self.cge(
            self.user_embedding.weight, self.item_id_embedding.weight, self.norm_adj
        )
        item_image, item_text, _, _, _, _ = self.mge()
        item_image, item_text, user_image, user_text = self._propagate_modalities(
            item_image, item_text
        )

        user_mm = 0.5 * (user_image + user_text)
        item_mm = 0.5 * (item_image + item_text)
        return user_embeddings + user_mm, item_embedding + item_mm

    def _get_eval_representations(self):
        if self.training:
            return self._compute_eval_representations()
        if self._eval_repr_cache is None:
            self._eval_repr_cache = self._compute_eval_representations()
        return self._eval_repr_cache

    def full_sort_predict(self, interaction):
        users, _ = interaction
        user_emb, item_emb = self._get_eval_representations()
        return user_emb[users] @ item_emb.T

    def forward(self):
        raise NotImplementedError("Use calculate_loss or full_sort_predict instead.")


DGMRecStateMoE = StateMoE
