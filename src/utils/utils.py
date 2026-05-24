# coding: utf-8

import datetime
import importlib
import os
import random
import re

import numpy as np
import torch


def get_local_time():
    return datetime.datetime.now().strftime("%b-%d-%Y-%H-%M-%S")


def _camel_to_snake(name):
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def get_model(model_name):
    module_candidates = [model_name.lower(), _camel_to_snake(model_name)]
    model_file_modules = []
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    if os.path.isdir(models_dir):
        normalized_target = model_name.lower()
        for file_name in sorted(os.listdir(models_dir)):
            if not file_name.endswith(".py") or file_name == "__init__.py":
                continue
            stem = file_name[:-3]
            model_file_modules.append(stem)
            if stem.replace("_", "").lower() == normalized_target:
                module_candidates.append(stem)
    module_candidates.extend(model_file_modules)
    for module_name in dict.fromkeys(module_candidates):
        module_path = ".".join(["models", module_name])
        if importlib.util.find_spec(module_path) is None:
            continue
        model_module = importlib.import_module(module_path)
        if hasattr(model_module, model_name):
            return getattr(model_module, model_name)
    raise ModuleNotFoundError(
        "Cannot locate model {}. Tried modules: {}".format(
            model_name, ", ".join(dict.fromkeys(module_candidates))
        )
    )


def get_trainer():
    return getattr(importlib.import_module("common.trainer"), "Trainer")


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def early_stopping(value, best, cur_step, max_step, bigger=True):
    stop_flag = False
    update_flag = False
    if bigger:
        if value > best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value < best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def dict2str(result_dict):
    result_str = ""
    for metric, value in result_dict.items():
        result_str += str(metric) + ": " + "%.04f" % value + "    "
    return result_str


def build_knn_neighbourhood(adj, topk):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    return torch.zeros_like(adj).scatter_(-1, knn_ind, knn_val)


def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)


def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    return torch.mm(context_norm, context_norm.transpose(1, 0))


def build_chunked_sparse_knn_graph(context, topk, chunk_size=1024, build_device="cpu", mask_indices=None):
    device = torch.device(build_device)
    context = context.detach().to(device)
    context = context.div(torch.norm(context, p=2, dim=-1, keepdim=True) + 1e-12)

    n_nodes = context.shape[0]
    rows_all = []
    cols_all = []
    vals_all = []

    for start in range(0, n_nodes, chunk_size):
        end = min(start + chunk_size, n_nodes)
        sim = torch.mm(context[start:end], context.transpose(1, 0))
        knn_val, knn_ind = torch.topk(sim, topk, dim=-1)

        rows = (
            torch.arange(start, end, device=device)
            .unsqueeze(1)
            .expand_as(knn_ind)
            .reshape(-1)
        )
        cols = knn_ind.reshape(-1)
        vals = knn_val.reshape(-1)

        rows_all.append(rows)
        cols_all.append(cols)
        vals_all.append(vals)

        del sim, knn_val, knn_ind

    row = torch.cat(rows_all, dim=0)
    col = torch.cat(cols_all, dim=0)
    val = torch.cat(vals_all, dim=0)

    if mask_indices is not None and len(mask_indices) > 0:
        mask_indices = torch.as_tensor(mask_indices, dtype=torch.long, device=device)
        missing_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        missing_mask[mask_indices] = True
        keep_mask = (~missing_mask[row]) & (~missing_mask[col])
        row = row[keep_mask]
        col = col[keep_mask]
        val = val[keep_mask]

        row = torch.cat([row, mask_indices], dim=0)
        col = torch.cat([col, mask_indices], dim=0)
        val = torch.cat(
            [val, torch.ones(mask_indices.shape[0], device=device, dtype=val.dtype)],
            dim=0,
        )

    deg = torch.zeros(n_nodes, dtype=val.dtype, device=device)
    deg.index_add_(0, row, val)
    d_inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)
    norm_val = d_inv_sqrt[row] * val * d_inv_sqrt[col]

    indices = torch.stack([row, col], dim=0)
    return torch.sparse_coo_tensor(indices, norm_val, (n_nodes, n_nodes)).coalesce()
