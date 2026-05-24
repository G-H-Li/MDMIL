# coding: utf-8
# @email  : enoche.chow@gmail.com

"""
Utility functions
##########################
"""

import numpy as np
import torch
import importlib
import datetime
import random


def evaluate_codebook_quality(i_code, item_token_cluster):
    """
    评价码本质量
    :param i_code: torch.LongTensor, [n_items, n_levels] 存储的是每个item在各层的索引
    :param item_token_cluster: int, 每一层码本的大小 (K)
    :return: dict 包含各项指标
    """
    n_items, n_levels = i_code.shape
    results = {}

    # --- 1. 计算码本利用率 (Usage / Perplexity) ---
    level_usages = []
    level_perplexities = []

    for l in range(n_levels):
        # 统计当前层被使用的唯一索引
        unique_codes = torch.unique(i_code[:, l])
        usage = len(unique_codes) / item_token_cluster
        level_usages.append(usage)

        # 计算困惑度 (Perplexity) - 衡量分布均匀性
        # 先算频率分布
        bincount = torch.bincount(i_code[:, l], minlength=item_token_cluster).float()
        prob = bincount / bincount.sum()
        # 过滤掉 0 以计算信息熵
        prob = prob[prob > 0]
        entropy = -torch.sum(prob * torch.log(prob + 1e-10))
        perplexity = torch.exp(entropy)
        level_perplexities.append(perplexity.item())

    results['usage_per_level'] = level_usages
    results['avg_usage'] = sum(level_usages) / n_levels
    results['perplexity_per_level'] = level_perplexities

    # --- 2. 计算 ID 碰撞率 (Collision Rate) ---
    # 只有当所有层级的 Code 完全一致时，才认为两个 Item 发生了碰撞
    # 将多列 Code 合并为一个元组或字符串进行唯一性检查
    # 转为列表以便处理元组
    code_list = i_code.tolist()
    unique_item_ids = set(tuple(c) for c in code_list)

    num_unique_ids = len(unique_item_ids)
    num_collisions = n_items - num_unique_ids
    collision_rate = num_collisions / n_items

    results['total_items'] = n_items
    results['unique_semantic_ids'] = num_unique_ids
    results['collision_count'] = num_collisions
    results['collision_rate'] = collision_rate

    return results


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y-%H-%M-%S')

    return cur


def get_model(model_name):
    r"""Automatically select model class based on model name
    Args:
        model_name (str): model name
    Returns:
        Recommender: model class
    """
    model_file_name = model_name.lower()
    module_path = '.'.join(['models', model_file_name])
    try:
        model_module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        raise ImportError(f"Could not import module {module_path}")

    model_class = getattr(model_module, model_name)
    return model_class


def get_trainer():
    return getattr(importlib.import_module('common.trainer'), 'Trainer')


def get_evaluator():
    return getattr(importlib.import_module('common.evaluator'), 'Evaluator')


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def early_stopping(value, best, cur_step, max_step, bigger=True):
    r""" validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
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
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric) + ': ' + '%.04f' % value + '    '
    return result_str


############ LATTICE Utilities #########

def build_knn_neighbourhood(adj, topk):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    # weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    indices0 = torch.arange(knn_ind.size(0)).unsqueeze(1).expand(-1, topk).to(adj.device)
    indices = torch.stack([indices0.flatten(), knn_ind.flatten()], dim=0)
    weighted_adjacency_matrix = torch.sparse_coo_tensor(indices, knn_val.flatten().squeeze(), adj.size())
    return weighted_adjacency_matrix


def compute_normalized_laplacian(adj):
    # rowsum = torch.sum(adj, -1)
    # d_inv_sqrt = torch.pow(rowsum, -0.5)
    # d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    # d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    # L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    # return L_norm
    indices = adj._indices()
    values = adj._values()

    # 计算每个节点的度
    row = indices[0]
    col = indices[1]
    rowsum = torch.sparse.sum(adj, dim=-1).to_dense()  # 每个节点的度
    d_inv_sqrt = torch.pow(rowsum, -0.5)  # D^(-1/2)
    d_inv_sqrt = torch.clamp(d_inv_sqrt, 0.0, 10.0)  # 防止数值不稳定

    # 计算归一化后的权重
    row_inv_sqrt = d_inv_sqrt[row]
    col_inv_sqrt = d_inv_sqrt[col]
    values = values * row_inv_sqrt * col_inv_sqrt

    # 构建归一化的拉普拉斯矩阵
    return torch.sparse_coo_tensor(indices, values, adj.shape)


def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim

def get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization='none'):
    from torch_scatter import scatter_add
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    if normalization == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif normalization == 'rw':
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight

def get_dense_laplacian(adj, normalization='none'):
    if normalization == 'sym':
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    elif normalization == 'rw':
        rowsum = torch.sum(adj, -1)
        d_inv = torch.pow(rowsum, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diagflat(d_inv)
        L_norm = torch.mm(d_mat_inv, adj)
    elif normalization == 'none':
        L_norm = adj
    return L_norm

def build_knn_normalized_graph(adj, topk, is_sparse, norm_type):
    device = adj.device
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    if is_sparse:
        tuple_list = [[row, int(col)] for row in range(len(knn_ind)) for col in knn_ind[row]]
        row = [i[0] for i in tuple_list]
        col = [i[1] for i in tuple_list]
        i = torch.LongTensor([row, col]).to(device)
        v = knn_val.flatten()
        edge_index, edge_weight = get_sparse_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
        return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)
    else:
        weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)

def get_masked_laplacian(edge_index, edge_weight, num_nodes, normalization='none'):
    from torch_scatter import scatter_add
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    if normalization == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        edge_weight.masked_fill_(edge_weight+1e-6 > 1.0, 0.)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        edge_weight.masked_fill_(edge_weight <= 1e-7, 1e-7)
    elif normalization == 'rw':
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight

def build_graph_from_adj(adj, is_sparse=True, norm_type='sym',mask=False):
    device = adj.device
    i = adj.coalesce().indices()
    v = adj.coalesce().values()
    if mask:
        edge_index, edge_weight = get_masked_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
    else:
        edge_index, edge_weight = get_sparse_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
    return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)

def build_non_zero_graph(adj, is_sparse=True, norm_type='sym'):
    device = adj.device
    nonzero_indices = adj.nonzero()
    i = nonzero_indices.T
    v = adj[nonzero_indices[:, 0], nonzero_indices[:, 1]]
    edge_index, edge_weight = get_sparse_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
    return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)