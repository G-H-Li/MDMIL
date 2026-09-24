import os

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn
import torch.nn.functional as F

from src.common.abstract_recommender import GeneralRecommender



class MDMIL(GeneralRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.config = config
        self.emb_size = config['embedding_size']
        self.n_intents = config['n_intents']
        self.n_ui_layers = config['n_ui_layers']
        self.n_ho_layers = config['n_ho_layers']

        self.tau = config['tau']
        self.lamda_de = config['lamda_de']
        self.lamda_int = config['lamda_int']
        self.lamda_cl = config['lamda_cl']

        self.knn_k = config['knn_k']
        self.adj_dropout = config['adj_dropout']
        self.m_dropout = config['m_dropout']

        # 交互图初始化
        self.n_nodes = self.n_users + self.n_items
        interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.ui_indices = torch.LongTensor(np.vstack((interaction_matrix.row, interaction_matrix.col))).to(self.device)

        # 模态特征、UU图、II图初始化
        if self.v_feat is not None:
            vision_features = torch.tensor(self.v_feat, dtype=torch.float32, device=self.device)
            self.i_v_emb = nn.Embedding.from_pretrained(vision_features, freeze=False)
            self.u_v_emb = nn.Embedding(self.n_users, self.emb_size).to(self.device)
            nn.init.xavier_normal_(self.u_v_emb.weight)
        if self.t_feat is not None:
            self.i_t_emb = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.u_t_emb = nn.Embedding(self.n_users, self.emb_size).to(self.device)
            nn.init.xavier_normal_(self.u_t_emb.weight)
        if self.v_feat is None and self.t_feat is None:
            raise ValueError("At least one of v_feat or t_feat must be provided.")

        self.u_c_emb = nn.Embedding(self.n_users, self.emb_size).to(self.device)
        nn.init.xavier_normal_(self.u_c_emb.weight)
        self.decouple_encoder = DecoupleEncoder(self.i_v_emb.weight.shape[1], self.t_feat.shape[1],
                                                self.emb_size, self.n_items, self.device, self.use_soft_fusion,
                                                dropout=self.m_dropout, tau=self.tau)

        with torch.no_grad():
            base_adj = get_base_adj(self.ui_indices.clone(), self.n_users, self.n_items, self.device)
            self.norm_base_adj = cal_norm_laplacian(base_adj)
            # item modality adjacency
            ii_v_adj = get_knn_adj(self.v_feat, knn_k=self.knn_k).to(self.device)
            ii_t_adj = get_knn_adj(self.t_feat, knn_k=self.knn_k).to(self.device)
            ii_c_adj = get_item_co_knn_adj(self.ui_indices.clone(), knn_k=self.knn_k,
                                           n_users=self.n_users, n_items=self.n_items).to(self.device)
            self.norm_ii_v_adj = cal_norm_laplacian(ii_v_adj)
            self.norm_ii_t_adj = cal_norm_laplacian(ii_t_adj)
            self.norm_ii_m_adj = cal_norm_laplacian(ii_v_adj + ii_t_adj)
            self.norm_ii_c_adj = cal_norm_laplacian(ii_c_adj)
            # user modality adjacency
            u_t_emb = self.cal_user_embedding_mean(self.i_t_emb.weight)
            u_v_emb = self.cal_user_embedding_mean(self.i_v_emb.weight)
            uu_t_adj = get_knn_adj(u_t_emb, knn_k=self.knn_k).to(self.device)
            uu_v_adj = get_knn_adj(u_v_emb, knn_k=self.knn_k).to(self.device)
            uu_c_adj = get_user_co_knn_adj(self.ui_indices.clone(), knn_k=self.knn_k,
                                           n_users=self.n_users, n_items=self.n_items).to(self.device)
            self.norm_uu_v_adj = cal_norm_laplacian(uu_v_adj)
            self.norm_uu_t_adj = cal_norm_laplacian(uu_t_adj)
            self.norm_uu_m_adj = cal_norm_laplacian(uu_t_adj + uu_v_adj)
            self.norm_uu_c_adj = cal_norm_laplacian(uu_c_adj)

        self.ui_gcn = P_GCN(self.n_ui_layers, self.adj_dropout)
        self.i_gcn = P_GCN(self.n_ho_layers, 0)
        self.u_gcn = P_GCN(self.n_ho_layers, 0)

        self.u_v_ints = IntentDisentangler(self.emb_size, self.emb_size, self.n_intents, self.device, self.use_intent)
        self.u_t_ints = IntentDisentangler(self.emb_size, self.emb_size, self.n_intents, self.device, self.use_intent)
        self.u_c_ints = IntentDisentangler(self.emb_size, self.emb_size, self.n_intents, self.device, self.use_intent)

    def cal_user_embedding_mean(self, embeddings):
        rows = self.ui_indices[0]
        cols = self.ui_indices[1]
        # 获取物品的image_embedding
        item_embeddings = embeddings[cols]
        # 初始化用户的image_embedding均值
        user_embedding_sum = torch.zeros((self.n_users, embeddings.size(-1)), device=self.device)
        user_interaction_count = torch.zeros(self.n_users, device=self.device)
        # 累加每个用户的交互物品的image_embedding
        user_embedding_sum.index_add_(0, rows, item_embeddings)
        user_interaction_count.index_add_(0, rows, torch.ones_like(rows, dtype=torch.float32))
        # 计算每个用户的image_embedding均值
        user_embedding_mean = user_embedding_sum / user_interaction_count.unsqueeze(1)
        user_embedding_mean = torch.nan_to_num(user_embedding_mean, nan=0.0, posinf=0.0, neginf=0.0)
        return user_embedding_mean

    def forward(self, uids=None, piids=None, niids=None):
        iids = torch.cat([piids, niids], dim=0) if piids is not None and niids is not None else None
        u_v_p, u_t_p, u_c_p = self.u_v_emb.weight, self.u_t_emb.weight, self.u_c_emb.weight
        i_v_p, i_t_p, i_c_p, align_loss, orth_loss = self.decouple_encoder(self.i_v_emb.weight,
                                                                               self.i_t_emb.weight,
                                                                               indices=iids)
        # 双向图学习模块
        ui_v_p = torch.cat([u_v_p, i_v_p], dim=0)
        ui_t_p = torch.cat([u_t_p, i_t_p], dim=0)
        ui_c_p = torch.cat([u_c_p, i_c_p], dim=0)
        ui_v_p = self.ui_gcn(ui_v_p, self.norm_base_adj)
        ui_t_p = self.ui_gcn(ui_t_p, self.norm_base_adj)
        ui_c_p = self.ui_gcn(ui_c_p, self.norm_base_adj)

        u_mv_p, i_mv_p = torch.split(ui_v_p, [self.n_users, self.n_items], dim=0)
        u_mt_p, i_mt_p = torch.split(ui_t_p, [self.n_users, self.n_items], dim=0)
        u_mc_p, i_mc_p = torch.split(ui_c_p, [self.n_users, self.n_items], dim=0)

        u_mv_bp = self.u_gcn(u_mv_p, self.norm_uu_v_adj)
        u_mt_bp = self.u_gcn(u_mt_p, self.norm_uu_t_adj)
        i_mv_bp = self.i_gcn(i_mv_p, self.norm_ii_v_adj)
        i_mt_bp = self.i_gcn(i_mt_p, self.norm_ii_t_adj)
        u_mc_bp = self.u_gcn(u_mc_p, self.norm_uu_c_adj + self.norm_uu_m_adj)
        i_mc_bp = self.i_gcn(i_mc_p, self.norm_ii_c_adj + self.norm_ii_m_adj)

        # 意图学习模块
        u_v_ip = torch.zeros_like(u_mv_bp)
        u_t_ip = torch.zeros_like(u_mt_bp)
        u_c_ip = torch.zeros_like(u_mc_bp)
        u_v_ip = self.u_v_ints(u_mv_bp)
        u_t_ip = self.u_t_ints(u_mt_bp)
        u_c_ip = self.u_c_ints(u_mc_bp)

        u_mv_emb = u_mv_bp + u_v_ip
        u_mt_emb = u_mt_bp + u_t_ip
        i_mv_emb = i_mv_bp
        i_mt_emb = i_mt_bp
        u_mc_emb = u_mc_bp + u_c_ip
        i_mc_emb = i_mc_bp

        if self.training:
            # rec_loss计算
            rec_loss = self.sl_loss(u_mv_emb[uids], i_mv_emb[piids], i_mv_emb[niids], temp=self.tau) + \
                    self.sl_loss(u_mt_emb[uids], i_mt_emb[piids], i_mt_emb[niids], temp=self.tau)+ \
                       self.sl_loss(u_mc_emb[uids], i_mc_emb[piids], i_mc_emb[niids], temp=self.tau)
            # decouple_loss计算
            decouple_loss = torch.tensor(0.0, device=self.device)
            decouple_loss += align_loss
            decouple_loss += orth_loss

            # intent_loss 计算
            intent_loss = self.u_v_ints.intent_loss(temp=self.tau) + self.u_t_ints.intent_loss(temp=self.tau) + \
                          self.u_c_ints.intent_loss(temp=self.tau)

            # cl_loss计算
            cl_loss = self.infonce_loss(u_mv_emb[uids], i_mv_emb[piids], temp=self.tau) + \
                         self.infonce_loss(u_mt_emb[uids], i_mt_emb[piids], temp=self.tau) + \
                         self.infonce_loss(u_mc_emb[uids], i_mc_emb[piids], temp=self.tau)

            return rec_loss, self.lamda_de * decouple_loss, self.lamda_int * intent_loss, self.lamda_cl * cl_loss
        else:
            scores = None
            scores_list = [torch.matmul(u_mv_emb[uids], i_mv_emb.T),
                           torch.matmul(u_mt_emb[uids], i_mt_emb.T),
                           torch.matmul(u_mc_emb[uids], i_mc_emb.T))]

            attn_weights = torch.softmax(scores_list, dim=0)
            scores = (attn_weights * scores_list).sum(dim=0)
            return scores

    def sl_loss(self, users, pos_items, neg_items, temp=0.1):
        pos_scores = F.cosine_similarity(users, pos_items)
        neg_scores = F.cosine_similarity(users.unsqueeze(1), neg_items, dim=2)  # (B, N)
        d = neg_scores - pos_scores.unsqueeze(1)
        loss = torch.logsumexp(d / temp, dim=1).mean()
        return loss

    def infonce_loss(self, emb1, emb2, temp=0.1):
        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)
        scores = torch.exp(torch.matmul(emb1, emb2.T) / temp)
        pos_sim = scores.diag()
        loss = -torch.log(pos_sim / torch.sum(scores, dim=1)).mean()
        return loss

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]
        loss = self.forward(users, pos_items, neg_items)
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        scores = self.forward(user)
        return scores


class P_GCN(nn.Module):
    def __init__(self, n_layers, dropout=0.3):
        super(P_GCN, self).__init__()
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, norm_adj):
        h = x
        for i in range(self.n_layers):
            # update after accepted
        return h


class MyMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(MyMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        return self.mlp(x)


class DecoupleEncoder(nn.Module):
    def __init__(self, input_v_dim, input_t_dim, hidden_dim, num_item, device, use_soft_fusion=True, dropout=0.3, tau=0.1):
        super(DecoupleEncoder, self).__init__()
        self.device = device
        self.dropout = dropout
        self.tau = tau
        self.use_soft_fusion = use_soft_fusion

        self.v_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.t_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.v_uni_mlp = MyMLP(input_v_dim, hidden_dim, self.dropout)
        self.t_uni_mlp = MyMLP(input_t_dim, hidden_dim, self.dropout)
        self.shared_com_v_mlp = MyMLP(input_v_dim, hidden_dim, self.dropout)
        self.shared_com_t_mlp = MyMLP(input_t_dim, hidden_dim, self.dropout)

        self.i_emb = nn.Embedding(num_item, hidden_dim).to(self.device)
        nn.init.xavier_normal_(self.i_emb.weight)

    def shared_fusion(self, h1, h2):
        term1 = torch.stack([h1 + h2, h1 + h2, h1, h2], dim=2)
        term2 = torch.stack([torch.zeros_like(h1), torch.zeros_like(h1), h1, h2], dim=2)
        feat_avg_shared = torch.logsumexp(term1, dim=2) - torch.logsumexp(term2, dim=2)
        return feat_avg_shared

    def orthogonality_loss(self, h1, h2):
        return torch.abs(F.cosine_similarity(h1, h2, dim=-1)).mean()

    def infonce_loss(self, emb1, emb2, temp=0.1):
        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)
        scores = torch.exp(torch.matmul(emb1, emb2.T) / temp)
        pos_sim = scores.diag()
        loss = -torch.log(pos_sim / torch.sum(scores, dim=1)).mean()
        return loss

    def forward(self, x_v, x_t, indices=None):
        x_v_aug = self.v_dropout(x_v)
        x_t_aug = self.t_dropout(x_t)

        x_v_uni = self.v_uni_mlp(x_v_aug)
        x_t_uni = self.t_uni_mlp(x_t_aug)

        x_t_com = self.shared_com_t_mlp(x_t)
        x_v_com = self.shared_com_v_mlp(x_v)
        if self.use_soft_fusion:
            x_shared = self.shared_fusion(x_v_com, x_t_com) * self.i_emb.weight
        else:
            x_shared = (x_v_com + x_t_com) / 2

        align_loss = torch.tensor(0.0, device=self.device)
        orth_loss = torch.tensor(0.0, device=self.device)
        if self.training and indices is not None:
            align_loss = self.infonce_loss(x_t_com[indices], x_v_com[indices], temp=self.tau)
            orth_loss = self.orthogonality_loss(x_v_uni[indices], x_v_com[indices]) +\
                    self.orthogonality_loss(x_t_uni[indices], x_t_com[indices])

        return F.normalize(x_v_uni), F.normalize(x_t_uni), F.normalize(x_shared), align_loss, orth_loss


class IntentDisentangler(nn.Module):
    def __init__(self, input_dim, latent_dim, n_intents, device, use_intent):
        super().__init__()
        if use_intent:
            self.encoder = nn.Sequential(nn.Linear(input_dim, latent_dim))
            self.projector = nn.Sequential(nn.Linear(latent_dim, latent_dim))
        else:
            self.encoder = nn.Identity()
            self.projector = nn.Identity()
        self.intent_book = nn.Parameter(torch.empty(n_intents, latent_dim))
        nn.init.xavier_uniform_(self.intent_book)

        self.n_intents = n_intents
        self.latent_dim = latent_dim
        self.device = device

    def intent_loss(self, temp=0.1):
        emb1 = F.normalize(self.intent_book, p=2, dim=-1)
        scores = torch.exp(torch.matmul(emb1, emb1.T) / temp)
        pos_sim = scores.diag()
        loss = -torch.log(1 / (torch.sum(scores, dim=1) - pos_sim)).mean()
        return loss

    def forward(self, x):
        h = self.encoder(x)
        codebook = self.intent_book  # [K, D]

        # update after accepted

        z = self.projector(z)
        return z


def get_knn_adj(embeddings, knn_k):
    context_norm = embeddings / torch.norm(embeddings, p=2, dim=-1, keepdim=True)
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    knn_val, knn_ind = torch.topk(sim, knn_k, dim=-1)
    indices0 = torch.arange(knn_ind.size(0)).unsqueeze(1).expand(-1, knn_k).to(embeddings.device)
    indices = torch.stack([indices0.flatten(), knn_ind.flatten()], dim=0)
    adj = torch.sparse_coo_tensor(indices, knn_val.flatten(), sim.size())
    return adj


def get_base_adj(ui_indices, n_users, n_items, device, vals=None):
    n_nodes = n_users + n_items
    adj_size = torch.Size((n_nodes, n_nodes))
    ui_indices[1] += n_users
    if vals is not None:
        ui_graph = torch.sparse_coo_tensor(ui_indices, vals, adj_size, device=device)
    else:
        ui_graph = torch.sparse_coo_tensor(ui_indices, torch.ones_like(ui_indices[0], dtype=torch.float32),
                                           adj_size, device=device)
    iu_graph = ui_graph.T
    base_adj = ui_graph + iu_graph
    return base_adj


def cal_norm_laplacian(adj):
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


class LightGCN(nn.Module):
    def __init__(self, n_layers):
        super(LightGCN, self).__init__()
        self.n_layers = n_layers

    def forward(self, ego_embeddings, norm_adj):
        adj = norm_adj
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        return all_embeddings


def get_user_co_knn_adj(ui_indices, knn_k, n_users, n_items):
    ui_adj = torch.sparse_coo_tensor(ui_indices, torch.ones_like(ui_indices[0], dtype=torch.float32),
                                     torch.Size((n_users, n_items)), device=ui_indices.device)
    iu_adj = ui_adj.T
    user_cooccur_adj = torch.sparse.mm(ui_adj, iu_adj).to_dense()
    user_cooccur_adj.fill_diagonal_(0)
    knn_val, knn_ind = torch.topk(user_cooccur_adj, knn_k, dim=-1)
    indices0 = torch.arange(knn_ind.size(0)).unsqueeze(1).expand(-1, knn_k).to(ui_indices.device)
    indices = torch.stack([indices0.flatten(), knn_ind.flatten()], dim=0)
    knn_val = F.softmax(knn_val, dim=-1).flatten()
    adj = torch.sparse_coo_tensor(indices, knn_val, torch.Size((n_users, n_users)), device=ui_indices.device)
    return adj

def get_item_co_knn_adj(ui_indices, knn_k, n_users, n_items):
    ui_adj = torch.sparse_coo_tensor(ui_indices, torch.ones_like(ui_indices[0], dtype=torch.float32),
                                     torch.Size((n_users, n_items)), device=ui_indices.device)
    iu_adj = ui_adj.T
    item_cooccur_adj = torch.sparse.mm(iu_adj, ui_adj).to_dense()
    item_cooccur_adj.fill_diagonal_(0)
    knn_val, knn_ind = torch.topk(item_cooccur_adj, knn_k, dim=-1)
    indices0 = torch.arange(knn_ind.size(0)).unsqueeze(1).expand(-1, knn_k).to(ui_indices.device)
    indices = torch.stack([indices0.flatten(), knn_ind.flatten()], dim=0)
    knn_val = F.softmax(knn_val, dim=-1).flatten()
    adj = torch.sparse_coo_tensor(indices, knn_val, torch.Size((n_items, n_items)), device=ui_indices.device)
    return adj