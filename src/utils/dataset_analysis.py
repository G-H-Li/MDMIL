import numpy as np


def user_sparse_group_analysis(inter_matrix, n_groups=4):
    """
    根据输入的UI交互矩阵生成按照交互数量划分的用户组，输出交互数量区间和区间内用户id
    例如 n_groups=5, 则输出5个用户组，分别为交互数量最多的20%用户组、20%-40%、40%-60%、60%-80%、80%-100%用户组
    保证每个分组的交互数量基本一致
    Args:
        n_groups:
        inter_matrix:

    Returns: Dict

    """
    # 每个用户的交互数量
    user_inter_counts = inter_matrix.sum(axis=1).astype(int).flatten()
    user_inter_counts = np.array(user_inter_counts).squeeze()
    total_interactions = user_inter_counts.sum()
    target_per_group = total_interactions / n_groups

    sorted_idx = np.argsort(user_inter_counts)
    sorted_counts = user_inter_counts[sorted_idx]

    groups = {}
    current_group = []
    current_sum = 0
    group_id = 1
    group_start = 0
    group_end = 0

    for uid, cnt in zip(sorted_idx, sorted_counts):
        if current_sum == 0:
            group_start = int(cnt)
        current_group.append(uid)
        current_sum += cnt

        if current_sum >= target_per_group and group_id < n_groups:
            group_end = int(cnt)
            groups[(group_start, group_end)] = current_group
            group_id += 1
            current_group = []
            current_sum = 0

    # 剩下的用户放最后一组
    if current_group:
        groups[(group_end, int(user_inter_counts.max()))] = current_group

    return groups


def item_popularity_group_analysis(inter_matrix, n_groups=4):
    """
    根据输入的UI交互矩阵生成按照交互数量划分的物品组，输出交互数量区间和区间内物品id
    例如 n_groups=5, 则输出5个物品组，分别为交互数量最多的20%物品组、20%-40%、40%-60%、60%-80%、80%-100%物品组
    保证每个分组的流行度基本一致
    Args:
        n_groups:
        inter_matrix:

    Returns: Dict

    """
    # 每个用户的交互数量
    item_inter_counts = inter_matrix.sum(axis=0).astype(int).flatten()
    item_inter_counts = np.array(item_inter_counts).squeeze()
    total_interactions = item_inter_counts.sum()
    target_per_group = total_interactions / n_groups

    sorted_idx = np.argsort(item_inter_counts)
    sorted_counts = item_inter_counts[sorted_idx]

    groups = {}
    current_group = []
    current_sum = 0
    group_id = 1
    group_start = 0
    group_end = 0

    for iid, cnt in zip(sorted_idx, sorted_counts):
        if current_sum == 0:
            group_start = int(cnt)
        current_group.append(iid)
        current_sum += cnt

        if current_sum >= target_per_group and group_id < n_groups:
            group_end = int(cnt)
            groups[(group_start, group_end)] = current_group
            group_id += 1
            current_group = []
            current_sum = 0

    # 剩下的用户放最后一组
    if current_group:
        groups[(group_end, int(item_inter_counts.max()))] = current_group

    return groups
