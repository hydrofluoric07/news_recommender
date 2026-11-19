import math
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from utils import get_item_user_time_dict
import collections
import faiss
import torch


# 定义用户活跃度权重
def get_user_activate_degree_dict(all_click_df):
    all_click_df_ = all_click_df.groupby('user_id')['click_article_id'].count().reset_index()

    # 用户活跃度归一化
    mm = MinMaxScaler()
    all_click_df_['click_article_id'] = mm.fit_transform(all_click_df_[['click_article_id']])
    user_activate_degree_dict = dict(zip(all_click_df_['user_id'], all_click_df_['click_article_id']))

    return user_activate_degree_dict

# UserCF算法
def usercf_sim(all_click_df, user_activate_degree_dict, save_path, use_cache=True):
    """
    用户相似性矩阵计算 + 缓存机制（支持版本路径）

    :param all_click_df: 用户点击日志
    :param user_activate_degree_dict: 用户活跃度字典
    :param save_path: 缓存路径（可为目录或具体文件名）
    :param use_cache: 是否使用缓存
    :return: 用户相似度矩阵 u2u_sim_
    """

    # === 路径处理：若为目录，拼接默认文件名 ===
    if os.path.splitext(save_path)[1] == '':
        save_path = os.path.join(save_path, 'usercf_u2u_sim.pkl')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # === 缓存加载逻辑 ===
    if use_cache and os.path.exists(save_path):
        print(f"[usercf_sim] ✅ 使用缓存文件：{save_path}")
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    print("[usercf_sim] 🚧 正在重新计算用户相似度矩阵...")

    # === 正式计算 ===
    item_user_time_dict = get_item_user_time_dict(all_click_df)
    u2u_sim = {}
    user_cnt = defaultdict(int)

    for item, user_time_list in tqdm(item_user_time_dict.items()):
        for u, click_time in user_time_list:
            user_cnt[u] += 1
            u2u_sim.setdefault(u, {})
            for v, click_time in user_time_list:
                if u == v:
                    continue
                u2u_sim[u].setdefault(v, 0)
                # 用户活跃度加权（可调）
                activate_weight = 100 * 0.5 * (user_activate_degree_dict[u] + user_activate_degree_dict[v])
                u2u_sim[u][v] += activate_weight / math.log(len(user_time_list) + 1)

    # === 归一化 ===
    u2u_sim_ = u2u_sim.copy()
    for u, related_users in u2u_sim.items():
        for v, wij in related_users.items():
            u2u_sim_[u][v] = wij / math.sqrt(user_cnt[u] * user_cnt[v])

    # === 保存相似度矩阵 ===
    with open(save_path, 'wb') as f:
        pickle.dump(u2u_sim_, f)

    print(f"[usercf_sim] ✅ 相似度矩阵已保存至：{save_path}")
    return u2u_sim_



# 基于用户的召回 u2u2i
def user_based_recommend(user_id, user_item_time_dict, u2u_sim, sim_user_topk, recall_item_num,
                         item_topk_click, item_created_time_dict, emb_i2i_sim):
    """基于用户的召回"""
    
    # 修改警告输出方式
    if user_id not in u2u_sim:
        if user_id % 1000 == 0:  # 每1000个用户才输出一次
            print(f"⚠️ 用户ID {user_id} 不在相似度矩阵中 (仅显示每1000个)")
        # 返回热门物品作为后备方案
        return [(item, -i-100) for i, item in enumerate(item_topk_click[:recall_item_num])]
    
    if user_id not in user_item_time_dict:
        print(f"⚠️ 用户 {user_id} 没有历史交互记录")
        # 返回热门物品作为后备方案
        return [(item, -i-100) for i, item in enumerate(item_topk_click[:recall_item_num])]
    
    # 历史交互
    user_item_time_list = user_item_time_dict[user_id]  # {item1: time1, item2: time2...}
    user_hist_items = set([i for i, t in user_item_time_list])  # 存在一个用户与某篇文章的多次交互， 这里得去重

    items_rank = {}
    for sim_u, wuv in sorted(u2u_sim[user_id].items(), key=lambda x: x[1], reverse=True)[:sim_user_topk]:
        for i, click_time in user_item_time_dict[sim_u]:
            if i in user_hist_items:
                continue
            items_rank.setdefault(i, 0)

            loc_weight = 1.0
            content_weight = 1.0
            created_time_weight = 1.0

            # 当前文章与该用户看的历史文章进行一个权重交互
            for loc, (j, click_time) in enumerate(user_item_time_list):
                # 点击时的相对位置权重
                loc_weight += 0.9 ** (len(user_item_time_list) - loc)
                # 内容相似性权重
                if emb_i2i_sim.get(i, {}).get(j, None) is not None:
                    content_weight += emb_i2i_sim[i][j]
                if emb_i2i_sim.get(j, {}).get(i, None) is not None:
                    content_weight += emb_i2i_sim[j][i]

                # 创建时间差权重
                created_time_weight += np.exp(0.8 * np.abs(item_created_time_dict[i] - item_created_time_dict[j]))

            items_rank[i] += loc_weight * content_weight * created_time_weight * wuv

    # 热度补全
    if len(items_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in items_rank:  # 修正：检查是否已在字典中，而不是检查是否在字典的items()中
                continue
            items_rank[item] = - i - 100  # 随便给个负数就行
            if len(items_rank) == recall_item_num:
                break

    items_rank = sorted(items_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

    return items_rank

# ============================
# ✅ UserCF相似度计算（传统相似用户）
# ============================
def generate_usercf_recall_dict(click_df, user_item_time_dict, u2u_sim, sim_user_topk,
                              recall_item_num, item_topk_click, item_created_time_dict, 
                              emb_i2i_sim, save_path='./cache/', use_cache=True):
    """生成基于用户的召回结果"""
    
    # 添加缓存路径
    cache_path = os.path.join(save_path, 'usercf_recall_dict.pkl')
    
    # 检查缓存
    if use_cache and os.path.exists(cache_path):
        print(f"[generate_usercf_recall_dict] ✅ 使用缓存：{cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    user_recall_items_dict = {}
    
    print("生成用户召回结果...")
    total_users = len(click_df['user_id'].unique())
    missing_users = 0
    
    for user_id in tqdm(click_df['user_id'].unique()):
        user_recall_items_dict[user_id] = []
        
        # 如果用户不在相似度矩阵中，记录并跳过
        if user_id not in u2u_sim:
            missing_users += 1
            continue
            
        # 获取相似用户及其相似度
        sim_users = u2u_sim[user_id]
        
        item_rank = {}
        for sim_user, sim_score in sim_users:
            # 获取相似用户的历史交互
            if sim_user not in user_item_time_dict:
                continue
                
            sim_user_items = user_item_time_dict[sim_user]
            for item_id, _ in sim_user_items:
                if item_id in item_rank:
                    continue
                    
                item_rank[item_id] = sim_score
                
        # 按照得分排序，取前N个
        item_rank_tuple = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]
        
        # 如果召回数量不足，用热门物品补充
        if len(item_rank_tuple) < recall_item_num:
            for i, item in enumerate(item_topk_click):
                if item not in dict(item_rank_tuple):
                    item_rank_tuple.append((item, -i-100))
                if len(item_rank_tuple) >= recall_item_num:
                    break
                    
        user_recall_items_dict[user_id] = item_rank_tuple
    
    # 打印相似度矩阵覆盖率统计
    coverage = (total_users - missing_users) / total_users
    print(f"\n[generate_usercf_recall_dict] 用户相似度矩阵覆盖率: {coverage:.4f}")
    print(f"总用户数: {total_users}, 缺失用户数: {missing_users}")
    
    # 保存结果
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(user_recall_items_dict, f)
    print(f"[generate_usercf_recall_dict] ✅ 召回结果已保存至: {cache_path}")
    
    return user_recall_items_dict


# ============================
# ✅ User Embedding 相似度计算（YouTubeDNN用户向量）
# ============================
def generate_ucercf_embedding_recall_dict(click_df, user_emb_dict, save_path, topk):
    user_list = []
    user_emb_list = []
    for user_id, user_emb in user_emb_dict.items():
        user_list.append(user_id)
        user_emb_list.append(user_emb)

    user_index_2_rawid_dict = {k: v for k, v in zip(range(len(user_list)), user_list)}
    user_emb_np = np.array(user_emb_list, dtype=np.float32)

    index = faiss.IndexFlatIP(user_emb_np.shape[1])
    index.add(user_emb_np)
    sim, idx = index.search(user_emb_np, topk)

    user_sim_dict = collections.defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(user_emb_np)), sim, idx)):
        target_raw_id = user_index_2_rawid_dict[target_idx]
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            rele_raw_id = user_index_2_rawid_dict[rele_idx]
            user_sim_dict[target_raw_id][rele_raw_id] = user_sim_dict.get(target_raw_id, {}).get(rele_raw_id, 0) + sim_value

    with open(os.path.join(save_path, 'youtube_u2u_sim.pkl'), 'wb') as f:
        pickle.dump(user_sim_dict, f)

    return user_sim_dict

# 使用Embedding的方式获取u2u的相似性矩阵
def u2u_embedding_sim(click_df, user_emb_dict, save_path='./cache/', topk=20, use_cache=True):
    """
    基于用户嵌入计算用户相似度
    """
    cache_path = save_path if save_path.endswith('.pkl') else os.path.join(save_path, 'youtube_u2u_sim.pkl')
    
    if use_cache and os.path.exists(cache_path):
        print(f"[u2u_embedding_sim] ✅ 加载用户相似度缓存: {cache_path}")
        with open(cache_path, 'rb') as f:
            u2u_sim = pickle.load(f)
    else:
        print("[u2u_embedding_sim] 计算用户相似度矩阵...")
        
        # 检查用户嵌入是否为空
        if not user_emb_dict:
            print("[u2u_embedding_sim] ⚠️ 用户嵌入字典为空！")
            return {}
            
        # 获取所有用户ID和对应的嵌入
        print(f"[u2u_embedding_sim] 用户嵌入数量: {len(user_emb_dict)}")
        
        # 转换嵌入格式
        all_user_ids = []
        user_embeddings = []
        for user_id, emb in user_emb_dict.items():
            # 确保嵌入是一维数组
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            if len(emb.shape) > 1:
                emb = emb.squeeze()
            
            all_user_ids.append(user_id)
            user_embeddings.append(emb)
        
        user_embeddings = np.array(user_embeddings, dtype=np.float32)
        print(f"[u2u_embedding_sim] 嵌入矩阵形状: {user_embeddings.shape}")
        
        # 归一化嵌入
        norms = np.linalg.norm(user_embeddings, axis=1, keepdims=True) # 计算L2范数
        norms = np.where(norms == 0, 1.0, norms)
        user_embeddings = user_embeddings / norms
        
        # 使用Faiss进行快速相似度计算
        dim = user_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(user_embeddings)
        
        # 计算相似度
        sim_scores, sim_idx = index.search(user_embeddings, topk + 1)
        
        # 构建用户相似度字典
        u2u_sim = {}
        for i, user_id in enumerate(all_user_ids):
            # 跳过第一个（自己）
            similar_users = [(all_user_ids[idx], float(score)) 
                           for idx, score in zip(sim_idx[i][1:], sim_scores[i][1:])]
            u2u_sim[user_id] = similar_users
        
        print(f"[u2u_embedding_sim] 计算完成，用户数: {len(u2u_sim)}")
        
        # 保存结果
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(u2u_sim, f)
    
    # 打印一些统计信息
    print(f"[u2u_embedding_sim] 相似度矩阵中的用户数: {len(u2u_sim)}")
    if len(u2u_sim) > 0:
        sample_user = next(iter(u2u_sim))
        print(f"[u2u_embedding_sim] 样例 - 用户{sample_user}的相似用户数: {len(u2u_sim[sample_user])}")
        # 打印一些样例相似度
        print("\n相似度样例:")
        for sim_user, sim_score in u2u_sim[sample_user][:3]:
            print(f"用户{sample_user} -> 用户{sim_user}: {sim_score:.4f}")
    
    return u2u_sim
