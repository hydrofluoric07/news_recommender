import math
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pickle
import os

from utils import get_user_item_time_dict

def itemcf_sim(df, item_created_time_dict, save_path, use_cache=True):
    """
        文章与文章之间的相似性矩阵计算
        :param df: 数据表
        :item_created_time_dict:  文章创建时间的字典
        return : 文章与文章的相似性矩阵
        思路: 基于物品的协同过滤(详细请参考上一期推荐系统基础的组队学习)， 在多路召回部分会加上关联规则的召回策略
        可自动缓存和复用已有结果
    """
    # 如果是目录路径，就拼接默认文件名
    if os.path.splitext(save_path)[1] == '':
        # 没有扩展名，说明是目录
        save_path = os.path.join(save_path, 'itemcf_i2i_sim.pkl')

    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 缓存判断
    if use_cache and os.path.exists(save_path):
        print(f"[itemcf_sim] ✅ 使用缓存文件：{save_path}")
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    # 否则重新计算
    print(f"[itemcf_sim] 重新计算相似度矩阵...")

    user_item_time_dict = get_user_item_time_dict(df)

    # 计算物品相似度
    i2i_sim = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        # 在基于商品的协同过滤优化的时候可以考虑时间因素
        # 在基于商品的协同过滤优化的时候可以考虑时间因素
        for loc1, (i, i_click_time) in enumerate(item_time_list):
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for loc2, (j, j_click_time) in enumerate(item_time_list):
                if i == j:
                    continue
                # 正向/反向点击顺序区分（位置关系）
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                # 位置权重（点击顺序越近，越相关）其中的参数可以调节
                loc_weight = loc_alpha * (0.9 ** (np.abs(loc2 - loc1) - 1))
                # 点击时间相近权重（可理解为 session 内更相关）其中的参数可以调节
                click_time_weight = np.exp(0.7 ** np.abs(i_click_time - j_click_time))
                # 文章发布时间相近权重（防止跨年代推荐）其中的参数可以调节
                created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))

                i2i_sim[i].setdefault(j, 0)
                # 考虑多种因素的权重计算最终的文章之间的相似度
                i2i_sim[i][j] += loc_weight * click_time_weight * created_time_weight / math.log(
                    len(item_time_list) + 1)

    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])

    # 将得到的相似性矩阵保存到本地
    with open(save_path, 'wb') as f:
        pickle.dump(i2i_sim_, f)
    print(f"[itemcf_sim] 相似度矩阵已保存至：{save_path}")

    return i2i_sim_


# 基于商品的召回i2i
def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click,
                         item_created_time_dict, emb_i2i_sim):
    """
        基于文章协同过滤的召回
        :param user_id: 用户id
        :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: {item1: time1, item2: time2..}...}
        :param i2i_sim: 字典，文章相似性矩阵
        :param sim_item_topk: 整数， 选择与当前文章最相似的前k篇文章
        :param recall_item_num: 整数， 最后的召回文章数量
        :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全
        :param emb_i2i_sim: 字典基于内容embedding算的文章相似矩阵

        return: 召回的文章列表 {item1:score1, item2: score2...}

    """
    # 获取用户历史交互的文章
    user_hist_items = user_item_time_dict[user_id]
    
    # 提取用户历史交互的物品ID，转为集合以加速查找
    hist_item_ids = set([x[0] for x in user_hist_items])

    i2i_sim = {
        i:sorted(related, key=lambda x:x[1], reverse=True)[:sim_item_topk]
        for i, related in i2i_sim
    }
    item_rank = {}
    for loc, (i, click_time) in enumerate(user_hist_items):
        for j, wij in i2i_sim[i].items():
            # 检查j是否在用户历史交互物品中
            if j in hist_item_ids:
                continue

            # 文章创建时间差权重
            created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
            # 相似文章和历史点击文章序列中历史文章所在的位置权重
            loc_weight = (0.9 ** (len(user_hist_items) - loc))

            content_weight = 1.0
            if emb_i2i_sim.get(i, {}).get(j, None) is not None:
                content_weight += emb_i2i_sim[i][j]
            if emb_i2i_sim.get(j, {}).get(i, None) is not None:
                content_weight += emb_i2i_sim[j][i]

            item_rank.setdefault(j, 0)
            item_rank[j] += created_time_weight * loc_weight * content_weight * wij

    # 不足10个，用热门商品补全
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank:  # 填充的item应该不在原来的列表中
                continue
            item_rank[item] = - i - 100  # 随便给个负数就行
            if len(item_rank) == recall_item_num:
                break

    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

    return item_rank

# 使用原始 itemcf 矩阵 + emb 权重融合；
def generate_itemcf_recall_dict(val_df,
                                     user_item_time_dict,
                                     i2i_sim,
                                     sim_item_topk,
                                     recall_item_num,
                                     item_topk_click,
                                     item_created_time_dict,
                                     emb_i2i_sim=None,
                                     save_path='cache/user_recall_itemcf.pkl',
                                     use_cache=True):
    """
    基于 ItemCF（带 Emb 权重融合）的召回列表生成
    """
    if os.path.splitext(save_path)[1] == '':
        save_path = os.path.join(save_path, 'user_recall_itemcf.pkl')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if use_cache and os.path.exists(save_path):
        print(f"[generate_user_recall_dict_itemcf] ✅ 使用缓存：{save_path}")
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    print("[generate_user_recall_dict_itemcf] 🚀 正在生成 ItemCF 召回列表...")
    user_recall_items_dict = {}
    for user in tqdm(val_df['user_id'].unique()):
        rec_items = item_based_recommend(
            user,
            user_item_time_dict,
            i2i_sim,
            sim_item_topk,
            recall_item_num,
            item_topk_click,
            item_created_time_dict,
            emb_i2i_sim
        )
        user_recall_items_dict[user] = rec_items

    with open(save_path, 'wb') as f:
        pickle.dump(user_recall_items_dict, f)

    print(f"[generate_user_recall_dict_itemcf] ✅ 召回列表保存成功：{save_path}")
    return user_recall_items_dict

# 仅使用 embedding 相似度作为召回通道
def generate_itemcf_embedding_recall_dict(val_df,
                                         emb_i2i_sim,
                                         user_item_time_dict,
                                         sim_item_topk,
                                         recall_item_num,
                                         item_topk_click,
                                         item_created_time_dict=None,
                                         save_path='cache/user_recall_embedding.pkl',
                                         use_cache=True):
    """
    基于 Embedding 相似度的召回列表生成
    """
    if os.path.splitext(save_path)[1] == '':
        save_path = os.path.join(save_path, 'user_recall_embedding.pkl')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if use_cache and os.path.exists(save_path):
        print(f"[generate_user_recall_dict_embedding] ✅ 使用缓存：{save_path}")
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    print("[generate_user_recall_dict_embedding] 🚀 正在生成 Embedding 召回列表...")
    user_recall_items_dict = {}
    for user in tqdm(val_df['user_id'].unique()):
        rec_items = item_based_recommend(
            user,
            user_item_time_dict,
            emb_i2i_sim,
            sim_item_topk,
            recall_item_num,
            item_topk_click,
            item_created_time_dict,
            emb_i2i_sim  # 注意这里传入自身即可，不影响推荐函数中使用权重逻辑
        )
        user_recall_items_dict[user] = rec_items

    with open(save_path, 'wb') as f:
        pickle.dump(user_recall_items_dict, f)

    print(f"[generate_user_recall_dict_embedding] ✅ 召回列表保存成功：{save_path}")
    return user_recall_items_dict





