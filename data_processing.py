import numpy as np
import pandas as pd
import pickle
import os
import faiss
import collections
from tqdm import tqdm

# === 数据 / 输出目录解析 ===
# 所有读取函数都用 data_path + 'xxx.csv' 拼接，所以这里返回的路径必须带结尾斜杠
_LOCAL_DATA_PATH = './data_raw/'
_KAGGLE_DATA_CANDIDATES = [
    '/kaggle/input/datasets/jiaweiwei0311/tianchinewsrec/',
    '/kaggle/input/tianchinewsrec/',  # 常见挂载形式，兜底
]
_REQUIRED_DATA_FILES = ('train_click_log.csv', 'testA_click_log.csv', 'articles.csv')


def _with_trailing_slash(path):
    return path if path.endswith(('/', '\\')) else path + '/'


def prepare_dir(path):
    """确保目录存在，并返回带结尾斜杠的路径（便于直接拼接文件名）。"""
    path = _with_trailing_slash(path)
    os.makedirs(path, exist_ok=True)
    return path


def resolve_data_path(data_path=None):
    """
    确定数据目录，优先级：
      1. 显式传入的 data_path
      2. 环境变量 NEWS_DATA_PATH
      3. 本地 ./data_raw/
      4. Kaggle 数据集挂载点
    返回带结尾斜杠的路径；都找不到时报错并列出尝试过的位置。
    """
    candidates = [
        data_path,
        os.environ.get('NEWS_DATA_PATH'),
        _LOCAL_DATA_PATH,
        *_KAGGLE_DATA_CANDIDATES,
    ]
    tried = []
    for candidate in candidates:
        if not candidate:
            continue
        candidate = _with_trailing_slash(candidate)
        tried.append(candidate)
        if all(os.path.exists(candidate + name) for name in _REQUIRED_DATA_FILES):
            return candidate
    raise FileNotFoundError(
        "找不到数据目录。已尝试：\n  - " + "\n  - ".join(tried) +
        "\n请把数据 csv 放到 ./data_raw/，或设置环境变量 NEWS_DATA_PATH 指向数据目录。"
    )


# debug模式：从训练集中划出一部分数据来调试代码
def get_all_click_sample(data_path='./data_raw/', sample_nums=10000):
    """
        训练集中采样一部分数据调试
        data_path: 原数据的存储路径
        sample_nums: 采样数目（这里由于机器的内存限制，可以采样用户做）
    """
    all_click = pd.read_csv(data_path + 'train_click_log.csv')
    all_user_ids = all_click.user_id.unique()

    np.random.seed(42)

    sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False)
    all_click = all_click[all_click['user_id'].isin(sample_user_ids)]

    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click


def get_all_click_df(data_path='./data_raw/', offline=True):
    # 自己在本地把已有数据切出一部分当“标准答案”，跑召回或推荐，再算指标
    if offline:
        all_click = pd.read_csv(data_path + 'train_click_log.csv')
    # 为testA生成预测
    else:
        trn_click = pd.read_csv(data_path + 'train_click_log.csv')
        tst_click = pd.read_csv(data_path + 'testA_click_log.csv')

        all_click = pd.concat([trn_click, tst_click], ignore_index=True)

    #去除完全重复的数据
    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click


# 读取文章的基本属性
def get_item_info_df(data_path) -> pd.DataFrame:
    item_info_df = pd.read_csv(data_path + 'articles.csv')

    # 为了方便与训练集中的click_article_id拼接，需要把article_id修改成click_article_id
    item_info_df = item_info_df.rename(columns={'article_id': 'click_article_id'})

    return item_info_df


# 读取文章的Embedding数据
def get_item_emb_dict(data_path, save_path):
    pkl_path = os.path.join(save_path, 'item_content_emb.pkl')

    # ✅ 如果文件已存在，直接读取返回
    if os.path.exists(pkl_path):
        print("🔄 已存在 pkl 文件，直接加载...")
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

    # 否则才计算并保存
    print("📥 正在读取 CSV 并计算 Embedding...")
    item_emb_df = pd.read_csv(os.path.join(data_path, 'articles_emb.csv'))

    # embedding 向量的归一化（按行单位化）
    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols])
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

    item_emb_dict = dict(zip(item_emb_df['article_id'], item_emb_np))
    with open(pkl_path, 'wb') as f:
        pickle.dump(item_emb_dict, f)

    return item_emb_dict

# 向量检索相似度计算
# topk指的是每个item, faiss搜索后返回最相似的topk个item
def embdding_sim(click_df, item_emb_df, save_path, topk):
    """
        基于内容的文章embedding相似性矩阵计算
        :param click_df: 数据表
        :param item_emb_df: 文章的embedding
        :param save_path: 保存路径
        :patam topk: 找最相似的topk篇
        return 文章相似性矩阵

        思路: 对于每一篇文章， 基于embedding的相似性返回topk个与其最相似的文章， 只不过由于文章数量太多，这里用了faiss进行加速
    """

    # 文章索引与文章id的字典映射
    item_idx_2_rawid_dict = dict(enumerate(item_emb_df['article_id'].values))

    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols].values, dtype=np.float32)
    # 向量进行单位化
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

    # 建立faiss索引
    item_index = faiss.IndexFlatIP(item_emb_np.shape[1])
    item_index.add(item_emb_np)
    # 相似度查询，给每个索引位置上的向量返回topk个item以及相似度
    sim, idx = item_index.search(item_emb_np, topk)  # 返回的是列表

    # 将向量检索的结果保存成原始id的对应关系
    item_sim_dict = collections.defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(item_emb_np)), sim, idx)):
        target_raw_id = item_idx_2_rawid_dict[target_idx]
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            rele_raw_id = item_idx_2_rawid_dict[rele_idx]
            item_sim_dict[target_raw_id][rele_raw_id] = item_sim_dict.get(target_raw_id, {}).get(rele_raw_id,
                                                                                                 0) + sim_value

    # 保存i2i相似度矩阵
    pickle.dump(item_sim_dict, open(save_path + 'emb_i2i_sim.pkl', 'wb'))

    return item_sim_dict