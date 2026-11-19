import os
import pickle
import pandas as pd
import warnings
import collections
from tqdm import tqdm

warnings.filterwarnings('ignore')

from data_processing import (
    get_all_click_df,
    get_item_info_df,
    prepare_dir,
    resolve_data_path,
)
from utils import get_user_item_time_dict, get_item_topk_click, get_item_info_dict
from models.recall_baseline import itemcf_sim, item_based_recommend, generate_user_recall_dict
from submission import submit

# 用于生成线上提交的流程（train + testA）
def main(sim_version='submit_v1', recall_version='submit_v1', use_cache=True):
    # 路径解析：环境变量 > 本地 ./data_raw/ > Kaggle 数据集挂载点
    data_path = resolve_data_path()
    save_path = prepare_dir(os.environ.get('NEWS_OUTPUT_DIR', './results/'))
    cache_dir = prepare_dir(os.environ.get('NEWS_CACHE_DIR', './cache/'))

    print("🚀 main() 启动，用于生成线上提交结果（train + testA）")
    print(f"📂 数据目录：{data_path}")
    print(f"📂 输出目录：{save_path}")
    print(f"📂 缓存目录：{cache_dir}")

    # Step 1：加载点击数据（包含 train 和 testA 用户）
    all_click_df = get_all_click_df(data_path=data_path, offline=False)
    print(f"✅ 数据加载完成，共 {len(all_click_df)} 条点击记录")

    # Step 2：计算物品相似度矩阵（版本化+缓存机制）
    print("📌 [Step 2] 计算物品相似度矩阵...")

    item_info_df = get_item_info_df(data_path)
    item_created_time_dict = get_item_info_dict(item_info_df)[2]

    sim_cache_file = os.path.join(cache_dir, f'itemcf_sim_{sim_version}.pkl')
    i2i_sim = itemcf_sim(all_click_df, item_created_time_dict, save_path=sim_cache_file, use_cache=use_cache)
    print("✅ 相似度矩阵准备完成")

    # Step 3：初始化用户召回字典
    print("📌 [Step 3] 初始化用户召回字典...")
    user_recall_items_dict = collections.defaultdict(dict)

    # Step 4：用户-文章时间序列
    print("📌 [Step 4] 计算用户-文章点击时间序列...")
    user_item_time_dict = get_user_item_time_dict(all_click_df)
    print(f"✅ 用户-文章时间序列计算完成，共 {len(user_item_time_dict)} 个用户")

    # Step 5~8：召回参数配置
    sim_item_topk = 10
    recall_item_num = 10
    print(f"📌 [Step 5] 相似物品数量：{sim_item_topk}")
    print(f"📌 [Step 6] 每个用户召回数量：{recall_item_num}")

    print("📌 [Step 7] 计算最热文章...")
    item_topk_click = get_item_topk_click(all_click_df, k=50)
    print(f"✅ 热门文章计算完成，前5篇为 {item_topk_click[:5]}")

    # Step 8：生成用户推荐列表（只对 testA 用户）
    print("📌 [Step 8] 加载测试集用户并生成推荐列表...")

    # 读取 testA 用户
    tst_click = pd.read_csv(data_path + 'testA_click_log.csv')
    tst_users_df = pd.DataFrame({'user_id': tst_click['user_id'].unique()})
    print(f"✅ 测试集用户加载完成，共 {len(tst_users_df)} 个用户")

    # 生成推荐缓存路径（支持版本命名）
    recall_cache_file = os.path.join(cache_dir, f'user_recall_items_{recall_version}.pkl')

    # 生成推荐字典（只为 testA 用户生成）
    user_recall_items_dict = generate_user_recall_dict(
        val_df=tst_users_df,
        user_item_time_dict=user_item_time_dict,
        i2i_sim=i2i_sim,
        sim_item_topk=10,
        recall_item_num=10,
        item_topk_click=item_topk_click,
        save_path=recall_cache_file,
        use_cache=use_cache
    )

    # Step 9：字典 → DataFrame
    print("📌 [Step 9] 推荐字典转 DataFrame...")
    user_item_score_list = []
    for user, items in tqdm(user_recall_items_dict.items()):
        for item, score in items:
            user_item_score_list.append([user, item, score])
    recall_df = pd.DataFrame(user_item_score_list, columns=['user_id', 'click_article_id', 'pred_score'])
    print(f"✅ 推荐 DataFrame 生成完成，共 {len(recall_df)} 条推荐记录")

    # Step 10：生成提交文件
    print("📌 [Step 10] 生成提交文件...")
    submit(recall_df, save_path, topk=5, model_name=f'itemcf_{recall_version}')
    print("✅ 提交文件生成完成！")

    print("🎉 main() 运行完成！")

if __name__ == "__main__":
    main(sim_version='baseline_v1', recall_version='baseline_v1', use_cache=False)
