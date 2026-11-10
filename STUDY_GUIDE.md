# 新闻推荐项目导读：每一步打印什么、该看到什么

> 这份导读配合 `main.py` 的线上链路 + `offline_evaluation_multi.py` 的线下链路使用。
> 每一步都只做三件事：**打印结构、看一条真实样本、验证一条不变量**。
> 目标不是"读完所有代码"，而是能画出数据流、预测改参数后的变化。

---

## 0. 开始之前（先解决 4 个会让程序起不来的问题）

### 0.1 当前仓库缺数据

`data_raw/`、`cache/`、`results/` 都不在版本库中（`.gitignore` 忽略了它们）。需要从比赛页下载并放成：

```
data_raw/
├── train_click_log.csv
├── testA_click_log.csv
├── articles.csv
└── articles_emb.csv
```

公开版数量级参考：训练点击 **200 万+ 行**、testA 点击 **数十万行**、文章 **20 万+ 篇**。
如果你看到的是几百行或者缺文件，说明放错位置了。行数/列数以你实际下载的版本为准，本机无法代你核对。

### 0.2 解释器与依赖

用系统默认的 `python`（scoop 3.14）会直接 `ModuleNotFoundError: pandas`。本机 conda 环境实测：

| 环境 | pandas | torch | faiss |
|---|---|---|---|
| `miniconda3\envs\funrec` | 2.0.3 | ✗ | ✗ |
| `miniconda3\envs\d2l` | 2.0.3 | ✓ | ✗ |
| `miniconda3\envs\dl_env` | 2.3.3 | ✓ | ✗ |
| 系统 `python`（scoop 3.14） | ✗ | ✗ | ✗ |

`data_processing.py` 顶部就 `import faiss`（第 5 行），所以哪怕只走 `main.py` 的 ItemCF 链路，也必须先装：

```powershell
& "C:\Users\admin\miniconda3\envs\funrec\python.exe" -m pip install faiss-cpu
```

不想装的话，把 `import faiss` 从模块顶部移到 `embdding_sim()` 函数内部（延迟导入），主链路就用不到它了。

**建议**：走 `main.py` 这条链路用 `funrec`；后面跑多路召回（torch 相关）用 `d2l` 或 `dl_env`。

### 0.3 启动依赖修复（✅ 已在当前工作区应用）

```python
# main.py:11
- from utils import get_user_item_time, get_item_topk_click
+ from utils import get_user_item_time_dict, get_item_topk_click

# main.py:44 和 main.py:71
- get_user_item_time(all_click_df)
+ get_user_item_time_dict(all_click_df)

# models/recall_baseline.py:7
- from utils import get_user_item_time
+ from utils import get_user_item_time_dict

# models/recall_baseline.py:34
- user_item_time_dict = get_user_item_time(df)
+ user_item_time_dict = get_user_item_time_dict(df)

# models/recall_baseline.py:91  热门补全的成员判断永远为假
- if item in item_rank.items():
+ if item in item_rank:

# models/recall_baseline.py:82  用元组列表判断 id，永远为假
- if j in user_hist_items:
+ hist_item_ids = {item for item, _ in user_hist_items}
+ if j in hist_item_ids:

# data_processing.py:45  注解引用了未导入的名字
- def get_item_info_df(data_path) -> DataFrame:
+ def get_item_info_df(data_path) -> pd.DataFrame:
```

`utils.py` 里只有 `get_user_item_time_dict`，没有 `get_user_item_time`，不修就是 `ImportError`；`-> DataFrame` 在**导入时**就 `NameError`（注解是当场求值的）。

### 0.4 建目录

`submission.submit()` 不做 `makedirs`，`results/` 不存在时 `to_csv` 会直接报错：

```powershell
python -c "import os; [os.makedirs(d, exist_ok=True) for d in ['cache','results']]"
```

### 0.5 学习模式 vs 全量模式

全量算 ItemCF 相似度是 O(用户历史长度²) 的循环，200 万+ 点击要跑很久。**先用采样把流程走通**：

```python
from data_processing import get_all_click_sample
df = get_all_click_sample('./data_raw/', sample_nums=2000)   # 注意：参数是"用户数"，不是点击行数
```

下文所有代码里的 `df` 都指"采样版或全量版"，先小后大。注意 `get_all_click_sample` 只从 train 采样，不包含 testA 用户。

---

## 1. 全局数据流（先记住这张图）

```
train_click_log.csv ─┐
                     ├─(offline=False)─> all_click_df              # 线上：含 testA
testA_click_log.csv ─┘                        │
                                              ├─> user_item_time_dict   {user: [(item, ts)...]}
                                              ├─> item_topk_click       [热门 item × 50]
                                              └─> i2i_sim               {item: {item: sim}}
                                                        │
testA user 列表 ────────────────────────────────────────┤
                                                        ▼
                                        user_recall_items_dict  {user: [(item, score) × 10]}
                                                        │
                                                        ▼
                                                     recall_df  [user_id, click_article_id, pred_score]
                                                        │  submit(topk=5)
                                                        ▼
                                       results/itemcf_<version>_<MM-DD>.csv
                                       user_id, article_1 ... article_5
```

线下链路只差第一行：用 `offline=True` 只读 train，然后自己切出每个用户的末次点击当答案。

**只要你能独立默写出这 7 层结构以及每层怎么变，这个项目的骨架就懂了。**

---

## 2. 逐步导读

新建一个 `walkthrough.py` 放在项目根目录，按下面的步骤一段段加代码再运行。
每一步的"该看到"里，**凡是标 `【不变量】` 的必须成立**；不成立说明你理解或数据有问题，先停下来查。

```python
# walkthrough.py 顶部
import pandas as pd
from data_processing import get_all_click_sample, get_all_click_df, get_item_info_df
from utils import get_item_info_dict, get_user_item_time_dict, get_item_topk_click
from models.recall_baseline import itemcf_sim, item_based_recommend, generate_user_recall_dict
from submission import submit

df = get_all_click_sample('./data_raw/', sample_nums=2000)
```

---

### S1 点击日志：先认识"原料"

**目的**：搞清一行数据代表什么、线上和线下的数据范围差在哪。

**看哪里**：`data_processing.py:30`

```python
off = get_all_click_df('./data_raw/', offline=True)
on  = get_all_click_df('./data_raw/', offline=False)

print('offline shape', off.shape)
print('online  shape', on.shape)
print('columns', on.columns.tolist())
print('dtypes\n', on.dtypes)
print('users/items', on.user_id.nunique(), on.click_article_id.nunique())
print('ts', pd.to_datetime(on.click_timestamp.min(), unit='ms'),
           pd.to_datetime(on.click_timestamp.max(), unit='ms'))
print('nan', on.isna().sum().sum())
```

**该看到**：

- 列里有 `user_id / click_article_id / click_timestamp`（还有环境、设备、地区等 9 列，代码只用前 3 列）。
- `click_timestamp` 是**毫秒时间戳**（约 1.5e12 量级），转成日期在 2019 年前后。
- `【不变量】` `on.shape[0] > off.shape[0]`，差值是去重后的 testA 点击。
- `【不变量】` `on.user_id.nunique() > off.user_id.nunique()`，多出来的是 testA 用户。
- `【不变量】` 没有 NaN。
- 唯一点击文章数远小于 `articles.csv` 的行数——大部分文章没人点过。这是后面"召回空间"的直觉来源。

**自问**：如果我把 `offline` 传成 `False` 去算线下指标，会发生什么？（答案：testA 无标签，根本没东西可对；而且如果硬拿 testA 自己的点击当历史去预测它自己，就是泄露。）

---

### S2 文章属性：从 DataFrame 变成 O(1) 查找表

**目的**：理解三个 dict 的用途，以及 `created_at_ts` 为什么必须先归一化。

**看哪里**：`data_processing.py:45`、`utils.py:95`

```python
info = get_item_info_df('./data_raw/')
print('info shape', info.shape, info.columns.tolist())

type_d, words_d, time_d = get_item_info_dict(info)

print('normalized range', info.created_at_ts.min(), info.created_at_ts.max())
print('dict lens', len(type_d), len(words_d), len(time_d))
aid = next(iter(time_d))
print('sample', aid, type_d[aid], words_d[aid], time_d[aid])
```

**该看到**：

- `articles.csv` 是 4 列：`click_article_id / category_id / created_at_ts / words_count`。
- 调用 `get_item_info_dict` **之后**，`info.created_at_ts` 的最小值变成 `0.0`、最大值变成 `1.0`——函数原地改写了你传进去的 DataFrame（副作用！）。
- `【不变量】` 三个 dict 长度相等，且等于文章数。
- `【不变量】` `time_d` 的所有值都落在 `[0, 1]`。
- `【不变量】` key 集合与 `info.click_article_id` 完全一致。

**自问**：为什么不用原始毫秒时间戳？去 `models/itemcf_recall.py:57` 看 `np.exp(0.8 ** |t_i - t_j|)`：如果差值以毫秒计是 1e9 量级，`0.8 ** 1e9` 下溢成 0，权重全灭。归一化后差值在 `[0,1]`，权重才温和。

---

### S3 用户点击序列：召回真正的输入

**目的**：确认数据结构、排序方向、元素类型。

**看哪里**：`utils.py:47`

```python
uit = get_user_item_time_dict(df)
print('users', len(uit), 'unique', df.user_id.nunique())

u = next(iter(uit))
seq = uit[u]
print('user', u, 'len', len(seq))
print('first 5', seq[:5])
print('elem types', type(seq), type(seq[0]), type(seq[0][0]))
print('ascending?', all(a[1] <= b[1] for a, b in zip(seq, seq[1:])))
print('matches raw', len(seq) == (df.user_id == u).sum())
```

**该看到**：

- 值是 **list of tuple**：`[(item, ts), (item, ts), ...]`，不是 docstring 里写的 `{item: time}`。
- 时间**升序**（函数先 `sort_values('click_timestamp')` 再 groupby）。
- 元素是 `numpy.int64` 之类的 numpy 标量，不是 Python int。
- `【不变量】` dict 长度 == 唯一用户数。
- `【不变量】` 每个用户的序列长度 == 他在点击表里的行数。

**自问**：`item_based_recommend` 里 `for j, wij in ...` 之后为什么 `loc` 是"位置"而不是 item？（因为后面的 `0.9 ** (len - loc)` 是位置衰减，越近的点击越重要。）

---

### S4 物品相似度：整个 ItemCF 的核心

**目的**：亲手算一次，看到共现 → 权重 → 归一化之后分数的形态。

**看哪里**：`models/recall_baseline.py:10`

```python
i2i = itemcf_sim(df, save_path='./cache/walk_sim.pkl', use_cache=False)
print('items in sim', len(i2i))

i = next(iter(i2i))
top5 = sorted(i2i[i].items(), key=lambda x: -x[1])[:5]
print('item', i, 'top5', top5)

vals = [v for d in list(i2i.values())[:200] for v in d.values()]
print('score range', min(vals), max(vals))

j = top5[0][0]
print('symmetry', i2i[i][j], i2i[j].get(i))
print('empty dicts', sum(1 for d in i2i.values() if not d))
```

**该看到**：

- `【不变量】` `len(i2i) == df.click_article_id.nunique()`（每个被点过的文章都有一行，哪怕为空）。
- 分数 `wij > 0`；常见情况下 `<= 1`，**但不是必然**：`item_cnt` 统计的是"点击次数"而不是"去重用户数"，同一用户反复点同一篇时（共现对累加多次），相似度可以超过 1。看到大于 1 不要以为代码错了。
- `【不变量】` 对称：`i2i[i][j] ≈ i2i[j][i]`（两个方向在同一个用户循环里按同样规则累加，再除以同样的 `sqrt(cnt_i*cnt_j)`）。到加权版 `itemcf_recall.py` 里这个对称性会被打破（`loc_alpha` 区分正向/反向点击顺序）。
- 会有少量空 dict：只在全站被点过一次、且该用户没有别的点击的文章。
- 校验一下直觉：`i2i[i]` 的 top5 里，`category_id` 和 `i` 相同的是不是居多？打印出来对比。

**自问**：`wij / sqrt(cnt_i * cnt_j)` 这一步在干什么？（惩罚热门文章的"万能相似"，否则热门文章和谁都像。）`1 / log(len(seq) + 1)` 呢？（长序列的共现更不可信，降权。）

**注意**：

- 这里传进去的 `item_created_time_dict` 是**死参数**，`recall_baseline.itemcf_sim` 函数体里完全没用它。真正用时间权重的是 `models/itemcf_recall.py:10` 那一版。
- **耗时**：全量 200 万+ 点击上算一次，可能要十几分钟到几十分钟；先用 2000 用户的采样版（S0.5）把逻辑跑通。

---

### S5 热门榜：补全和冷启动的兜底

**看哪里**：`utils.py:137`

```python
hot = get_item_topk_click(df, k=50)
print(type(hot), len(hot), list(hot[:5]))
print(df.click_article_id.value_counts().head(5))
```

**该看到**：

- `【不变量】` `len(hot) == 50`（数据里至少 50 篇被点过）。
- 类型是 pandas `Index`，不是 list。
- `hot[:5]` 的顺序 == `value_counts().head(5)` 的 index 顺序。

---

### S6 单个用户的召回：把分数拆开看

**目的**：这是全项目最值得逐行读的函数，10 行逻辑决定了最终 5 个推荐位。

**看哪里**：`models/recall_baseline.py:63`

```python
u = list(uit.keys())[0]
print('history len', len(uit[u]), uit[u][:5])

rec = item_based_recommend(u, uit, i2i, 10, 10, hot)
print('recall len', len(rec))
for item, score in rec:
    print(item, round(float(score), 6), '| in history:', item in {x[0] for x in uit[u]})

print('negative count', sum(1 for _, s in rec if s < 0))
print('descending?', all(a[1] >= b[1] for a, b in zip(rec, rec[1:])))
```

**该看到**：

- `【不变量】` 返回长度 == 10（除非该用户历史覆盖了全部候选，实际不会发生）。
- `【不变量】` 返回的 item **都不在他的历史里**。
- 分数降序；负分只可能来自热门补全（`-i-100`，i 是热门榜下标，所以是 -100、-101、…）。
- `negative count > 0` 说明这个用户的 i2i 候选不足 10 个，被热门补齐了。
- 对同一个用户手动复算一条：取历史里最后一篇 `i_last`，看 `i2i[i_last]` 的 top10，应该和输出有交集。

**自问**：如果用户历史里有 30 篇，为什么不是召回 30×10=300 个候选？（会合并同一个 j 的分数，最后再截断到 10。）热门补全为什么要给负分？（保证永远排在真实相似度后面。）

---

### S7 批量召回：一致性检查

**看哪里**：`models/recall_baseline.py:101`

```python
users_df = pd.DataFrame({'user_id': df.user_id.unique()})   # 先只取 200 个用户练手
users_df = users_df.head(200)

recall = generate_user_recall_dict(
    val_df=users_df, user_item_time_dict=uit, i2i_sim=i2i,
    sim_item_topk=10, recall_item_num=10, item_topk_click=hot,
    save_path='./cache/walk_recall.pkl', use_cache=False)

from collections import Counter
print('users', len(recall))
print('len distribution', Counter(len(v) for v in recall.values()))
print('coverage of items', len({i for v in recall.values() for i, _ in v}))
print('sample user', list(recall.items())[0])
```

**该看到**：

- `【不变量】` `len(recall) == 200`，且**每个用户都恰好 10 条**。
- 物品覆盖率（被推荐过的不重复文章数）通常是几千级，远小于文章总数——召回多样性有限，这是后面加多路召回的原因之一。

**自问**：如果把 `sim_item_topk` 从 10 调到 3，覆盖率会变大还是变小？先猜，再跑。

---

### S8 召回字典 → DataFrame

**看哪里**：`main.py:81-87`

```python
rows = [[u, i, s] for u, lst in recall.items() for i, s in lst]
recall_df = pd.DataFrame(rows, columns=['user_id', 'click_article_id', 'pred_score'])
print(recall_df.shape)
print(recall_df.groupby('user_id').size().unique())   # 应该只有 [10]
print(recall_df.isna().sum().to_dict())
print(recall_df.dtypes.to_dict())
```

**该看到**：

- `【不变量】` 行数 == 用户数 × 10。
- `【不变量】` `groupby('user_id').size().unique()` 只有一个值 10。
- 无 NaN。

---

### S9 生成提交文件：透视这一步最容易看懵

**看哪里**：`submission.py:7`

```python
submit(recall_df, './results/', topk=5, model_name='walkthrough')

import glob
f = sorted(glob.glob('./results/walkthrough_*.csv'))[-1]
sub = pd.read_csv(f)
print(f)
print(sub.shape, sub.columns.tolist())
print(sub.head())
print('nan', sub.isna().sum().sum(), 'users', sub.user_id.nunique())
```

**该看到**：

- 文件名带当天日期：`walkthrough_MM-DD.csv`。
- 列恰好是 `['user_id', 'article_1', 'article_2', 'article_3', 'article_4', 'article_5']`。
  （内部 `rank` 是 float，`1.0` 能匹配 `rename` 里的整数 key，因为 `hash(1) == hash(1.0)`；这一点我本机实测过。）
- 行数 == 用户数，无 NaN。
- `submit` 里有个 `assert tmp.min() >= topk`：任何用户召回不足 5 条会直接抛异常。这就是它"安全阀"的作用。

**自问**：这一步丢掉了什么信息？（第 6~10 名的分数和顺序全丢了；如果要做重排/融合，必须在 `submit` 之前做。）

---

### S10 线下评估：为什么这个分数可信

**目的**：理解"切分 + 算指标"，并亲眼看到"不能用全量算相似度"的原因（泄露）。

**看哪里**：`offline_evaluation_multi.py:28`、`offline_evaluation_multi.py:43`、`recall_evaluation.py:4`

为了避免 import 到 torch/faiss，这里把指标函数内联（它们只有几行）：

```python
# 练手时先用采样版，全量同样很慢：
# off = get_all_click_sample('./data_raw/', sample_nums=2000)
off = get_all_click_df('./data_raw/', offline=True)
off = off.sort_values(['user_id', 'click_timestamp'])
val_df = off.groupby('user_id').tail(1)
trn_df = off.drop(val_df.index)

uit_tr = get_user_item_time_dict(trn_df)          # 只用 train 的历史
hot_tr = get_item_topk_click(trn_df, k=50)        # 只用 train 的热门
i2i_tr = itemcf_sim(trn_df, save_path='./cache/walk_sim_off.pkl', use_cache=False)

users = pd.DataFrame({'user_id': val_df.user_id.unique()}).head(2000)
recall_tr = generate_user_recall_dict(
    users, uit_tr, i2i_tr, 10, 10, hot_tr,
    save_path='./cache/walk_recall_off.pkl', use_cache=False)

def recall_at_k(recall, val, k):
    gt = dict(zip(val.user_id, val.click_article_id))
    hit = sum(1 for u, lst in recall.items()
              if u in gt and gt[u] in [i for i, _ in lst[:k]])
    return hit / len(recall)

def mrr_at_k(recall, val, k=5):
    gt = dict(zip(val.user_id, val.click_article_id))
    total = 0.0
    for u, lst in recall.items():
        if u not in gt:
            continue
        for rank, (item, _) in enumerate(lst[:k], start=1):
            if item == gt[u]:
                total += 1.0 / rank
                break
    return total / len(recall)

print('Recall@10', recall_at_k(recall_tr, val_df, 10))
print('MRR@5    ', mrr_at_k(recall_tr, val_df, 5))
```

**该看到**：

- `【不变量】` `len(val_df) == off.user_id.nunique()`，即每个用户恰好 1 条验证点击。
- `【不变量】` `len(trn_df) + len(val_df) == len(off)`。
- `【不变量】` `val_df` 的每一条都是该用户 `click_timestamp` 最大的那条。
- Recall@10 明显大于 MRR@5；单路 ItemCF 的 MRR@5 大致在 **0.1~0.25** 区间。你仓库 README 记录的多路融合结果是**离线 0.24 / 线上 0.12**——线上低一半是正常的（不同用户群、文章更新、样本量差异）。
- MRR 的读法：命中第 1 名贡献 1.0，第 2 名 0.5，第 5 名 0.2；所以 0.2 大致意味着"约两成用户的第一名就猜中"这个量级。

**亲手验证泄露**：把 `i2i_tr` 换成用 `off` 全量算的 `i2i`，`uit_tr` 换成 `uit`（含 val 那条），再评估一次。指标会**变高**——因为答案已经参与了相似度和历史统计。这就是为什么线下必须严格用切分后的 `trn_df`。

**自问**：`offline_evaluation_multi.py:377` 把 `train_df + val_df` 一起喂给 YouTubeDNN，这算不算泄露？（算，那一路的线下指标偏乐观，不能和 ItemCF 那几行横着比。）

---

### S11 多路融合：归一化才是关键

**看哪里**：`offline_evaluation_multi.py:72`（真实实现）、`online.py:188`（同一套逻辑）

内联一个最小版本，用两路"同一份结果、分数整体缩放"来演示：

```python
def norm(lst):
    if len(lst) < 2:
        return lst
    lo, hi = lst[-1][1], lst[0][1]
    if hi <= lo:
        return [(i, 1.0) for i, _ in lst]
    return [(i, (s - lo) / (hi - lo)) for i, s in lst]

def combine(multi, weights, topk=10):
    acc = {}
    for name, d in multi.items():
        w = weights.get(name, 1.0)
        for u, lst in d.items():
            for item, s in norm(lst):
                acc.setdefault(u, {})
                acc[u][item] = acc[u].get(item, 0) + w * s
    return {u: sorted(d.items(), key=lambda x: -x[1])[:topk] for u, d in acc.items()}

u0 = list(recall_tr)[0]
copy_ = {u: [(i, s * 0.5) for i, s in lst] for u, lst in recall_tr.items()}
merged = combine({'a': recall_tr, 'b': copy_}, {'a': 1.0, 'b': 0.0})
print('original', recall_tr[u0][:5])
print('merged  ', merged[u0][:5])
```

**该看到**：

- 第二路整体乘了 0.5，但**归一化后完全一样**，所以权重给 0 时合并结果与原始完全一致。
- `【不变量】` 合并后每个用户的 item 顺序与原始一致（在这个构造下）。
- 真实场景里不同召回的分数尺度差很大（i2i 是 0~1、UserCF 是计数、YouTubeDNN 是内积），**不归一化直接相加 = 分数大的那一路霸榜**。

**注意**：`combine_recall_results` 会**原地修改**你传进去的每个召回字典（把值替换成归一化后的列表）。调试时先 `copy.deepcopy`。

---

### S12 线上 vs 线下：只差"是否包含 testA"

最后跑一次完整的 `main.py`（修 bug 之后），对照两边的产物：

| 维度 | 线下 | 线上 |
|---|---|---|
| 输入 | `train_click_log` | `train + testA` |
| 相似度统计范围 | 切分后的 train | 全部已知点击 |
| 用户历史 | 切分后的 train | 含 testA 历史 |
| 预测对象 | 每人的末次点击 | testA 每人隐藏的下一次点击 |
| 能否本地打分 | 能 | 不能 |
| 产物 | Recall/MRR 数字 | `results/itemcf_*.csv` |

**该看到**：线上跑出来的召回字典里，用户数明显多于线下（多了 testA 用户），且每个用户仍然是 10 条。把 `main.py` 的 `use_cache` 设为 `False` 跑一次，再设为 `True` 跑一次——第二次会打印"使用缓存"，耗时接近 0，这就是缓存的边界。

---

## 3. 必须成立的 8 条不变量（自查表）

| # | 不变量 | 在哪验证 |
|---|---|---|
| 1 | 每个用户的历史序列按时间升序 | S3 |
| 2 | `len(user_item_time_dict) == 唯一用户数` | S3 |
| 3 | `len(i2i_sim) == 被点击过的文章数` | S4 |
| 4 | i2i 分数对称、`s > 0`（重复点击时可能 > 1） | S4 |
| 5 | 召回列表不含用户历史里的文章 | S6 |
| 6 | 每个用户恰好 10 条召回，分数降序 | S6/S7 |
| 7 | 提交文件每用户恰好 5 列文章、无 NaN | S9 |
| 8 | 线下 `len(train) + len(val) == len(all)` 且 val 是每人最后一条 | S10 |

---

## 4. 十二个坑（按类别）

### 环境类

1. **系统 `python` 没有 pandas**：用 conda 环境的解释器，或激活环境。
2. **全环境都没有 faiss**：`data_processing.py` 顶部 `import faiss` 会让所有导入失败；装 `faiss-cpu` 或改成延迟导入。
3. **`results/` 不存在**：`submit()` 不建目录，`to_csv` 直接报错。

### 代码类

4. **`get_user_item_time` 未定义**：`main.py:11` 和 `recall_baseline.py:7` 都写错了。
5. **热门补全 `item in item_rank.items()` 永远为假**（`recall_baseline.py:91`）：会把已经正常召回的热门文章负分覆盖。
6. **`recall_baseline.itemcf_sim` 的 `item_created_time_dict` 是死参数**：传了不用。
7. **两套 ItemCF**：`recall_baseline.py`（无加权，教学版）和 `itemcf_recall.py`（位置/时间加权，实际提交用的那条链路）。
8. **`itemcf_recall.py:55` 的 `click_time_weight` 实际恒等于 1**：`i_click_time` 是**毫秒**时间戳，`0.7 ** 1e9` 下溢为 0，`exp(0)=1`。只有两次点击恰好同一毫秒时才是 `exp(1)`。这个"时间相近权重"等于没起作用。
9. **`get_item_info_dict` 原地归一化**：调用后你手里的 `created_at_ts` 已经不是原始值。
10. **`combine_recall_results` 原地改输入**。

> **额外两个（本次会话发现并已修复）**
>
> - `recall_baseline.py:82` 用 `if j in user_hist_items` 判断"用户是否看过"，但 `user_hist_items` 是 `(item, time)` 元组列表，判断**恒为假** → 看过的文章会占掉推荐位。正确做法是先建 `{item for item, _ in user_hist_items}`。
> - `data_processing.py:45` 的 `-> DataFrame` 注解引用了未导入的名字 → 导入时 `NameError`。用 `pd.DataFrame` 即可。

### 概念类

11. **缓存会掩盖逻辑**：`use_cache=True` 直接 `pickle.load`，函数体一行不跑。学习时永远用 `use_cache=False`，或换一个 `save_path`。
12. **线下泄露**：`offline_evaluation_multi.py:377` 把 `train + val` 一起喂给 YouTubeDNN；`main.py` 的线上结果也不能拿来当线下指标。

---

## 5. 十个问题（能答上就算真懂）

1. `all_click_df` 里为什么线上要包含 testA，线下却不包含？
2. `user_item_time_dict` 的值是 dict 还是 list？docstring 写错了吗？
3. i2i 原始共现分为什么除以 `sqrt(cnt_i * cnt_j)`？
4. `1 / log(len(seq) + 1)` 在惩罚什么？
5. 热门补全为什么用负分而不是 0 分？
6. 为什么召回列表要先把用户看过的文章剔除？在哪一步做的？
7. `submit` 里 `unstack` 前后，DataFrame 的 index 和 columns 各是什么？
8. 线下指标 0.24、线上 0.12，差距可能来自哪三件事？
9. 多路融合为什么必须先做 per-user min-max 归一化？
10. 如果把 `sim_item_topk` 从 10 改成 3，覆盖率、MRR 各会怎么变？为什么？

---

## 6. 学完主线后的进阶顺序

1. **对比两套 ItemCF**：`models/itemcf_recall.py:10` 逐行对照 `recall_baseline.py:10`，只多了三个权重因子（位置、点击时间、发布时间），理解它们为什么乘在一起。
2. **UserCF**：`models/usercf_recall.py` 先读 `get_user_activate_degree_dict`（活跃度）、`usercf_sim`（用户相似度），再看 `user_based_recommend`。核心变化是"相似的人看过什么"。
3. **Embedding 召回**：`data_processing.embdding_sim`（faiss 内积检索）→ `models/itemcf_recall.generate_itemcf_embedding_recall_dict`（把向量相似度当第 4 路）。
4. **双塔**：`models/YouTubeDNN_torch.py` 的 `gen_data_set`（滑窗构造样本）→ `train_youtube_dnn` → `youtubednn_u2i_dict`（用户向量×物品向量取 topk）。注意它内部**自己**做了序列切分。
5. **融合与提交**：`online.py` 的 `online_predict`，对照 `offline_evaluation_multi.py`，找出"仅多一步把 testA 拼进来"的差异。

---

## 附：这篇导读验证过什么

- `submit()` 透视后的列名：在本机 `pandas 2.0.3` 上实测为 float `1.0…5.0`，经 `rename` 后正确输出 `article_1…article_5`（`hash(1) == hash(1.0)`）。
- 本机 conda 环境的 pandas/torch/faiss 可用性：见 0.2 表格（实测）。
- `data_raw/`、`cache/`、`results/` 在当前 checkout 中不存在（实测）。
- 其余"该看到"均为代码推导的结构性断言（不变量），不依赖具体数据量，运行即可验证。
