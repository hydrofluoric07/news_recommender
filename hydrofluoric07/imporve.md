Ran inspect notebook cells overview
Ran print cell 0 output lines
Analyzed YouTubeDNN_torch_fast.py:L1-370
Analyzed itemcf_recall.py:L1-237
Analyzed offline_evaluation_multi.py:L100-345
Analyzed data_processing.py:L1-150
Ran list models directory
Analyzed YouTubeDNN_torch.py:L1-100
Analyzed YouTubeDNN_torch.py:L680-760

从输出表格（第 133-139 行）里其实藏着一个非常关键的信号：

* **`YouTubeDNN` 的 `Recall@50`（0.48186）并不差**（已经高于 `UserCF-Emb` 的 0.45806，接近 `ItemCF` 的 0.53633）；
* 但它的 **`Recall@10`（0.21631）和 `MRR@5`（0.07534）断崖式落后**，从 Top-10 到 Top-50 之间出现了高达 `+0.265` 的涨幅。

这说明**双塔并不是没学到用户的兴趣方向，而是“头部精排能力（Top-1 ~ Top-5）”被代码实现上的几个硬伤严重破坏了**。结合源码可以定位出以下 5 个核心原因：

---

### 一、代码实现层面的 4 个直接硬伤（最主要原因）

#### 1. 推理召回时未过滤“用户已点击的历史文章”，挤占了 Top-5 黄金坑位
* **ItemCF 做了严格去重**：在 [itemcf_recall.py](file:///c:/develop/learning/news_recommender/models/itemcf_recall.py#L106-L107) 中明确写了 `if j in hist_item_ids: continue`，Top-50 坑位全部留给**未读过的新文章**。
* **YouTubeDNN 没有去重**：在 [YouTubeDNN_torch_fast.py](file:///c:/develop/learning/news_recommender/models/YouTubeDNN_torch_fast.py#L564-L575) 中，Faiss 直接从全库取 Top-50 返回，**没有剔除用户已经点击过的历史文章**。
* 由于用户向量本来就是用历史点击序列聚合出来的，在向量空间中离用户向量最近的往往正是**用户刚看过的那几篇文章**。这些已读文章占掉了 Top-1 ~ Top-5，导致真正要预测的下一篇新文章被挤到第 10~50 名，直接把 `MRR@5` 和 `Recall@10` 打废，而 `Recall@50` 依然能兜回来。

#### 2. `user_dnn` 最后一层错误加了 `ReLU()` 和 `Dropout(0.3)`，导致向量坍缩
在 [YouTubeDNN_torch_fast.py](file:///c:/develop/learning/news_recommender/models/YouTubeDNN_torch_fast.py#L30-L35) 和 [L440-L441](file:///c:/develop/learning/news_recommender/models/YouTubeDNN_torch_fast.py#L440-L441)：
```python
for unit in hidden_units:  # hidden_units=(128, 64, 64)
    layers.append(nn.Linear(input_dim, unit))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout))  # dropout=0.3
```
这意味着双塔用户侧的**最终输出向量 `user_dnn_out`** 也被套了一层 `ReLU` 和 `Dropout`：
* **`ReLU()` 导致向量空间坍缩**：所有用户向量的 64 个维度被强行截断为 $\ge 0$（全部挤在正象限内，约一半维度直接死掉归零）。从日志第 93-95 行就能看出端倪：随机用户之间的余弦相似度高达 **`0.9988`、`0.9987`、`0.9986`**，向量严重扎堆（各向异性/坍缩），无法与有正有负的 `item_embedding` 精细对齐。
* **`Dropout(0.3)` 干扰点积训练**：训练时在点积前直接随机抹掉 30% 的输出维度，严重破坏内积打分的稳定性。
* **标准做法**：最后一层映射层只保留 `nn.Linear`，绝不能加激活函数和 Dropout。

#### 3. 训练未归一化 + 检索仅归一化了 User，热门文章靠“模长”霸榜
* **训练阶段**（[YouTubeDNN_torch_fast.py](file:///c:/develop/learning/news_recommender/models/YouTubeDNN_torch_fast.py#L75)）：用**未归一化点积** + `BCEWithLogitsLoss` 训练。这种模式下，全局热门文章的 `item_embedding` 会学出极大的**向量模长（L2 范数）**来充当偏置项。
* **检索阶段**（[L491](file:///c:/develop/learning/news_recommender/models/YouTubeDNN_torch_fast.py#L491) vs [L528-L530](file:///c:/develop/learning/news_recommender/models/YouTubeDNN_torch_fast.py#L528-L530)）：代码中只对 `user_embs` 做了 L2 归一化，却**漏掉了对 `item_embs` 的归一化**（原版 `YouTubeDNN_torch.py` L683 是有归一化的）。
* 用 `IndexFlatIP` 检索时，$\mathbf{u} \cdot \mathbf{v}_i = \|\mathbf{v}_i\| \cos\theta$，少数超高频文章仅凭巨大的模长就能排进几乎所有用户的 Top-10，压制了与用户真实兴趣方向贴合的细粒度文章。

#### 4. `negsample=4` 的二分类 BCE 损失缺乏全库排序区分度
* 在 [YouTubeDNN_torch_fast.py](file:///c:/develop/learning/news_recommender/models/YouTubeDNN_torch_fast.py#L309) 和 [L453](file:///c:/develop/learning/news_recommender/models/YouTubeDNN_torch_fast.py#L453)，每个正样本仅配了 4 个全库随机负样本并用 `BCEWithLogitsLoss` 独立算分。
* 从 3.1 万篇文章里随机抽 4 篇大概率是完全不搭边的文章，模型只学到“正样本分高、随机文章分低”（所以 Train Loss 降到 0.09），**但没有学习“最相似的几十篇文章里哪一篇更应该排在 Top-1”**。相比之下，YouTubeDNN 原论文采用的是 **Sampled Softmax（多分类交叉熵 / In-batch Negatives）**，专门迫使正样本在大量竞品中脱颖而出。

---

### 二、天池新闻场景与模型结构的天然差距

除了上述代码问题，这套 `ItemCF` 和当前双塔在“信息利用程度”上也不在一个量级：

1. **新闻场景极度依赖“时间信号”和“近期意图”，而双塔全部丢弃了**
   * 看 [itemcf_recall.py](file:///c:/develop/learning/news_recommender/models/itemcf_recall.py#L51-L122)：`ItemCF` 显式融合了 **4 个强业务先验**：文章发布时间差（排除过期老新闻）、点击时间差（捕捉同一 Session 连续阅读）、点击序列位置衰减（最近点过的权重最高）、以及官方预训练内容向量 `emb_i2i_sim`。
   * 而当前的 `YouTubeDNN` **完全没有接入时间戳特征、文章发布时间、文章分类/字数、以及官方内容 Embedding**，只靠纯 ID 训练；并且对历史序列用**等权平均池化**（[L58-L60](file:///c:/develop/learning/news_recommender/models/YouTubeDNN_torch_fast.py#L58-L60)），把“刚看的上一篇”和“30 篇之前看的”一视同仁，丢掉了最重要的短期点击趋势。
2. **样本极度稀疏下的参数过拟合**
   * 本次数据共 20 万用户、91 万条训练点击，**平均每个用户仅 4.5 条训练点击**。
   * 模型在 [L21-L25](file:///c:/develop/learning/news_recommender/models/YouTubeDNN_torch_fast.py#L21-L25) 却同时定义了 20 万维度的 `user_embedding` 和两套独立的 3.1 万维度物品矩阵（`item_embedding` 和 `hist_embedding` 未共享权重），仅 Embedding 层参数就高达 **1680 万**，极易过拟合（日志里 Epoch 9 之后 `Val Loss` 从 0.3519 反弹到 0.3692 就是过拟合信号）。而经典 YouTubeDNN 论文为了泛化，通常**不设 `user_id` 嵌入**，完全由历史序列和上下文特征表征用户。

*(补充观察：在 [offline_evaluation_multi.py](file:///c:/develop/learning/news_recommender/offline_evaluation_multi.py#L377-L389) 中，传入 `youtubednn_u2i_dict` 的是拼接了 `val_df` 的 `all_df`，导致生成用户向量时其实已经包含了验证集的那次点击——这也是为什么基于这份用户向量的 `UserCF-Emb` 跑出了 0.18958 的高 MRR@5。即便如此，`YouTubeDNN` 自己本身的 `MRR@5` 依然只有 0.07534，恰恰印证了上面“未过滤已读 + 最后一层 ReLU + 未归一化 + 弱负采样”才是真正的瓶颈。)*

---

### 三、立即可见的改进方向（按收益排序）

1. **召回阶段过滤已读历史**：在 Faiss 检索时先取 `topk + max_hist_len`（比如多取 50 个），过滤掉用户在 `train_df` 中点击过的文章后再截断到 `topk`（同时注意把推理输入改为只用 `train_df` 避免信息泄露）。
2. **去掉 `user_dnn` 最后一层的 `ReLU` 和 `Dropout`**，让最后一层纯线性输出，恢复完整 64 维全象限表征空间。
3. **改用余弦相似度 + 温度系数 + In-batch Negatives / Sampled Softmax 训练**：训练和推理时**同时对用户向量和物品向量做 L2 归一化**，用带温度系数的交叉熵（InfoNCE）代替 `negsample=4` 的 BCE，能立竿见影拉升 `MRR@5` 和 `Recall@10`。
4. **融合新闻内容特征与时间衰减**：历史序列池化加入基于时间/位置的指数衰减权重（越近权重越大）；并将官方 `articles_emb.csv` 向量作为物品塔的初始权重或辅助特征，能大幅缓解长尾新闻冷启动。

即使存在上述问题，最终表格中 **`合并结果` 的 Recall@50（0.62293）和 MRR@5（0.22766）依然明显高于单路 `ItemCF`（0.53633 / 0.15527）**，说明双塔学到的全局语义与 `ItemCF` 的共现规则已经形成了很强的互补；修掉上述问题后，合并效果还会有较大提升空间。