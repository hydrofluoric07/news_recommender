import os
import pickle
import random
import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


# 双塔模型定义
class YouTubeDNNModel(nn.Module):
    def __init__(self, user_count, item_count, embedding_dim=16, hidden_units=(64, 16), dropout=0.2):
        super(YouTubeDNNModel, self).__init__()
        self.embedding_dim = embedding_dim

        # 用户和物品的嵌入层
        self.user_embedding = nn.Embedding(user_count, embedding_dim)
        self.item_embedding = nn.Embedding(item_count, embedding_dim)

        # 历史物品序列聚合
        self.hist_embedding = nn.Embedding(item_count, embedding_dim)

        # 用户塔深度网络
        layers = []
        input_dim = embedding_dim * 2  # 用户ID嵌入 + 历史序列嵌入
        for unit in hidden_units:
            layers.append(nn.Linear(input_dim, unit))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = unit
        self.user_dnn = nn.Sequential(*layers)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_id, hist_item_seq, target_item, seq_len):
        # 用户ID Embedding
        user_emb = self.user_embedding(user_id)  # [B, E]

        # 历史物品序列Embedding
        hist_emb = self.hist_embedding(hist_item_seq)  # [B, L, E]

        # 计算序列的平均值
        device = user_id.device
        mask = torch.arange(hist_item_seq.size(1), device=device).unsqueeze(0) < seq_len.unsqueeze(1)
        mask = mask.unsqueeze(2).float()
        hist_emb = (hist_emb * mask).sum(dim=1) / seq_len.unsqueeze(1).float().clamp(min=1)  # [B, E]

        # 连接用户嵌入和历史物品嵌入
        user_feature = torch.cat([user_emb, hist_emb], dim=1)  # [B, 2E]

        # 用户塔DNN
        user_dnn_out = self.user_dnn(user_feature)  # [B, last_hidden]

        # 目标物品Embedding
        item_emb = self.item_embedding(target_item)  # [B, E]

        # 计算点积
        if len(item_emb.shape) == 3:  # 批量计算多个物品 [B, N, E]
            score = torch.bmm(user_dnn_out.unsqueeze(1), item_emb.transpose(1, 2)).squeeze(1)  # [B, N]
        else:  # 单个物品 [B, E]
            score = torch.sum(user_dnn_out * item_emb, dim=1)  # [B]

        return score

    def get_user_embedding(self, user_id, hist_item_seq, seq_len):
        user_emb = self.user_embedding(user_id)
        hist_emb = self.hist_embedding(hist_item_seq)

        device = user_id.device
        # 获取张量第 dim 个维度的长度
        # (1,L),(B,1)->(B,L)
        mask = torch.arange(hist_item_seq.size(1), device=device).unsqueeze(0) < seq_len.unsqueeze(1)
        mask = mask.unsqueeze(2).float() # (B,L)->(B,L,1)
        hist_emb = (hist_emb * mask).sum(dim=1) / seq_len.unsqueeze(1).float().clamp(min=1)

        user_feature = torch.cat([user_emb, hist_emb], dim=1)
        user_dnn_out = self.user_dnn(user_feature)

        return user_dnn_out

    def get_item_embedding(self, item_id):
        return self.item_embedding(item_id)


def _sample_negatives(all_items, pos_set, count):
    """快速负采样：对全量物品数组随机索引 + 冲突重采，避免 O(|all_items|) 集合差集。"""
    n_all = len(all_items)
    if count == 0:
        return np.empty(0, dtype=np.int32)

    # 若用户点击覆盖了大部分物品，退回差集采样避免死循环
    if len(pos_set) >= n_all * 0.5:
        candidates = np.array([x for x in all_items if x not in pos_set], dtype=np.int32)
        if len(candidates) == 0:
            return np.zeros(count, dtype=np.int32)
        return np.random.choice(candidates, size=count, replace=(len(candidates) < count))

    res = np.empty(count, dtype=np.int32)
    filled = 0
    while filled < count:
        need = count - filled
        batch = all_items[np.random.randint(0, n_all, size=max(need * 2, need + 8))]
        # 批量抽，超额采样作为安全冗余，因为抽中的可能有正样本
        for item in batch:
            if int(item) not in pos_set:
                res[filled] = item
                filled += 1
                if filled == count:
                    break
    return res


def gen_data_set(data, negsample=0, max_hist_len=30, cache_path=None, use_cache=True):
    """
    向量化生成训练集和测试集张量字典（避免百万级 Python 集合差集与循环开销），支持缓存。
    """
    if use_cache and cache_path and os.path.exists(cache_path):
        print(f"✅ 从缓存加载训练集和测试集: {cache_path}")
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        if isinstance(cached, dict) and 'train' in cached and 'test' in cached:
            return cached['train'], cached['test']

    print("🚀 开始生成训练集和测试集（加速版）...")
    sorted_df = data.sort_values(['user_id_encoded', 'click_timestamp'], kind='mergesort')
    # 先按用户编号分组排列，同一用户内再按点击时间从早到晚排，并且时间相同的行保持原来的先后顺序，归并排序稳定确保结果可复现
    u_arr = sorted_df['user_id_encoded'].to_numpy(dtype=np.int32)
    i_arr = sorted_df['click_article_id_encoded'].to_numpy(dtype=np.int32)

    all_items = np.unique(i_arr)
    max_item_id = int(all_items.max())
    print(f"编码后物品ID范围: 0-{max_item_id}")

    # 用数组切片切出每个用户的历史序列，避免 20 万次 pandas groupby
    split_idx = np.flatnonzero(u_arr[1:] != u_arr[:-1]) + 1
    starts = np.concatenate(([0], split_idx))
    ends = np.concatenate((split_idx, [len(u_arr)]))

    lengths = ends - starts
    # 精确计算 train / test 总行数，提前预分配连续数组
    # n_train = int(np.sum(np.where(lengths == 1, 1, np.maximum(lengths - 2, 0) * (1 + negsample))))
    # np.maximum()对两个数组（或数组与标量）逐元素比较，每个位置取较大的那个值，返回新数组
    n_train = int(np.sum(np.where(lengths == 1, 1, (lengths - 2) * (1 + negsample)))) # 训练样本数量
    n_test = len(lengths) # 测试样本数量

    tr_user = np.empty(n_train, dtype=np.int32) # 用户id
    tr_seq = np.zeros((n_train, max_hist_len), dtype=np.int32) # 用户历史行为矩阵，未被点击填充0
    tr_target = np.empty(n_train, dtype=np.int32) # 要预测的目标物品ID
    tr_label = np.empty(n_train, dtype=np.float32) #标签，正样本1负样本0
    tr_len = np.empty(n_train, dtype=np.int32) # 真实历史序列长度

    te_user = np.empty(n_test, dtype=np.int32)
    te_seq = np.zeros((n_test, max_hist_len), dtype=np.int32)
    te_target = np.empty(n_test, dtype=np.int32)
    te_label = np.ones(n_test, dtype=np.float32)
    te_len = np.empty(n_test, dtype=np.int32)

    tr_ptr = 0
    te_ptr = 0

    for s, e in tqdm(zip(starts, ends), total=len(starts), desc="构建训练样本"):
        uid = int(u_arr[s])
        pos_list = i_arr[s:e]
        L = len(pos_list)

        if L == 1:
            """
            模型用这个物品去预测自己,强信息泄露,测试集要求对全部用户进行评估/生成召回结果,丢弃了 L=1 的用户，测试集就会缺失这些用户的评估甚至直接报错
            模型非常容易学到投机规律：“只要历史里有物品 A，输出直接匹配 A 就能拿满分”。新闻推荐场景下，通常有“已读去重“，不一定会直接复读，但在训练时会有特征偏差
            更严谨处理方式：历史留空，用纯 User ID 预测目标物品、
            """
            tr_user[tr_ptr] = uid
            tr_seq[tr_ptr, 0] = pos_list[0]
            tr_target[tr_ptr] = pos_list[0]
            tr_label[tr_ptr] = 1.0
            tr_len[tr_ptr] = 1
            tr_ptr += 1

            te_user[te_ptr] = uid
            te_seq[te_ptr, 0] = pos_list[0]
            te_target[te_ptr] = pos_list[0]
            te_len[te_ptr] = 1
            te_ptr += 1
            continue
        
        # 一次性为该用户采样所有负样本
        neg_need = (L - 2) * negsample
        if neg_need > 0:
            neg_items = _sample_negatives(all_items, set(pos_list.tolist()), neg_need)
        neg_ptr = 0

        for idx in range(1, L): #idx=0 时前面没有任何交互，无法构成序列样本，因此从1开始
            h_start = max(0, idx - max_hist_len) #限制最大历史长度
            hist_rev = pos_list[h_start:idx][::-1] #翻转，[最近, ..., 最早]排序
            hlen = len(hist_rev)

            if idx != L - 1: # 不是最后一次点击，pos_list[idx]就可以作为正样本
                # 正样本
                tr_user[tr_ptr] = uid
                tr_seq[tr_ptr, :hlen] = hist_rev
                tr_target[tr_ptr] = pos_list[idx]
                tr_label[tr_ptr] = 1.0
                tr_len[tr_ptr] = hlen
                tr_ptr += 1

                # 负样本（共享同一段历史序列）
                if negsample > 0:
                    end_ptr = tr_ptr + negsample
                    tr_user[tr_ptr:end_ptr] = uid
                    tr_seq[tr_ptr:end_ptr, :hlen] = hist_rev
                    tr_target[tr_ptr:end_ptr] = neg_items[neg_ptr:neg_ptr + negsample]
                    tr_label[tr_ptr:end_ptr] = 0.0
                    tr_len[tr_ptr:end_ptr] = hlen
                    tr_ptr = end_ptr
                    neg_ptr += negsample
            else:
                te_user[te_ptr] = uid
                te_seq[te_ptr, :hlen] = hist_rev
                te_target[te_ptr] = pos_list[idx]
                te_len[te_ptr] = hlen
                te_ptr += 1

    # 在最后统一打乱训练集与测试集，不打乱会导致每个批次的梯度剧烈震荡、模型更新严重偏向个别用户，极难收敛。
    if n_train > 1:
        perm_tr = np.random.permutation(n_train) # 生成一个长度为 n_train 的随机索引排列
        # 5 个平行数组需要用同一个排列索引同步打乱，所以不用np.random.shuffle
        tr_user = tr_user[perm_tr]
        tr_seq = tr_seq[perm_tr]
        tr_target = tr_target[perm_tr]
        tr_label = tr_label[perm_tr]
        tr_len = tr_len[perm_tr]

    if n_test > 1:
        perm_te = np.random.permutation(n_test)
        te_user = te_user[perm_te]
        te_seq = te_seq[perm_te]
        te_target = te_target[perm_te]
        te_label = te_label[perm_te]
        te_len = te_len[perm_te]

    train_data = {
        'user_id': torch.from_numpy(tr_user), # 将一个 NumPy ndarray 零拷贝地转换为 PyTorch Tensor
        'hist_item_seq': torch.from_numpy(tr_seq),
        'target_item': torch.from_numpy(tr_target),
        'label': torch.from_numpy(tr_label),
        'seq_len': torch.from_numpy(tr_len),
    }
    test_data = {
        'user_id': torch.from_numpy(te_user),
        'hist_item_seq': torch.from_numpy(te_seq),
        'target_item': torch.from_numpy(te_target),
        'label': torch.from_numpy(te_label),
        'seq_len': torch.from_numpy(te_len),
    }

    print("✅ 数据集验证通过")
    print(f"训练集大小: {n_train}")
    print(f"测试集大小: {n_test}")

    if cache_path:
        print(f"💾 保存训练集和测试集到缓存: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({'train': train_data, 'test': test_data}, f)

    return train_data, test_data


def _prepare_device_tensors(tensor_dict, device):
    """尽量把训练张量整块常驻 GPU（几百 MB 级别），显存不足时回退到 CPU 按批搬运。"""
    if device.type == 'cuda':
        try:
            return {k: v.to(device, non_blocking=True) for k, v in tensor_dict.items()}, True
        except RuntimeError:
            torch.cuda.empty_cache()
    return tensor_dict, False


def train_youtube_dnn(train_data, test_data, model, device, epochs=5, batch_size=2048,
                      learning_rate=0.002, weight_decay=1e-6):
    """
    训练 YouTubeDNN 模型：直接基于连续张量按大 batch 切片，移除 CUDA_LAUNCH_BLOCKING 同步阻塞。
    """
    print(f"[train_youtube_dnn] 开始训练，设备: {device}, batch_size: {batch_size}")

    model = model.to(device)
    use_amp = (device.type == 'cuda') # Automatic Mixed Precision自动混合精度
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        autocast_ctx = lambda: torch.amp.autocast('cuda', enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=use_amp)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_tensors, tr_on_gpu = _prepare_device_tensors(train_data, device)
    test_tensors, te_on_gpu = _prepare_device_tensors(test_data, device)
    print(f"[train_youtube_dnn] 张量常驻GPU: train={tr_on_gpu}, test={te_on_gpu}")

    n_train = len(train_tensors['user_id'])
    n_test = len(test_tensors['user_id'])
    tr_dev = device if tr_on_gpu else torch.device('cpu')
    te_dev = device if te_on_gpu else torch.device('cpu')

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0

        perm = torch.randperm(n_train, device=tr_dev) # 保证每个 Epoch 的 Batch 顺序都是全新的，防止模型过拟合固定的数据顺序
        num_steps = (n_train + batch_size - 1) // batch_size # 整除向上取整公式，计算当前 Epoch 总共需要跑多少个 Batch

        for step in tqdm(range(num_steps), desc=f'Epoch {epoch+1}/{epochs}'):
            idx = perm[step * batch_size:(step + 1) * batch_size] # 获取当前批次需要的样本下标
            user_id = train_tensors['user_id'][idx].long()
            hist_item_seq = train_tensors['hist_item_seq'][idx].long()
            target_item = train_tensors['target_item'][idx].long()
            seq_len = train_tensors['seq_len'][idx].long()
            label = train_tensors['label'][idx].float()

            if not tr_on_gpu: # 退回到传统模式：当前 Batch 需要哪 2048 条数据，就临时把这 2048 条数据通过 .to(device) 搬到 GPU 显存上
                user_id = user_id.to(device, non_blocking=True) # 异步数据传输。它告诉 PyTorch 不需要卡死 CPU 等待数据完全拷到显卡，而是让 CUDA 数据传输流和计算流重叠，提升吞吐效率
                hist_item_seq = hist_item_seq.to(device, non_blocking=True)
                target_item = target_item.to(device, non_blocking=True)
                seq_len = seq_len.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(): # 开启自动混合精度环境
                scores = model(user_id, hist_item_seq, target_item, seq_len)
                loss = criterion(scores, label)

            scaler.scale(loss).backward() # 自动将 Loss 放大（乘以缩放因子），防止在 FP16 下较小的梯度发生数值下溢（变成 0），然后反向传播求导。
            scaler.step(optimizer) # 把梯度除以缩放因子还原回真实数值
            scaler.update() # 动态调整下一个 Step 的缩放倍数（连续没有溢出就尝试放大倍数，一旦溢出就缩小倍数）

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / max(1, train_batches)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_steps = (n_test + batch_size - 1) // batch_size

        with torch.no_grad():
            for step in range(val_steps):
                start = step * batch_size
                end = min(start + batch_size, n_test)
                user_id = test_tensors['user_id'][start:end].long()
                hist_item_seq = test_tensors['hist_item_seq'][start:end].long()
                target_item = test_tensors['target_item'][start:end].long()
                seq_len = test_tensors['seq_len'][start:end].long()
                label = test_tensors['label'][start:end].float()

                if not te_on_gpu:
                    user_id = user_id.to(device, non_blocking=True)
                    hist_item_seq = hist_item_seq.to(device, non_blocking=True)
                    target_item = target_item.to(device, non_blocking=True)
                    seq_len = seq_len.to(device, non_blocking=True)
                    label = label.to(device, non_blocking=True)

                with autocast_ctx():
                    scores = model(user_id, hist_item_seq, target_item, seq_len)
                    loss = criterion(scores, label)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            # detach()将张量从 Autograd 计算图中剥离，阻断梯度传播。
            # cpu()把参数从 GPU 显存拷贝回电脑物理内存条（RAM）中暂存
            # clone()在物理内存中开辟一块全新的独立存储，进行物理深拷贝

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"已恢复最佳模型，验证损失: {best_val_loss:.4f}")

    print("[train_youtube_dnn] 训练完成!")
    return model


def youtubednn_u2i_dict(data, save_path="./cache/", topk=20, epochs=5, batch_size=2048,
                        embedding_dim=32, use_cache=True):
    model_cache = os.path.join(save_path, 'youtube_model.pth') # 神经网络模型权重
    # 20 万用户经过 User 塔映射并 L2 归一化后的 64 维向量，3 万多篇文章经过 Item 塔后的 64 维向量
    embeddings_cache = os.path.join(save_path, 'youtube_embeddings.pkl')
    cache_path = os.path.join(save_path, 'youtube_u2i_dict.pkl') # 最终召回结果字典
    dataset_cache = os.path.join(save_path, 'youtube_dataset_fast.pkl') # 原始数据处理后得到的数据集

    print("[youtubednn_u2i_dict] 🚀 开始YouTubeDNN处理（GPU加速版）...")
    os.makedirs(save_path, exist_ok=True)

    if use_cache and os.path.exists(cache_path):
        print(f"[youtubednn_u2i_dict] ✅ 使用召回缓存：{cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # 1. ID 编码
    print("[youtubednn_u2i_dict] 开始ID编码...")
    data = data.copy()
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    data['user_id_encoded'] = user_encoder.fit_transform(data['user_id'])
    data['click_article_id_encoded'] = item_encoder.fit_transform(data['click_article_id'])

    user_count = len(user_encoder.classes_) + 1
    item_count = len(item_encoder.classes_) + 1
    print(f"[youtubednn_u2i_dict] 编码后 - 用户数量: {user_count}, 物品数量: {item_count}")

    model = YouTubeDNNModel(
        user_count,
        item_count,
        embedding_dim=embedding_dim,
        hidden_units=(128, 64, embedding_dim),
        dropout=0.3
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 训练或加载模型
    if use_cache and os.path.exists(model_cache):
        print(f"[youtubednn_u2i_dict] ✅ 加载预训练模型：{model_cache}")
        model.load_state_dict(torch.load(model_cache, map_location='cpu'))
    else:
        train_data, test_data = gen_data_set(
            data,
            negsample=4,
            max_hist_len=30,
            cache_path=dataset_cache,
            use_cache=use_cache
        )

        # 大 batch 时适度提高学习率，兼顾收敛速度与稳定性
        lr = 0.003 if batch_size >= 1024 else 0.001
        model = train_youtube_dnn(
            train_data=train_data,
            test_data=test_data,
            model=model,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            weight_decay=1e-6
        )
        torch.save(model.state_dict(), model_cache)

    # 3. 批量提取用户与物品嵌入（替代 20 万次逐行扫描 DataFrame）
    user_embeddings = {}
    item_embeddings = {}

    if use_cache and os.path.exists(embeddings_cache):
        print(f"[youtubednn_u2i_dict] ✅ 发现嵌入缓存: {embeddings_cache}")
        with open(embeddings_cache, 'rb') as f:
            cache_data = pickle.load(f)
            user_embeddings = cache_data['user_embeddings']
            item_embeddings = cache_data['item_embeddings']
    else:
        print(f"[youtubednn_u2i_dict] 批量计算用户和物品嵌入（设备: {device}）...")
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            # 计算所有物品嵌入
            encoded_items = torch.arange(len(item_encoder.classes_), dtype=torch.long, device=device)
            item_embs_np = model.get_item_embedding(encoded_items).detach().cpu().numpy()
            item_embeddings = dict(zip(item_encoder.classes_.tolist(), item_embs_np))

            # 用一次排序 + 数组切分收集每个用户的最近 30 条历史
            sorted_df = data.sort_values(['user_id_encoded', 'click_timestamp'], kind='mergesort')
            u_enc_arr = sorted_df['user_id_encoded'].to_numpy(dtype=np.int32)
            i_enc_arr = sorted_df['click_article_id_encoded'].to_numpy(dtype=np.int32)

            split_idx = np.flatnonzero(u_enc_arr[1:] != u_enc_arr[:-1]) + 1
            starts = np.concatenate(([0], split_idx))
            ends = np.concatenate((split_idx, [len(u_enc_arr)]))

            n_users = len(starts)
            max_hist_len = 30
            u_ids_enc = np.empty(n_users, dtype=np.int64)
            u_hist_mat = np.zeros((n_users, max_hist_len), dtype=np.int64)
            u_seq_lens = np.empty(n_users, dtype=np.int64)

            for idx_u, (s, e) in enumerate(zip(starts, ends)):
                u_ids_enc[idx_u] = u_enc_arr[s]
                hist_items = i_enc_arr[max(s, e - max_hist_len):e]
                hlen = len(hist_items)
                u_hist_mat[idx_u, :hlen] = hist_items
                u_seq_lens[idx_u] = hlen

            # 批量推理用户嵌入
            infer_bs = max(batch_size, 4096)
            all_u_embs = np.empty((n_users, embedding_dim), dtype=np.float32)

            for start in tqdm(range(0, n_users, infer_bs), desc="批量计算用户嵌入"):
                end = min(start + infer_bs, n_users)
                u_t = torch.from_numpy(u_ids_enc[start:end]).to(device, non_blocking=True)
                h_t = torch.from_numpy(u_hist_mat[start:end]).to(device, non_blocking=True)
                l_t = torch.from_numpy(u_seq_lens[start:end]).to(device, non_blocking=True)
                u_out = model.get_user_embedding(u_t, h_t, l_t).detach().cpu().numpy()
                all_u_embs[start:end] = u_out

            norms = np.linalg.norm(all_u_embs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            all_u_embs = all_u_embs / norms

            orig_user_ids = user_encoder.inverse_transform(u_ids_enc)
            user_embeddings = dict(zip(orig_user_ids.tolist(), all_u_embs))

        cache_data = {
            'user_embeddings': user_embeddings,
            'item_embeddings': item_embeddings,
            'user_encoder': user_encoder,
            'item_encoder': item_encoder
        }
        with open(embeddings_cache, 'wb') as f:
            pickle.dump(cache_data, f)

    # 4. Faiss 向量检索（优先尝试 GPU Faiss，若为 CPU 版则回退）
    print("[youtubednn_u2i_dict] 使用Faiss进行向量检索...")
    user_ids = list(user_embeddings.keys())
    user_embs = np.array([user_embeddings[user_id] for user_id in user_ids], dtype=np.float32)

    item_ids = list(item_embeddings.keys())
    item_embs = np.array([item_embeddings[item_id] for item_id in item_ids], dtype=np.float32)

    index = faiss.IndexFlatIP(embedding_dim)
    if torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources'):
        try:
            res = faiss.StandardGpuResources() # 负责为 GPU 上的向量检索预分配和管理所需的显存
            index = faiss.index_cpu_to_gpu(res, 0, index) # 将索引拷贝到显卡
            print("[youtubednn_u2i_dict] ✅ Faiss 使用 GPU 加速检索")
        except Exception:
            pass
    index.add(np.ascontiguousarray(item_embs))

    sim, idx = index.search(np.ascontiguousarray(user_embs), topk)

    user_recall_items_dict = {}
    n_items = len(item_ids)
    for i, user_id in enumerate(user_ids):
        row_idx = idx[i]
        row_sim = sim[i]
        user_recall_items_dict[user_id] = [
            (item_ids[item_idx], float(score))
            for item_idx, score in zip(row_idx, row_sim)
            if 0 <= item_idx < n_items
        ]

    with open(cache_path, 'wb') as f:
        pickle.dump(user_recall_items_dict, f)
    print(f"[youtubednn_u2i_dict] ✅ 召回结果已保存至: {cache_path}")

    return user_recall_items_dict
