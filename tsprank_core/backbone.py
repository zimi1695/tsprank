# tsprank_core/backbone.py
"""
DoubleEnsemble Backbone（对齐 W4 超参）

修复内容（vs v2.0）：
  [Fix 2] import os 从文件底部移至顶部。
           原因：os 在 save() 中被 os.makedirs() 使用，
           虽然 Python 模块级导入在类实例化前全部执行，
           底部放置会让静态分析工具（pylint/mypy）和 IDE 误报 NameError，
           且违反 PEP 8，增加维护风险。

  [Fix 2] save() 中 os.path.dirname(path) 在 path 无目录时返回 ''，
           os.makedirs('') 会抛出 FileNotFoundError，
           原代码已用 or "." 规避，保留该修复。

v2.0 保留内容（诊断说明）：
  1. 不使用 early_stopping（CSZScoreNorm 后 label 方差≈1，
     valid L2≈1.0，early_stopping 会在第 1 轮触发）
  2. n_trees_per_model 直接赋值 n_estimators，不用 best_iteration
  3. predict_leaf 不再用于 TSPRank 输入
  4. num_iteration=-1 确保使用所有树

参考：
  r2_c06_densemble_alpha158_w4_sim2019_s20260508 / config.yaml
  DoubleEnsemble: arxiv 2010.01265
"""

from __future__ import annotations

import os            # [Fix 2] 移至顶部（原在文件底部）
import pickle
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import lightgbm as lgb
import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ============================================================
# W4 超参（直接来自 config.yaml，两处变动已注释说明）
# ============================================================
W4_PARAMS: dict = {
    "objective":            "mse",
    "boosting":             "gbdt",
    "num_leaves":           210,
    "max_depth":            8,
    "colsample_bytree":     0.8879,
    "learning_rate":        0.05,      # config=0.2 → 降低，因label已CSZScoreNorm
    "subsample":            0.8789,
    "lambda_l1":            205.6999,
    "lambda_l2":            580.9768,
    "seed":                 20260508,
    "feature_fraction_seed":20260508,
    "bagging_seed":         20260508,
    "data_random_seed":     20260508,
    "deterministic":        True,
    "force_col_wise":       True,
    "verbose":             -1,
    "n_jobs":              -1,
}

N_ESTIMATORS  = 200    # config=28 → 增大，配合降低的 lr
NUM_MODELS    = 3
SAMPLE_RATIOS = [0.8, 0.7, 0.6, 0.5, 0.4]   # 按 num_models 取前 3 项
ALPHA1        = 1.0    # SR 权重
ALPHA2        = 1.0    # FS 权重
BINS_SR       = 10
BINS_FS       = 5


# ============================================================
# DEBackbone
# ============================================================

class DEBackbone:
    """
    DoubleEnsemble：Sample Reweighting + Feature Selection × num_models 个 LGB 子模型。
    """

    def __init__(self) -> None:
        self.models:            List[lgb.Booster] = []
        self.feat_weights_list: List[np.ndarray]  = []
        self.n_trees_per_model: int = N_ESTIMATORS
        self.n_features:        int = 0

    # ── 工具：计算样本权重（Sample Reweighting）─────────────────
    @staticmethod
    def _sample_weights(
        pred: np.ndarray,
        y:    np.ndarray,
        bins: int = BINS_SR,
        alpha: float = ALPHA1,
    ) -> np.ndarray:
        """
        按预测排名 vs 真实排名的差距计算样本权重。
        预测越准（差距小）权重越低；差距大的样本加权，迫使模型关注难例。
        """
        N = len(y)
        pred_rank  = np.argsort(np.argsort(pred)).astype(float)
        true_rank  = np.argsort(np.argsort(y)).astype(float)
        diff       = np.abs(pred_rank - true_rank) / N

        bin_idx    = np.minimum((diff * bins).astype(int), bins - 1)
        bin_counts = np.bincount(bin_idx, minlength=bins).astype(float)
        bin_counts = np.where(bin_counts == 0, 1.0, bin_counts)
        weights    = 1.0 / bin_counts[bin_idx]
        weights    = weights ** alpha
        weights   /= weights.sum() / N
        return weights.astype(np.float32)

    # ── 工具：计算特征权重（Feature Selection）────────────────
    @staticmethod
    def _feature_weights(
        X:     np.ndarray,
        resid: np.ndarray,
        bins:  int   = BINS_FS,
        alpha: float = ALPHA2,
    ) -> np.ndarray:
        """
        按特征与当前模型残差的 Spearman 相关系数分配权重。
        相关越高说明该特征还有信息量，赋予更高权重。
        """
        F = X.shape[1]
        ic = np.zeros(F, dtype=np.float32)
        for j in range(F):
            c, _ = spearmanr(X[:, j], resid)
            ic[j] = abs(c) if not np.isnan(c) else 0.0

        q = np.quantile(ic, np.linspace(0, 1, bins + 1))
        q[-1] += 1e-8
        bin_idx = np.searchsorted(q[1:], ic, side="left")
        bin_idx = np.clip(bin_idx, 0, bins - 1)

        weights = (bin_idx + 1).astype(float) ** alpha
        weights /= weights.sum() / F
        return weights.astype(np.float32)

    # ── 主训练流程 ────────────────────────────────────────────
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
    ) -> "DEBackbone":
        """
        训练 NUM_MODELS 个子模型，每个子模型迭代：
          1. 用上一轮残差计算样本权重
          2. 用上一轮残差计算特征权重
          3. 缩放特征：X_scaled = X * feat_weights
          4. 加权采样 → lgb.Dataset
          5. 训练 N_ESTIMATORS 轮（无 early stopping）
        """
        self.n_features = X_train.shape[1]
        self.models.clear()
        self.feat_weights_list.clear()

        resid_train = y_train.copy().astype(np.float32)
        resid_valid = y_valid.copy().astype(np.float32)

        ratio_list = SAMPLE_RATIOS[:NUM_MODELS]

        for m_idx in range(NUM_MODELS):
            ratio = ratio_list[m_idx]
            print(f"[backbone] 训练子模型 {m_idx+1}/{NUM_MODELS}  ratio={ratio}")

            # 1. 样本权重
            if m_idx == 0:
                s_weights = np.ones(len(y_train), dtype=np.float32)
            else:
                prev_pred = self.models[-1].predict(
                    X_train * self.feat_weights_list[-1],
                    num_iteration=-1,
                )
                s_weights = self._sample_weights(prev_pred, y_train)

            # 2. 特征权重
            if m_idx == 0:
                f_weights = np.ones(self.n_features, dtype=np.float32)
            else:
                f_weights = self._feature_weights(X_train, resid_train)
            self.feat_weights_list.append(f_weights)

            # 3. 特征缩放
            Xs_train = X_train * f_weights
            Xs_valid = X_valid * f_weights

            # 4. 加权采样
            N = len(y_train)
            n_sample = max(int(N * ratio), 100)
            prob     = s_weights / s_weights.sum()
            idx      = np.random.choice(N, size=n_sample, replace=False, p=prob)

            dtrain = lgb.Dataset(
                Xs_train[idx], label=y_train[idx],
                weight=s_weights[idx],
                free_raw_data=True,
            )
            dvalid = lgb.Dataset(
                Xs_valid, label=y_valid,
                reference=dtrain,
                free_raw_data=True,
            )

            # 5. 训练（不使用 early stopping）
            params = {**W4_PARAMS, "seed": W4_PARAMS["seed"] + m_idx}
            model  = lgb.train(
                params,
                dtrain,
                num_boost_round=N_ESTIMATORS,
                valid_sets=[dvalid],
                callbacks=[lgb.log_evaluation(period=50)],
                # ⚠️ 无 early_stopping_callback：
                #    label 被 CSZScoreNorm 后方差≈1，
                #    valid L2≈1.0 被认为是第1轮最优，会立即停止。
            )
            self.models.append(model)
            self.n_trees_per_model = N_ESTIMATORS

            # 更新残差
            resid_train = y_train - model.predict(Xs_train, num_iteration=-1)
            resid_valid = y_valid - model.predict(Xs_valid, num_iteration=-1)

            ic_val, _ = spearmanr(
                model.predict(Xs_valid, num_iteration=-1), y_valid
            )
            print(f"  → sub-model {m_idx+1} valid Rank IC = {ic_val:.4f}")

        print(f"[backbone] 训练完成，共 {NUM_MODELS} 个子模型，"
              f"每模型 {self.n_trees_per_model} 棵树")
        return self

    # ── 推理：预测得分（ensemble 平均）────────────────────────
    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """
        输出 ensemble 平均预测分数，shape=(N,)。
        """
        assert self.models, "模型未训练，请先调用 fit()"
        preds = []
        for model, fw in zip(self.models, self.feat_weights_list):
            preds.append(model.predict(X * fw, num_iteration=-1))
        return np.mean(preds, axis=0).astype(np.float32)

    # ── 诊断：Rank IC ─────────────────────────────────────────
    def eval_rank_ic(
        self,
        X_dict: dict,
        y_dict: dict,
        dates:  Optional[list] = None,
        max_days: int = 100,
    ) -> Tuple[float, float, float]:
        """返回 (mean_IC, std_IC, IR)。"""
        if dates is None:
            dates = sorted(X_dict.keys())[:max_days]
        ics = []
        for d in dates:
            if d not in X_dict or d not in y_dict:
                continue
            pred = self.predict_score(X_dict[d])
            ic, _ = spearmanr(pred, y_dict[d])
            if not np.isnan(ic):
                ics.append(ic)
        if not ics:
            return 0.0, 0.0, 0.0
        arr = np.array(ics)
        mean_ic  = float(np.mean(arr))
        std_ic   = float(np.std(arr)) + 1e-8
        ir       = mean_ic / std_ic
        pos_rate = float(np.mean(arr > 0))
        print(f"[backbone] Rank IC={mean_ic:.4f}  std={std_ic:.4f}  "
              f"IR={ir:.4f}  正IC率={pos_rate:.1%}  n={len(ics)}")
        return mean_ic, std_ic, ir

    # ── 诊断：检测旧格式 pkl ──────────────────────────────────
    def is_degraded(self) -> bool:
        """检测是否为旧格式 pkl（feat_weights 全为单位权重，IC 退化）。"""
        if not self.feat_weights_list:
            return True
        return all(np.allclose(fw, 1.0) for fw in self.feat_weights_list)

    # ── 序列化 ────────────────────────────────────────────────
    def save(self, path: str) -> None:
        # [Fix 2] os 已在顶部正确导入，此处 os.makedirs 可正常调用
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "models":            self.models,
                    "feat_weights_list": self.feat_weights_list,
                    "n_trees_per_model": self.n_trees_per_model,
                    "n_features":        self.n_features,
                },
                f,
            )
        print(f"[backbone] 已保存至 {path}")

    def load(self, path: str) -> "DEBackbone":
        with open(path, "rb") as f:
            ckpt = pickle.load(f)

        # ── 新格式（本次重写的 save 格式）─────────────────────
        if isinstance(ckpt, dict) and "models" in ckpt:
            self.models            = ckpt["models"]
            self.n_trees_per_model = ckpt.get("n_trees_per_model", N_ESTIMATORS)
            self.n_features        = ckpt.get("n_features", 0)
            if "feat_weights_list" in ckpt:
                self.feat_weights_list = ckpt["feat_weights_list"]
            else:
                print(f"[backbone] ⚠️  旧格式 pkl，feat_weights_list 缺失，"
                      f"用单位权重代替（预测结果基本不变）")
                n_feat = (self.models[0].num_feature()
                          if self.models else self.n_features)
                self.feat_weights_list = [
                    np.ones(n_feat, dtype=np.float32)
                    for _ in self.models
                ]
                self.n_features = n_feat

        # ── 旧格式（直接 pickle 整个对象）─────────────────────
        elif hasattr(ckpt, "models"):
            self.models            = ckpt.models
            self.feat_weights_list = getattr(
                ckpt, "feat_weights_list",
                [np.ones(ckpt.models[0].num_feature(), dtype=np.float32)
                 for _ in ckpt.models]
            )
            self.n_trees_per_model = getattr(ckpt, "n_trees_per_model", N_ESTIMATORS)
            self.n_features        = getattr(ckpt, "n_features", 0)

        else:
            raise ValueError(
                f"未知的 backbone pkl 格式，"
                f"keys={list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
            )

        print(f"[backbone] 已从 {path} 加载，"
              f"{len(self.models)} 个子模型 × {self.n_trees_per_model} 棵树")
        return self
