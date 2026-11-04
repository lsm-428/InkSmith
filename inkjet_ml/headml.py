# -*- coding: utf-8 -*-
"""三套喷头 ML 共用的建模核心：特征矩阵、5 个模型、交叉验证评估、报告输出。

模型与超参完全沿用论文 Table 3（Kim et al., Sci Rep 2022, doi:10.1038/s41598-022-08784-y）
已验证的一套，避免各喷头各写一套导致无法横向比较：

    SVM     C=10, gamma=0.5(分类) / 0.3(回归)
    kNN     k=20(分类) / 10(回归)
    RFs     Max_depth=20
    MLP     hidden=(100,100)(分类) / (200,100)(回归)
    XGBoost Max_depth=5, Subsample=0.7, Min_child_weight=20(分类)

交叉验证**按测量会话分组**（GroupKFold by session）：同一套波形下测的多个喷嘴
高度相关，随机划分会造成信息泄漏，让指标虚高。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_squared_error,
                             precision_recall_fscore_support, r2_score)
from sklearn.model_selection import (GroupKFold, KFold, LeaveOneOut,
                                      StratifiedKFold, cross_val_predict)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42


def _scale(est) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", est)])


def classifiers() -> Dict[str, object]:
    return {
        "SVM": _scale(SVC(C=10, gamma=0.5, random_state=RANDOM_STATE)),
        "kNN": _scale(KNeighborsClassifier(n_neighbors=20)),
        "RFs": RandomForestClassifier(max_depth=20, random_state=RANDOM_STATE),
        "MLP": _scale(MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=800,
                                    early_stopping=True, n_iter_no_change=15,
                                    random_state=RANDOM_STATE)),
        "XGBoost": XGBClassifier(max_depth=5, subsample=0.7, min_child_weight=20,
                                 n_estimators=200, learning_rate=0.1, n_jobs=4,
                                 random_state=RANDOM_STATE, verbosity=0),
    }


def regressors() -> Dict[str, object]:
    return {
        "SVM": _scale(SVR(C=10, gamma=0.3)),
        "kNN": _scale(KNeighborsRegressor(n_neighbors=10)),
        "RFs": RandomForestRegressor(max_depth=20, random_state=RANDOM_STATE),
        "MLP": _scale(MLPRegressor(hidden_layer_sizes=(200, 100), max_iter=800,
                                   early_stopping=True, n_iter_no_change=15,
                                   random_state=RANDOM_STATE)),
        "XGBoost": XGBRegressor(max_depth=5, n_estimators=200, learning_rate=0.1,
                                n_jobs=4, random_state=RANDOM_STATE, verbosity=0),
    }


# --------------------------------------------------------------------------- #
# 数据容器
# --------------------------------------------------------------------------- #
@dataclass
class HeadDataset:
    """某个喷头的训练数据：特征矩阵 + 标签 + 分组。"""
    features: List[str]
    X: np.ndarray
    y_cls: np.ndarray                      # 0/1/2
    y_vel: np.ndarray                      # m/s，NaN 表示无液滴
    groups: np.ndarray                     # 会话名（用于分组交叉验证）
    meta: List[dict] = field(default_factory=list)
    head_name: str = ""
    head_type: str = ""
    #: 喷嘴分组（按孔号）。用于 **leave-nozzle-out** 评估——回答
    #: 「在部分喷嘴上训练/优化的波形，能否迁移到没见过的喷嘴」
    #: （PROJECT_RECORD §1.5 创新点 1）。与 `groups`（会话分组）正交：
    #: `groups` 防同源泄漏，`groups_nozzle` 防同孔泄漏。
    groups_nozzle: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        return self.X.shape[0]

    def vel_mask(self) -> np.ndarray:
        return ~np.isnan(self.y_vel)

    def describe(self) -> str:
        vm = self.vel_mask()
        from collections import Counter
        cnt = Counter(self.y_cls.tolist())
        lines = [
            "喷头: %s (HeadType=%s)" % (self.head_name, self.head_type),
            "样本: %d  特征: %d  会话: %d"
            % (self.n, len(self.features), len(set(self.groups.tolist()))),
            "特征: %s" % ", ".join(self.features),
            "类别分布: 不喷 %d / 单滴 %d / 卫星 %d"
            % (cnt.get(0, 0), cnt.get(1, 0), cnt.get(2, 0)),
            "速度可用样本: %d  范围 %.2f–%.2f m/s"
            % (vm.sum(), float(np.nanmin(self.y_vel[vm])) if vm.any() else 0,
               float(np.nanmax(self.y_vel[vm])) if vm.any() else 0),
        ]
        return "\n".join(lines)


def make_cv(n: int, y: np.ndarray, groups: Optional[np.ndarray]):
    """分组优先的交叉验证；组数不足时退化为分层 KFold，再不足用留一法。"""
    if groups is not None and len(set(groups.tolist())) >= 3:
        return GroupKFold(n_splits=min(5, len(set(groups.tolist())))), True
    _, counts = np.unique(y, return_counts=True)
    if n >= 2 * 5 and counts.min() >= 5:
        return StratifiedKFold(n_splits=5, shuffle=True,
                               random_state=RANDOM_STATE), False
    return LeaveOneOut(), False


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #
def evaluate_classification(X, y, groups=None, n_splits=5) -> dict:
    cv, is_group = make_cv(len(X), y, groups)
    kwargs = {"groups": groups} if is_group else {}
    out = {}
    for name, clf in classifiers().items():
        try:
            pred = cross_val_predict(clf, X, y, cv=cv, **kwargs)
        except Exception as exc:
            out[name] = {"error": "%s: %s" % (type(exc).__name__, exc)}
            continue
        p, r, f1, _ = precision_recall_fscore_support(
            y, pred, average="macro", zero_division=0)
        out[name] = {
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(p), "recall": float(r), "f1": float(f1),
            "confusion": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        }
    return out


def evaluate_regression(X, y, groups=None) -> dict:
    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]
    g = groups[mask] if groups is not None else None
    if g is not None and len(set(g.tolist())) >= 3:
        cv, is_group = GroupKFold(n_splits=min(5, len(set(g.tolist())))), True
    elif len(y) >= 10:
        cv, is_group = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), False
    else:
        cv, is_group = LeaveOneOut(), False
    kwargs = {"groups": g} if is_group else {}
    out = {}
    for name, reg in regressors().items():
        try:
            pred = cross_val_predict(reg, X, y, cv=cv, **kwargs)
        except Exception as exc:
            out[name] = {"error": "%s: %s" % (type(exc).__name__, exc)}
            continue
        out[name] = {
            "r2": float(r2_score(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(np.mean(np.abs(y - pred))),
        }
    return out


# --------------------------------------------------------------------------- #
# leave-nozzle-out 评估（§1.5 创新点 1：波形可迁移性）
# --------------------------------------------------------------------------- #
def evaluate_leave_nozzle_out(X, y, groups_nozzle) -> dict:
    """按喷嘴分组做交叉验证：训练集与测试集**不含同一个孔**。

    与按会话分组的评估回答的是**不同问题**：
      · 会话分组 → 「同一批喷嘴换个时间测，还准不准」（时间外推）；
      · 孔号分组 → 「换个喷嘴还准不准」（**空间外推**，工业化的真正痛点）。

    分组数不足 3 时无法构造有意义的折，返回空字典而不是退化成随机划分
    ——随机划分会给出虚高的指标（同一孔的 9 个会话样本同时进训练与测试）。
    """
    if groups_nozzle is None:
        return {}
    n_groups = len(set(groups_nozzle.tolist()))
    if n_groups < 3:
        return {}
    cv = GroupKFold(n_splits=min(5, n_groups))
    out = {}
    for name, reg in regressors().items():
        try:
            pred = cross_val_predict(reg, X, y, cv=cv, groups=groups_nozzle)
        except Exception as exc:
            out[name] = {"error": "%s: %s" % (type(exc).__name__, exc)}
            continue
        out[name] = {
            "r2": float(r2_score(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(np.mean(np.abs(y - pred))),
        }
    return out


def evaluate_leave_nozzle_out_cls(X, y, groups_nozzle) -> dict:
    """分类任务的 leave-nozzle-out 评估（对应 `evaluate_leave_nozzle_out` 的分类版）。

    与回归版回答同一个问题的分类面：**换到一个没见过的喷嘴，
    模型还能不能正确判断它会不会喷、喷几滴。**
    """
    if groups_nozzle is None:
        return {}
    n_groups = len(set(groups_nozzle.tolist()))
    if n_groups < 3:
        return {}
    cv = GroupKFold(n_splits=min(5, n_groups))
    out = {}
    for name, clf in classifiers().items():
        try:
            pred = cross_val_predict(clf, X, y, cv=cv, groups=groups_nozzle)
        except Exception as exc:
            out[name] = {"error": "%s: %s" % (type(exc).__name__, exc)}
            continue
        p, r, f1, _ = precision_recall_fscore_support(
            y, pred, average="macro", zero_division=0)
        out[name] = {
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(p), "recall": float(r), "f1": float(f1),
            "confusion": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        }
    return out


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def _fmt_table(title: str, res: dict, keys: Sequence[tuple], order: Sequence[str]):
    lines = ["### " + title, "",
             "| 模型 | " + " | ".join(k[0] for k in keys) + " |",
             "|---" * (len(keys) + 1) + "|"]
    for name in order:
        r = res.get(name)
        if not r:
            continue
        if "error" in r:
            lines.append("| %s | %s |" % (name, "失败: " + r["error"]))
            continue
        cells = []
        for disp, fmt, key in keys:
            v = r.get(key)
            cells.append("—" if v is None else (fmt % v if fmt else str(v)))
        lines.append("| %s | %s |" % (name, " | ".join(cells)))
    return "\n".join(lines)


CLS_KEYS = [("准确率", "%.4f", "accuracy"), ("宏F1", "%.4f", "f1"),
            ("宏精确", "%.4f", "precision"), ("宏召回", "%.4f", "recall")]
REG_KEYS = [("R²", "%.4f", "r2"), ("RMSE(m/s)", "%.4f", "rmse"), ("MAE(m/s)", "%.4f", "mae")]
MODEL_ORDER = ["SVM", "kNN", "RFs", "MLP", "XGBoost"]


def select_best_classifier(X, y, groups=None):
    """用与 evaluate_classification 同一套交叉验证选出最优分类模型。

    论文(MLP)只是候选之一，不是无条件最优——真实数据上完全可能 SVM/kNN/XGBoost 更优。
    这里不写死模型，保证「评估选出的 best」与「实际用于部署的模型」是同一个。
    """
    res = evaluate_classification(X, y, groups)
    name, _ = best_by(res, "accuracy")
    return (classifiers()[name] if name in classifiers() else classifiers()["MLP"]), name


def select_best_regressor(X, y, groups=None):
    """用与 evaluate_regression 同一套交叉验证选出最优回归模型（按 R²）。"""
    res = evaluate_regression(X, y, groups)
    name, _ = best_by(res, "r2")
    return (regressors()[name] if name in regressors() else regressors()["MLP"]), name


def fit_best_classifier(X, y, groups=None):
    """训练并返回最优分类模型（先按交叉验证选模型，再在全量数据上 fit）。"""
    est, _ = select_best_classifier(X, y, groups)
    est.fit(X, y)
    return est


def fit_best_regressor(X, y, groups=None):
    """训练并返回最优回归模型（先按交叉验证选模型，再在全量数据上 fit）。"""
    est, _ = select_best_regressor(X, y, groups)
    est.fit(X, y)
    return est


def best_by(res: dict, metric: str):
    """按指定指标挑最优模型，返回 (名字, 值)。失败模型不参与。"""
    ok = [(n, r[metric]) for n, r in res.items() if "error" not in r and metric in r]
    if not ok:
        return ("-", float("nan"))
    return max(ok, key=lambda t: t[1])


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8-sig") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)