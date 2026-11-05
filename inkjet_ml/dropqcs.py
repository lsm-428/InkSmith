# -*- coding: utf-8 -*-
"""DropQCS 规则集执行器 —— 解析 `Rules\\*.RulesConfig.xml` 并在测量数据上求值。

为什么需要它
------------
`ncalc.py` 只负责**单条表达式**求值。但规则集是**有状态**的两层结构：

    <NozzlesRule>  逐孔判，动作（Accept/Reject）会写 **[Result]**
    <HeadsRule>    整头判，同样可引用上一条规则的 [Result]

`[Result]` 是"上一条规则的动作结果"，**不是**纯表达式能表达的量，
必须由执行器维护状态机。预置的嵌套规则样例正是靠它实现
"先按位置拒，但速度高又放行"这种嵌套逻辑。

支持的动作（只读执行，不产生机台副作用）
------------------------------------------
    <Reject/> <Accept/>            → 只记录，不停用孔
    <Stop/> <Prompt/> <Warning/> <Error/>  → 只记录
    <Jet/> <Purge/> <Wipe/> <Suction/> <AdvanceCloth/> <DropQCS/>  → 只记录

⚠️ 本执行器**不做任何机器动作**（那是机台上的 DropQCSJob 的事）。
它只在离线数据上复现"哪些孔会被判不合格"，供 `ada_rank.py` 做跨组合排序。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ncalc import NCalcError, eval_expr

__all__ = ["RuleAction", "NozzleRule", "HeadRule", "RuleSet",
           "load_ruleset", "evaluate_nozzle", "evaluate_ruleset",
           "head_statistics"]


#: 执行器认识的全部动作标签 → 动作名
ACTION_TAGS = (
    "Reject", "Accept", "Stop", "Prompt", "Warning", "Error",
    "Jet", "Purge", "Wipe", "Suction", "AdvanceCloth", "DropQCS",
)


@dataclass
class RuleAction:
    """一条动作（执行器只记录，不执行）。"""
    kind: str
    message: str = ""


@dataclass
class NozzleRule:
    """逐孔规则：条件 + 动作。"""
    expression: str
    actions: List[RuleAction] = field(default_factory=list)
    nozzles: str = "all"          # Nozzles 属性："all" / "1-128" / "odd" …

    def eval_condition(self, variables: Dict[str, Any]) -> bool:
        return bool(eval_expr(self.expression, variables))

    def eval_condition_strict(self, variables: Dict[str, Any],
                               strict: bool = True) -> bool:
        """同 `eval_condition`，但可放宽"缺变量"为 None（用于整头级变量缺失的场景）。"""
        return bool(eval_expr(self.expression, variables, strict=strict))


@dataclass
class HeadRule:
    """整头规则：条件 + 动作。"""
    expression: str
    actions: List[RuleAction] = field(default_factory=list)

    def eval_condition(self, variables: Dict[str, Any]) -> bool:
        return bool(eval_expr(self.expression, variables))


@dataclass
class RuleSet:
    """一份规则集（一个 `<Rule>`）。"""
    name: str = ""
    nozzle_rules: List[NozzleRule] = field(default_factory=list)
    head_rules: List[HeadRule] = field(default_factory=list)
    source: str = ""

    def all_expressions(self) -> List[str]:
        return ([r.expression for r in self.nozzle_rules]
                + [r.expression for r in self.head_rules])


def _parse_actions(cond: ET.Element) -> List[RuleAction]:
    """把 `<Condition>` 下的动作子元素转成 `RuleAction` 列表。"""
    out: List[RuleAction] = []
    for child in cond:
        tag = child.tag
        if tag not in ACTION_TAGS:
            continue
        out.append(RuleAction(kind=tag, message=child.get("Message", "") or ""))
    return out


def load_ruleset(path: str) -> RuleSet:
    """读一份 `RulesConfig.xml`，返回 `RuleSet`。

    只取**第一个** `<Rule>`（预置样例都只有一个）。
    """
    p = Path(path)
    try:
        root = ET.parse(p).getroot()
    except (OSError, ET.ParseError) as e:
        raise NCalcError("规则文件读取失败：%s（%s）" % (p, e)) from e

    rule_el = root.find("Rule")
    if rule_el is None:
        raise NCalcError("规则文件里没有 <Rule>：%s" % p)

    rs = RuleSet(name=rule_el.get("Name", "") or p.stem, source=str(p))
    for cond in rule_el.findall("./NozzlesRule/Condition"):
        expr = cond.get("Expression")
        if not expr:
            continue
        rs.nozzle_rules.append(NozzleRule(
            expression=expr, actions=_parse_actions(cond),
            nozzles=cond.get("Nozzles", "all") or "all"))
    for cond in rule_el.findall("./HeadsRule/Condition"):
        expr = cond.get("Expression")
        if not expr:
            continue
        rs.head_rules.append(HeadRule(
            expression=expr, actions=_parse_actions(cond)))
    return rs


def evaluate_nozzle(ruleset: RuleSet, variables: Dict[str, Any],
                    strict: bool = False) -> Dict[str, Any]:
    """对**单个喷嘴**按规则集求值。

    按顺序跑 `NozzlesRule`，每次命中就把该规则的动作结果写进 `[Result]`
    （Accept → True，Reject → False），供后续规则用 `[Result]` 引用。

    `strict=False`（默认）：允许整头级变量（`[PrintheadSpeedAverage]` 等）
    在这一阶段缺失——它们要等整头汇总后才算得出。缺失变量求值为 `None`，
    与数值比较会抛 `NCalcError`（见 `ncalc.eval_expr`）。

    返回 `{"rejected": bool, "result": bool, "actions": [...], "rules_hit": [...]}`。
    """
    result = False          # [Result] 的初值（还没有任何规则动作）
    actions: List[RuleAction] = []
    hit: List[str] = []

    for rule in ruleset.nozzle_rules:
        v = dict(variables)
        v["Result"] = result
        if not rule.eval_condition_strict(v, strict):
            continue
        hit.append(rule.expression)
        actions.extend(rule.actions)
        for a in rule.actions:
            if a.kind == "Reject":
                result = False
            elif a.kind == "Accept":
                result = True

    return {
        "rejected": any(a.kind == "Reject" for a in actions),
        "result": result,
        "actions": actions,
        "rules_hit": hit,
    }


def head_statistics(nozzle_vars: List[Dict[str, Any]]) -> Dict[str, Any]:
    """整头级统计量 —— `[Printhead*]` 变量的来源。

    规则引擎的整头级变量包括：
        [PrintheadXCenterAverage] [PrintheadYCenterAverage]
        [PrintheadSpeedAverage]  [PrintheadAngleAverage]
        [PrintheadVolumeTotalAverage]
        [PrintheadSpeedStdDev]   [PrintheadAngleStdDev]
        [PrintheadVolumeTotalStdDev]

    **取整头所有孔的均值/标准差**（`AverageSpeedWithinMargin` 的语义：
    "速度在整头均值 ±20% 内"）。

    ⚠️ **"整头"的作用域 = 一个测量场景（一份 `Results.csv`）**（2026-11-04 明确）。
    机台上"整头"是一次测量里的**整个喷头**；本项目做的是**跨波形组合的离线排序**，
    一个扫描目录下有**成百上千份 `Results.csv`**（每份 = 一个波形组合 = 一次测量）。
    若把整头统计量跨 `Results.csv` 汇总，相对判据（如 `[PrintheadSpeedAverage]`）
    的参照系就变成"所有组合的池化均值"——那既不对应机台上的任何一次测量，
    也会让**每个组合自比自**（自己减去自己的均值），判据恒等价于单组合内的离散度，
    失去"这个孔是否偏离本头平均水平"的语义。

    故本函数的入参**必须是同一份 `Results.csv` 的逐孔变量**，
    整头统计量**逐场景独立计算**。
    例：`share` 下 9 个 `NOA89-test-*` = **9 个场景**，各有各的 `PrintheadSpeedAverage`
    （实测 4.20 / 2.74 / 3.69 / 3.81 / 3.94 / 3.83 / 3.03 / 3.19 / 2.50 m/s）。
    """
    import statistics

    out: Dict[str, Any] = {}

    def agg(prefix: str, key: str):
        vals = [nv[key] for nv in nozzle_vars
                if isinstance(nv.get(key), (int, float)) and not isinstance(nv.get(key), bool)]
        if not vals:
            out[prefix] = 0.0
            out[prefix + "StdDev"] = 0.0
            return
        out[prefix] = sum(vals) / len(vals)
        out[prefix + "StdDev"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0

    agg("PrintheadSpeedAverage", "SpeedAverage")
    agg("PrintheadAngleAverage", "AngleAverage")
    agg("PrintheadVolumeTotalAverage", "VolumeTotal")
    agg("PrintheadXCenterAverage", "XCenterAverage")
    agg("PrintheadYCenterAverage", "YCenterAverage")
    return out


def evaluate_ruleset(ruleset: RuleSet,
                     nozzle_vars: List[Dict[str, Any]]) -> Dict[str, Any]:
    """完整执行一份规则集（两级语义）。

    ⚠️ **入参 `nozzle_vars` 必须是"同一个测量场景"内的逐孔变量**（同一份 `Results.csv`）。
    整头级统计量只在这个场景内汇总——见 `head_statistics()` 的说明。

    ⚠️ **执行顺序很关键**：规则里**孔级规则也可能引用整头级变量**
    （`AverageSpeedWithinMargin` 就是：`[SpeedAverage] < ([PrintheadSpeedAverage]*0.8)`，
    `FastestDropletWithinScreenMargin` / `NestedNozzleRule` 同理），
    而整头级变量（`[PrintheadSpeedAverage]` 等）要**先由本场景所有孔的数据汇总**才能得出。
    所以正确顺序是：

        1. 汇总本场景全部孔 → 算整头级统计量
        2. 带上整头统计量跑**孔级**规则（逐孔）
        3. 用孔级结果（被拒孔数）跑**整头**规则

    这与机台上的执行顺序一致（DropQCSJob 先测完整头，再判）。
    """
    # 1) 整头级统计量（先算，供孔级规则引用）
    stats = head_statistics(nozzle_vars)

    # 2) 孔级规则（带上整头统计量）
    per_nozzle = []
    for nv in nozzle_vars:
        v = dict(nv)
        v.update(stats)
        r = evaluate_nozzle(ruleset, v, strict=False)
        v["_rejected"] = r["rejected"]
        per_nozzle.append((v, r))

    n = len(per_nozzle)
    rejected = sum(1 for _v, r in per_nozzle if r["rejected"])

    # 3) 整头规则
    v = dict(stats)
    v.update({
        "NrNozzles": n,
        "NrRejectedNozzles": rejected,
        "NrMaxRejectedAdjacentNozzles": 0,
        "PercentageAnalyzed": 100.0 if n else 0.0,
        "Result": False,
    })
    head_actions: List[RuleAction] = []
    head_hit: List[str] = []
    for rule in ruleset.head_rules:
        if not rule.eval_condition(v):
            continue
        head_hit.append(rule.expression)
        head_actions.extend(rule.actions)

    return {
        "n_nozzles": n,
        "n_rejected": rejected,
        "nozzles": per_nozzle,
        "head_actions": head_actions,
        "head_rules_hit": head_hit,
        "head_variables": v,
    }
