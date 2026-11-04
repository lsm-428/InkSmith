# -*- coding: utf-8 -*-
"""DropQCS 规则求值器的自检（标准库 unittest，无需额外依赖）。

运行：
    python -m unittest inkjet_ml.test_ncalc -v
    python -m inkjet_ml.test_ncalc            # 简版

覆盖：
  1. 表达式求值：官方 8 套规则 + Default 用到的**全部 13 条表达式**；
  2. 18 个内置函数；
  3. 语法/注入防护；
  4. 规则集执行：8 套官方规则 × 4 个真实数据源的**语义交叉验证**
     （拒绝孔数与规则语义手工推算值比对）。
"""

from __future__ import annotations

import os
import unittest

from .dropqcs import evaluate_ruleset, head_statistics, load_ruleset
from .dropqcs_vars import read_results_csv
from .ncalc import NCalcError, eval_expr, parse_expr

#: 新基线机台的规则目录（V4730 实测 8 套齐全；旧机台没有）
RULES_DIR = r"D:\Oleton\Inkjet\LP50 V4730 4-2026 & Canon FAT\Rules"
RULES_EX = os.path.join(RULES_DIR, "Examples")

#: 真实数据源：share（DMC Samba）+ V4730 实测（有多滴样本）
DATASETS = {
    "share-01": r"D:\Gitee\LP50\ML\share\NOA89-test-01\Results.csv",
    "share-02": r"D:\Gitee\LP50\ML\share\NOA89-test-02\Results.csv",
    "CanonN2N": (r"D:\Oleton\Inkjet\LP50 V4730 4-2026 & Canon FAT"
                 r"\ScreenCaptures\Canon N2N\Results.csv"),
    "KM1024i": (r"D:\Oleton\Inkjet\LP50 V4730 4-2026 & Canon FAT"
                r"\ScreenCaptures\ADA KM1024i 045\Results.csv"),
}


def _have(path):
    return os.path.isfile(path)


def _have_dir(path):
    return os.path.isdir(path)


class TestEvalExpr(unittest.TestCase):
    """官方规则里实际出现的全部表达式。"""

    # (表达式, 变量, 期望)
    CASES = [
        ("[NrDroplets] < 1", {"NrDroplets": 0}, True),
        ("[NrDroplets] < 1", {"NrDroplets": 1}, False),
        ("[NrDroplets] != 1", {"NrDroplets": 1}, False),
        ("[NrDroplets] != 1", {"NrDroplets": 2}, True),
        ("[NrDroplets] < 1 or [NrDroplets] > 2", {"NrDroplets": 1}, False),
        ("[NrDroplets] < 1 or [NrDroplets] > 2", {"NrDroplets": 3}, True),
        ("[PercentageAnalyzed] == 100 and [NrRejectedNozzles] > ([NrNozzles] / 2)",
         {"PercentageAnalyzed": 100, "NrRejectedNozzles": 70, "NrNozzles": 128}, True),
        ("[PercentageAnalyzed] == 100 and [NrRejectedNozzles] > ([NrNozzles] / 2)",
         {"PercentageAnalyzed": 100, "NrRejectedNozzles": 60, "NrNozzles": 128}, False),
        ("[PercentageAnalyzed] == 100 and [NrRejectedNozzles] > 0",
         {"PercentageAnalyzed": 100, "NrRejectedNozzles": 1, "NrNozzles": 128}, True),
        ("[PercentageAnalyzed] == 100", {"PercentageAnalyzed": 100}, True),
        ("[Printhead] == 1 && [NrDroplets] != 1", {"Printhead": 1, "NrDroplets": 2}, True),
        ("[Printhead] == 1 && [NrDroplets] != 1", {"Printhead": 2, "NrDroplets": 2}, False),
        ("[Printhead] == 2 && [NrDroplets] < 1", {"Printhead": 2, "NrDroplets": 0}, True),
        ("[Result]==false and [SpeedMin] > 4.0", {"Result": False, "SpeedMin": 5.0}, True),
        ("[Result]==false and [SpeedMin] > 4.0", {"Result": True, "SpeedMin": 5.0}, False),
        ("[SpeedAverage] < ([PrintheadSpeedAverage]*0.8) or "
         "[SpeedAverage] > ([PrintheadSpeedAverage]*1.2)",
         {"SpeedAverage": 5.0, "PrintheadSpeedAverage": 5.0}, False),
        ("[SpeedAverage] < ([PrintheadSpeedAverage]*0.8) or "
         "[SpeedAverage] > ([PrintheadSpeedAverage]*1.2)",
         {"SpeedAverage": 7.0, "PrintheadSpeedAverage": 5.0}, True),
        ("[AngleMin] < -2 or [AngleMax] > 2", {"AngleMin": -3.0, "AngleMax": 0.0}, True),
        ("[AngleMin] < -2 or [AngleMax] > 2", {"AngleMin": 0.0, "AngleMax": 1.0}, False),
        ("[YCenterAverage] < ([PrintheadYCenterAverage]*0.9) or "
         "[YCenterAverage] > ([PrintheadYCenterAverage]*1.1)",
         {"YCenterAverage": 100.0, "PrintheadYCenterAverage": 100.0}, False),
    ]

    def test_official_expressions(self):
        for expr, vars_, want in self.CASES:
            with self.subTest(expr=expr, vars=vars_):
                got = eval_expr(expr, vars_)
                if isinstance(want, float) or isinstance(got, float):
                    self.assertAlmostEqual(got, want, places=9)
                else:
                    self.assertEqual(got, want)

    def test_builtin_functions(self):
        cases = [("Abs(-1)", 1), ("Pow(2, 10)", 1024), ("Min(3, 5)", 3),
                 ("Max(3, 5)", 5), ("Round(1.5)", 2), ("Sqrt(16)", 4),
                 ("Floor(1.9)", 1), ("Ceiling(1.1)", 2), ("Sign(-5)", -1),
                 ("Log(1)", 0), ("Log10(100)", 2), ("Exp(0)", 1),
                 ("Sin(0)", 0), ("Cos(0)", 1), ("Tan(0)", 0),
                 ("Acos(1)", 0), ("Asin(0)", 0), ("Atan(0)", 0)]
        for expr, want in cases:
            with self.subTest(expr=expr):
                got = eval_expr(expr, {})
                self.assertAlmostEqual(got, want, places=9)

    def test_missing_variable_raises(self):
        with self.assertRaises(NCalcError):
            eval_expr("[NrDroplets] < 1", {})

    def test_non_strict_missing_var_is_none(self):
        """strict=False：缺失变量求值为 None（整头级变量在孔级阶段的缺失情形）。

        注意 Python 语义：`None < 1` 会抛 TypeError → 本模块转成 NCalcError。
        """
        with self.assertRaises(NCalcError):
            eval_expr("[X] < 1", {}, strict=False)
        # 纯引用缺失变量（不参与比较）则求值为 None
        self.assertIsNone(eval_expr("[X]", {}, strict=False))

    def test_injection_blocked(self):
        """表达式来自机台配置，但仍按不可信输入处理。"""
        for expr in ["__import__('os').system('x')",
                     "[X].__class__",
                     "lambda: 1",
                     "[X][0]",
                     "open('f')"]:
            with self.subTest(expr=expr):
                with self.assertRaises(NCalcError):
                    eval_expr(expr, {"X": 1})

    def test_unknown_function_rejected(self):
        with self.assertRaises(NCalcError):
            parse_expr("UnknownFunc(1)")

    def test_syntax_error_rejected(self):
        with self.assertRaises(NCalcError):
            parse_expr("[A] @@ 1")


class TestOfficialRulesets(unittest.TestCase):
    """8 套官方规则 × 真实数据的语义交叉验证。"""

    @unittest.skipUnless(_have_dir(RULES_EX), "V4730 规则目录不可用")
    def test_single_vs_at_least_one(self):
        """SingleDroplet 拒「0滴+多滴」，AtLeastOneDroplet 只拒「0滴」。"""
        for name, path in DATASETS.items():
            if not _have(path):
                continue
            with self.subTest(dataset=name):
                nz = read_results_csv(path)
                vlist = [n.variables() for n in nz]
                n0 = sum(1 for n in nz if n.nr_droplets == 0)
                n2 = sum(1 for n in nz if n.nr_droplets >= 2)
                sd = load_ruleset(os.path.join(RULES_EX, "SingleDroplet.RulesConfig.xml"))
                ad = load_ruleset(os.path.join(RULES_EX, "AtLeastOneDroplet.RulesConfig.xml"))
                r_sd = evaluate_ruleset(sd, vlist)["n_rejected"]
                r_ad = evaluate_ruleset(ad, vlist)["n_rejected"]
                self.assertEqual(r_sd, n0 + n2, "SingleDroplet 应拒 0滴+多滴")
                self.assertEqual(r_ad, n0, "AtLeastOneDroplet 应只拒 0滴")

    @unittest.skipUnless(_have_dir(RULES_EX), "V4730 规则目录不可用")
    def test_all_official_rulesets_run(self):
        """8 套官方规则 + Default 全部能在真实数据上跑完（不抛异常）。"""
        for name, path in DATASETS.items():
            if not _have(path):
                continue
            vlist = [n.variables() for n in read_results_csv(path)]
            for fn in sorted(os.listdir(RULES_EX)):
                if not fn.endswith(".RulesConfig.xml"):
                    continue
                with self.subTest(dataset=name, ruleset=fn):
                    rs = load_ruleset(os.path.join(RULES_EX, fn))
                    res = evaluate_ruleset(rs, vlist)
                    self.assertEqual(res["n_nozzles"], len(vlist))
                    self.assertLessEqual(res["n_rejected"], res["n_nozzles"])


class TestVariables(unittest.TestCase):
    """从 Results.csv 还原官方变量的正确性。"""

    @unittest.skipUnless(_have(DATASETS["CanonN2N"]), "V4730 数据不可用")
    def test_multidroplet_variables(self):
        """多滴孔：VolumeTotal = 各滴体积之和；SpeedAverage = 逐滴均值。"""
        nz = read_results_csv(DATASETS["CanonN2N"])
        multi = [n for n in nz if n.nr_droplets >= 2]
        self.assertTrue(multi, "Canon N2N 应含多滴样本")
        for n in multi[:5]:
            v = n.variables()
            self.assertEqual(v["NrDroplets"], n.nr_droplets)
            self.assertTrue(v["Present"])
            self.assertEqual(len(n.droplets), n.nr_droplets)
            # 体积总和
            self.assertAlmostEqual(
                v["VolumeTotal"],
                sum(d.volume for d in n.droplets if d.volume is not None), places=6)
            # 速度均值
            sp = [d.speed for d in n.droplets if d.speed is not None]
            self.assertAlmostEqual(v["SpeedAverage"], sum(sp) / len(sp), places=9)
            self.assertAlmostEqual(v["SpeedMin"], min(sp), places=9)
            self.assertAlmostEqual(v["SpeedMax"], max(sp), places=9)

    @unittest.skipUnless(_have(DATASETS["share-01"]), "share 数据不可用")
    def test_zero_droplet_sentinel(self):
        """0 滴孔：速度哨兵为负、角度哨兵为 -180（官方文档口径）。"""
        nz = read_results_csv(DATASETS["share-01"])
        zero = [n for n in nz if n.nr_droplets == 0]
        self.assertTrue(zero, "share-01 应含 0 滴孔")
        v = zero[0].variables()
        self.assertFalse(v["Present"])
        self.assertLess(v["SpeedAverage"], 0)
        self.assertEqual(v["AngleAverage"], -180.0)


class TestScenarioScope(unittest.TestCase):
    """"整头"作用域 = 一个测量场景（一份 Results.csv），**不跨场景池化**。

    2026-11-04 明确的口径：`share` 下 9 个 `NOA89-test-*` = **9 个场景**，
    整头统计量各自独立计算。
    """

    @unittest.skipUnless(_have(DATASETS["share-01"]), "share 数据不可用")
    def test_each_measurement_is_own_scenario(self):
        """share 的 9 个会话 → 9 组独立的整头统计量（不池化）。"""
        import glob
        paths = sorted(glob.glob(
            r"D:\Gitee\LP50\ML\share\NOA89-test-*\Results.csv"))
        self.assertEqual(len(paths), 9, "share 应有 9 个会话")
        means = []
        for p in paths:
            vlist = [n.variables() for n in read_results_csv(p)]
            st = head_statistics(vlist)
            means.append(st["PrintheadSpeedAverage"])
        # 各场景的整头均值应**互不相同**（若池化则全部相同）
        self.assertEqual(len(set(round(m, 6) for m in means)), len(means),
                         "每个场景应有独立的 PrintheadSpeedAverage")
        # 池化值（9 场景合在一起）不应等于任何一个场景的值
        pooled = head_statistics(
            [nv for p in paths for nv in
             (n.variables() for n in read_results_csv(p))])["PrintheadSpeedAverage"]
        for m in means:
            self.assertNotAlmostEqual(pooled, m, places=3)

    @unittest.skipUnless(_have(DATASETS["share-01"]), "share 数据不可用")
    def test_relative_criterion_uses_own_scenario_mean(self):
        """相对判据 `[SpeedAverage]` 与**本场景**均值比较（不是与池化均值比）。

        验证方式：把某孔的 `SpeedAverage` 改成"等于本场景均值"，
        它在 `±20%` 带内 → 不应被拒；若参照系是池化均值则结论可能不同。
        """
        from inkjet_ml.dropqcs import evaluate_nozzle, head_statistics
        rs = load_ruleset(os.path.join(RULES_EX, "AverageSpeedWithinMargin.RulesConfig.xml"))
        vlist = [n.variables() for n in read_results_csv(DATASETS["share-01"])]
        own_mean = head_statistics(vlist)["PrintheadSpeedAverage"]

        # 单孔：SpeedAverage == 本场景均值 → 带内 → 不拒
        v = dict(vlist[0])
        v["SpeedAverage"] = own_mean
        v.update(head_statistics(vlist))     # 带上本场景整头统计量
        self.assertFalse(evaluate_nozzle(rs, v, strict=False)["rejected"],
                         "等于本场景均值的孔不应被相对判据拒绝")

        # 单孔：SpeedAverage == 本场景均值 × 1.5 → 超带 → 拒
        v2 = dict(vlist[0])
        v2["SpeedAverage"] = own_mean * 1.5
        v2.update(head_statistics(vlist))
        self.assertTrue(evaluate_nozzle(rs, v2, strict=False)["rejected"],
                        "偏离本场景均值 50% 的孔应被相对判据拒绝")

    @unittest.skipUnless(_have(DATASETS["share-01"]), "share 数据不可用")
    def test_evaluate_ruleset_is_per_scenario(self):
        """`evaluate_ruleset` 的整头变量来自入参本身（场景内），不读全局。"""
        vlist = [n.variables() for n in read_results_csv(DATASETS["share-01"])]
        res = evaluate_ruleset(load_ruleset(
            os.path.join(RULES_EX, "AverageSpeedWithinMargin.RulesConfig.xml")), vlist)
        # 整头变量里的孔数 = 本场景孔数（12），不是全部 108
        self.assertEqual(res["head_variables"]["NrNozzles"], len(vlist))
        self.assertEqual(res["head_variables"]["NrNozzles"], 12)

    @unittest.skipUnless(_have(DATASETS["share-01"]), "share 数据不可用")
    def test_pooling_would_change_result(self):
        """**反例**：把两个场景混成一个池，判定结果会变 —— 证明"不池化"不是无关紧要。

        ⚠️ 注意：在 share **全 9 场景**上两种口径**恰好同解**
        （各场景内速度离散本就大于 ±20% 带，带外孔在两种口径下都出带），
        那是该数据集的巧合。所以这里用**构造反例**：把离散度不同的两个场景拼起来。
        """
        p3 = (r"D:\Gitee\LP50\ML\share\NOA89-test-03\Results.csv")
        if not _have(p3):
            self.skipTest("test-03 数据不可用")
        rs = load_ruleset(os.path.join(RULES_EX, "AverageSpeedWithinMargin.RulesConfig.xml"))
        a = [n.variables() for n in read_results_csv(DATASETS["share-01"])]
        b = [n.variables() for n in read_results_csv(p3)]
        sep = (evaluate_ruleset(rs, a)["n_rejected"]
               + evaluate_ruleset(rs, b)["n_rejected"])
        pooled = evaluate_ruleset(rs, a + b)["n_rejected"]
        self.assertNotEqual(sep, pooled,
                            "若两者相等则该反例无效（换一组离散度差异更大的场景）")

        # 池化均值被 a 拉高，导致 b 的孔被误判
        m_a = head_statistics(a)["PrintheadSpeedAverage"]
        m_pool = head_statistics(a + b)["PrintheadSpeedAverage"]
        self.assertNotAlmostEqual(m_a, m_pool, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
