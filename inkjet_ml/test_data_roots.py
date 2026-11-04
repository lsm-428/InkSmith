# -*- coding: utf-8 -*-
"""`--data` 数据接入链路的自检（标准库 unittest，无需额外依赖）。

运行：
    python -m unittest inkjet_ml.test_data_roots -v

覆盖（2026-11-04 新增，锁住 `--data` 打通闭环后修掉的 5 个真问题）：
  1. `resolve_roots()`：默认 / 追加 / 替换 三种语义；
  2. **多根 session 名去冲突**——否则分组交叉验证会把不同来源当同一组，
     导致分组泄漏、指标虚高；
  3. **真实 ADA 落盘的波形参数能被取到**
     （V4730 的 ADA 写 `Pulse0Voltage` / `PulseSlewRate`，
      而 `share` 那批是 AutoADA 写的，两套标签名都认）；
  4. **settings 保留全部原始键**——KM1024i 的分段宽度标签
     （`KM1024iPulse0Section0Width`）不在固定键表里，
     只留固定键会让真实数据被静默跳过；
  5. **单组时 CV 不得退化成留一法**——机台一次扫描 = 1 个 session，
     留一法会让 1024 孔跑 5120 次训练（实测 >15 分钟跑不完）。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from .dataset import DEFAULT_ROOTS, load_all, resolve_roots

#: V4730 新基线机台的 ADA 落盘根（真实数据，含 KM1024i / Canon N2N）
V4730_SCREEN = (r"D:\Oleton\Inkjet\LP50 V4730 4-2026 & Canon FAT"
                r"\ScreenCaptures")
#: 本项目默认数据源
SHARE_ROOT = r"D:\Gitee\LP50\ML\share"


class TestResolveRoots(unittest.TestCase):
    """`resolve_roots()` 的三种语义。"""

    def test_no_extra_gives_defaults(self):
        """不给 `--data` → 默认根（去重后）。"""
        r = resolve_roots(None)
        self.assertEqual(r, resolve_roots([]))
        # 默认根里「本机绝对路径」与「仓库相对路径」在开发机上指向同一目录，
        # 故去重后可能少于 DEFAULT_ROOTS 的长度——断言的是**去重语义**。
        self.assertEqual(len(r), len({os.path.normcase(os.path.abspath(x))
                                      for x in DEFAULT_ROOTS}))
        self.assertTrue(r)

    def test_extra_is_merged_with_defaults(self):
        new = r"D:\some\new\root"
        r = resolve_roots([new])
        defaults = resolve_roots(None)
        self.assertEqual(len(r), len(defaults) + 1)
        self.assertEqual(r[-1], new)
        # 默认根在前，顺序稳定 ⇒ 输出可复现
        self.assertEqual(r[:len(defaults)], defaults)

    def test_replace_drops_defaults(self):
        r = resolve_roots([r"D:\only\this"], replace=True)
        self.assertEqual(r, [r"D:\only\this"])

    def test_duplicate_paths_are_deduped(self):
        """同一目录（含绝对/相对两种写法）只保留一次。"""
        one = r"D:\same\root"
        r = resolve_roots([one, one], replace=True)
        self.assertEqual(len(r), 1)
        # 与默认根重复的也要去重
        r2 = resolve_roots([DEFAULT_ROOTS[0]])
        self.assertEqual(len(r2), len(resolve_roots(None)))
        # 绝对路径与相对路径指向同一目录 → 也只算一次
        rel = os.path.relpath(DEFAULT_ROOTS[0], os.getcwd())
        r3 = resolve_roots([rel], replace=True)
        self.assertEqual(len(r3), 1)


class TestSessionNoCollision(unittest.TestCase):
    """多根时 session 名必须全局唯一（防分组交叉验证泄漏）。"""

    @unittest.skipUnless(os.path.isdir(SHARE_ROOT), "share 数据不可用")
    def test_same_named_dirs_in_two_roots_do_not_collide(self):
        tmp = tempfile.mkdtemp(prefix="lp50roots_")
        try:
            # 造一个与 share 里 NOA89-test-01 **同名** 的根
            src = os.path.join(SHARE_ROOT, "NOA89-test-01")
            dst = os.path.join(tmp, "NOA89-test-01")
            os.makedirs(dst)
            for f in os.listdir(src):
                if f == "Results.csv" or f == "Measurement.xml" or f.startswith("noz"):
                    shutil.copy(os.path.join(src, f), os.path.join(dst, f))
            ms = load_all([SHARE_ROOT, tmp])
            sess = [m.session for m in ms]
            self.assertEqual(len(sess), len(set(sess)),
                             "session 名撞名了：%s" % sorted(sess))
            # 两个来源都还在
            self.assertTrue(any(s.startswith("share/") for s in sess))
            self.assertTrue(any("NOA89-test-01" in s and not s.startswith("share/")
                                for s in sess))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipUnless(os.path.isdir(SHARE_ROOT), "share 数据不可用")
    def test_single_root_keeps_plain_session_names(self):
        """单根时不加前缀——保持既有报告里的会话名不变。"""
        ms = load_all([SHARE_ROOT])
        self.assertTrue(ms)
        for m in ms:
            self.assertNotIn("/", m.session.replace(os.sep, "/"))

    @unittest.skipUnless(os.path.isdir(SHARE_ROOT), "share 数据不可用")
    def test_same_dir_via_two_spellings_not_counted_twice(self):
        """同一目录用「绝对路径 + 相对路径」两种写法指认 → 只算一次。

        `DEFAULT_ROOTS` 正是这种情形（本机绝对路径 + 仓库相对路径）。
        若不去重，108 孔会被算成 216 孔，样本翻倍、指标失真。
        """
        abs_root = os.path.abspath(SHARE_ROOT)
        rel_root = os.path.relpath(abs_root, os.getcwd())
        if os.path.normcase(abs_root) == os.path.normcase(os.path.abspath(rel_root)):
            # 相对路径必须真的指向同一目录，否则该用例无意义
            self.assertTrue(os.path.isdir(rel_root))
        one = load_all([abs_root])
        two = load_all([abs_root, rel_root])
        self.assertEqual(len(one), len(two),
                         "同一目录被计入两次：%d vs %d" % (len(one), len(two)))
        self.assertEqual(sum(m.n for m in one), sum(m.n for m in two))


class TestRealAdaSettings(unittest.TestCase):
    """真实 ADA 落盘的设定值必须能取到（否则新扫的数据白采）。"""

    @unittest.skipUnless(os.path.isdir(V4730_SCREEN), "V4730 落盘根不可用")
    def test_km1024i_section_widths_are_read(self):
        """KM1024i 的分段宽度标签 `KM1024iPulse0Section0Width` 必须被识别。"""
        from .heads import km1024i
        ms = [m for m in load_all([V4730_SCREEN]) if m.head_name == "KM1024i"]
        if not ms:
            self.skipTest("V4730 上暂无 KM1024i 数据")
        mat = km1024i.collect_sections(ms[0])
        self.assertIsNotNone(mat,
                             "真实落盘的分段宽度取不到 → build() 会返回 None")
        import numpy as np
        self.assertGreaterEqual(int((~np.isnan(mat)).sum()), km1024i.MIN_SECTIONS)

    @unittest.skipUnless(os.path.isdir(SHARE_ROOT), "share 数据不可用")
    def test_samba_waveform_settings_are_complete(self):
        """Samba 的 5 个波形特征一个都不能缺（缺了 features_of() 返回 None）。"""
        ms = [m for m in load_all([SHARE_ROOT]) if m.head_name == "DMC Samba"]
        self.assertTrue(ms)
        for m in ms:
            for k in ("pulse0_voltage", "pulse0_width", "pulse0_tail",
                      "slewrate", "voltage_supply"):
                self.assertIsNotNone(m.settings.get(k),
                                     "%s 的 %s 取不到" % (m.session, k))

    @unittest.skipUnless(os.path.isdir(V4730_SCREEN), "V4730 落盘根不可用")
    def test_settings_keep_all_raw_keys(self):
        """settings 必须保留 Measurement.xml / CSV 尾部的**全部**原始键。"""
        ms = load_all([V4730_SCREEN])
        self.assertTrue(ms)
        m = ms[0]
        # KM1024i 的分段宽度标签必须在内（否则 KM1024i 模块取不到）
        self.assertTrue(any(k.startswith("KM1024iPulse") for k in m.settings),
                        "原始键没被保留，head 模块取不到自己的参数")


class TestCVFolds(unittest.TestCase):
    """单组时 CV 不得退化成留一法（机台一次扫描 = 1 个 session）。"""

    def test_single_group_uses_stratified_not_leave_one_out(self):
        import numpy as np
        from sklearn.model_selection import LeaveOneOut, StratifiedKFold
        from .headml import make_cv
        y = np.array([0] * 4 + [1] * 1020)
        g = np.array(["only-one-session"] * len(y))
        cv, is_group = make_cv(len(y), y, g)
        self.assertNotIsInstance(cv, LeaveOneOut,
                                 "单组却用了留一法 → 1024 折 × 5 模型会跑不完")
        self.assertIsInstance(cv, StratifiedKFold)
        self.assertFalse(is_group)

    def test_three_groups_still_uses_group_kfold(self):
        import numpy as np
        from sklearn.model_selection import GroupKFold
        from .headml import make_cv
        y = np.array([0] * 6 + [1] * 12)
        g = np.array(["a"] * 6 + ["b"] * 6 + ["c"] * 6)
        cv, is_group = make_cv(len(y), y, g)
        self.assertIsInstance(cv, GroupKFold)
        self.assertTrue(is_group)


if __name__ == "__main__":
    unittest.main(verbosity=2)
