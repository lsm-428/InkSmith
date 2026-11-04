# -*- coding: utf-8 -*-
"""ADA + DropQCS：快速找最优喷射波形的实验设计辅助（全部基于本机真实配置）。

用户的两条约束
--------------
1. **不计算耗时**。ADA 的时间由机器固有点火频率决定，与扫描参数组合数不是线性关系，
   本模块不输出任何时间估算，只输出"要填什么"和"会得到多少张图"。
2. **只用两层闭环，不引入 APO**。APO（99 Automated-Print-Optimization）只是"打印规划"：
   它把图案打到衬底上、靠人眼评图，**没有任何判据，也无法一次读取视觉结果**，
   所以不进自动寻优回路。能自动闭环的只有：
     * **ADA（95 Advanced Drop Analysis）** —— 勾选参数 → 全析因扫描 →
       机器自己读出 `Nr. of droplets / Speed / Volume / Angle`（Results.csv + noz*.xml，
       CSV/XML 都可程序化处理，不需要人眼看图）
     * **DropQCS（96）** —— 规则引擎对这些量做判据（NCalc 表达式），不合格 `Reject`，
       必要时可带维护动作并按 `MaxRepeat` 重测
   完全不涉及 Z / 黏度 / 表面张力等任何测不到的物理量。

证据来源（逐条可核）
--------------------
A. ADA 扫描机制 —— `C:\\LP50\\Recipes\\95 ADA\\Advanced-Drop-Analysis.cs`
   6016 `GetSeparateParameterValues`：`a:b:c` → for(v=a; v<=c; v+=b)（闭区间）；
        `a:b` → step=1；`;` 分隔多个序列/单值；Nozzle 走字符串解析。
   5364 `ProcessParameterValues`：只对**勾选**的参数递归全析因笛卡尔积，
        每个组合在最后一个参数处 `Dropview()` 成像一次。
   4882 `ConfigParameters`：`IsSelected()` 为假的参数一律移除 → 不勾选 = 不扫描。
   6016 同文件的引号感知属性解析（本模块沿用 explore/parse_recipes.py 的实现）。
B. ADA 扫描维度与范围 —— `C:\\LP50\\Recipes\\95 ADA\\Advanced-Drop-Analysis-DMCSamba.xml`。
C. DropQCS 判据语法 —— `<LP50 V4730>\\Rules\\Default.RulesConfig.xml`
   与 `Rules\\Examples\\*.RulesConfig.xml`
   （NozzlesRule / HeadsRule + NCalc 表达式 + Reject / Accept / Stop / Jet 动作）。
   ⚠️ 注意：`C:\\LP50` 这台机器上 `DropQCSJob` / `RulesEngine` 在 438 个程序集里
   不存在（`C:\\LP50\\Recipes\\95 ADA-APA\\apa_measure.cs:24-26` 逐字记载），
   判据要跑起来需使用带 ScriptSupport.dll(含 DropQCSJob) 的安装。
D. 速度测量 —— ADA.cs:4872-4873 `m_dDelay2 = m_dDelay1 + LedDelayIncr`（双闪成像），
   结果落盘在 Results.csv 的 `Speed`[m/s] 列与 noz*.xml 的 `<Speed>` 节点。
E. 刻度参考 —— ADA.cs 同族流程 `Automated-Print-Optimization-*.xml`（APO）里 SUSS 对
   同样这些脉冲参数给的是更窄的默认区间（如 Pulse 宽 `1.4:0.2:4` 而非 `1:0.1:4`）。
   **仅作为"该用多密的刻度"的参考，不代表 APO 流程本身参与寻优。**

用法
----
    python -m inkjet_ml.ada --plan screen      # 阶段1：粗筛可行波形域
    python -m inkjet_ml.ada --plan narrow     # 阶段2：围绕最优点细化
    python -m inkjet_ml.ada --plan uniform    # 阶段3：定形后做喷嘴一致性
    python -m inkjet_ml.ada --config <计划名>  # 输出可直接照填的 ADA recipe 片段
    python -m inkjet_ml.ada --rules           # 输出可直接照填的 DropQCS 判据文件
    python -m inkjet_ml.ada --dims            # 列出全部可扫描维度与默认范围
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# LP50 安装位置
#
# **基线 = V4730 (HMI 4.7.3)**，本项目所有参数以此版为准。
#   4.7.3 相对 4.4.5 的实质差异（逐文件核实，见 PROJECT_RECORD §8.7）：
#     · 新增 CanonC29 喷头流程；KM204 流程被移除
#     · KM1024i：11 → 115 个可扫维度（新增 105 个分段宽度 + 2 个驱动供电调整，
#       同时**移除** Width / VoltageAdjustLeft / VoltageAdjustRight）
#     · KM512：DropPeriod / OnPulseWidth / OffPulseWidth / PhaseLength
#       的流程默认值被改小（80:10:150→10:20 等），4.4.5 的旧值只出现在
#       4.7.3 的 recipe 里 —— 用哪套必须按实机确认
#     · DMCSamba / SpectraSClass：两版完全一致
# --------------------------------------------------------------------------- #
NEW_LP50 = r"D:\Oleton\Inkjet\LP50 V4730 4-2026 & Canon FAT"
ADA_DIR_C = NEW_LP50 + r"\Recipes\95 ADA"
ADA_FLOW = ADA_DIR_C + r"\Advanced-Drop-Analysis-DMCSamba.xml"
# 旧安装（LP50-4450, HMI 4.4.5）仅作对照：它的 DropQCSJob/RulesEngine 缺失
OLD_LP50 = r"C:\LP50"
RULES_DIR = NEW_LP50 + r"\Rules"

# --------------------------------------------------------------------------- #
# 各喷头的**物理孔数**
#
# ⚠️ 必须与 `DropAnalysis.Nozzle` 参数值严格区分——这是两个不同概念：
#
#   · **物理孔数**（本表）：喷头实际有多少个喷嘴，是硬件事实；
#   · **`DropAnalysis.Nozzle` 参数**：**要测量哪些喷嘴**的序列（`1:12`、`1;2;3`…），
#     是扫描范围，可以只测一部分。ADA 源码（`Advanced-Drop-Analysis.cs:5140`）
#     把它当 "Active Nozzle" 解析；`:6018` 明确它按字符串而非数字解析。
#
# 所以 flow 默认的 `1:128` **不是**"这机器有 128 孔"，
# 而是"默认要测 1..128 号孔"——对 KM512（512 孔）它就只测了前 1/4。
#
# 依据（2026-11-03 核实）：
#   · DMC Samba   → 规格书 *Addressable Nozzles – 12 jets*；
#                   实机 recipe `Nozzle[0] = 1:12`；实测每会话覆盖 1..12 孔。
#   · KM512       → LP50 培训手册 *512 nozzles: 256 each side, 3 rows*；
#                   实机 `Configuration/config/DataGenerator.xml` 的
#                   NozzleLayout 3/4 首喷嘴号 257/258/259（左右分侧）。
#   · KM1024i     → NozzleLayout 19 首喷嘴号 513/514/515；
#                   flow 与 recipe 均为 `1:1024`。
#   · Spectra S   → 手册 *The SE/SM/SL printheads have 128 nozzles*
#                   （0.508 mm 间距 / 50 DPI，A/B 两组压电分别驱动偶/奇孔）。
#   · SX3         → 128 喷嘴；**已停产**，同属 S-class PHA。
#   · Canon C29   → recipe 覆盖为 `1:256`。
# --------------------------------------------------------------------------- #
HEAD_NOZZLE_COUNT = {
    "DMCSamba": 12,
    "KM512": 512,
    "KM1024i": 1024,
    "SpectraSClass": 128,
    "CanonC29": 256,
}

#: 已停产的喷头（不再作为新项目的开发/验证目标）
DISCONTINUED_HEADS = {"Spectra SX3": 128}

# Samba 波形脉冲参数：ADA 流程默认范围
#   label        name                                ada_default   unit     说明
SAMBA_WAVEFORM = [
    ("Pulse0 电压", "DropAnalysis.DMC.Samba.Pulse0.Voltage", "10:2.5:40", "V", "主脉冲幅度"),
    ("Pulse0 宽度", "DropAnalysis.DMC.Samba.Pulse0.Width", "1:0.1:4", "µs", "主脉冲宽度≈停留时间"),
    ("Pulse0 尾隙", "DropAnalysis.DMC.Samba.Pulse0.Tail", "2:0.2:8", "µs", "主脉冲后间隔，抑制卫星"),
    ("Pulse1 电压", "DropAnalysis.DMC.Samba.Pulse1.Voltage", "10:2.5:40", "V", "第二脉冲幅度"),
    ("Pulse1 宽度", "DropAnalysis.DMC.Samba.Pulse1.Width", "1:0.1:4", "µs", ""),
    ("Pulse1 尾隙", "DropAnalysis.DMC.Samba.Pulse1.Tail", "2:0.2:8", "µs", ""),
    ("Pulse2 电压", "DropAnalysis.DMC.Samba.Pulse2.Voltage", "10:2.5:40", "V", ""),
    ("Pulse2 宽度", "DropAnalysis.DMC.Samba.Pulse2.Width", "1:0.1:4", "µs", ""),
    ("Pulse2 尾隙", "DropAnalysis.DMC.Samba.Pulse2.Tail", "2:0.2:8", "µs", ""),
    ("Pulse3 电压", "DropAnalysis.DMC.Samba.Pulse3.Voltage", "10:2.5:40", "V", ""),
    ("Pulse3 宽度", "DropAnalysis.DMC.Samba.Pulse3.Width", "1:0.1:4", "µs", ""),
    ("Pulse3 尾隙", "DropAnalysis.DMC.Samba.Pulse3.Tail", "2:0.2:8", "µs", ""),
    ("边沿速率", "DropAnalysis.DMC.Samba.Slewrate", "25:5:55", "V/µs", "同时决定上升/下降沿斜率"),
    ("供电电压", "DropAnalysis.DMC.Samba.VoltageSupply", "110:2:120", "V", "喷头供电"),
]

PROCESS = [
    ("点火频率", "DropAnalysis.Frequency", "1000:1000:10000", "Hz"),
    ("墨路压力", "DropAnalysis.InkPressure", "-25:2.5:-10", "mbar"),
    ("喷头温度", "DropAnalysis.HeadTemperature", "30:5:55", "ºC"),
    ("LED 延时", "DropAnalysis.LEDDelay", "0:25:250", "µs"),
    # `1:12` = **要测量的喷嘴序列**，取 Samba 全头 12 孔
    #（物理孔数见 `HEAD_NOZZLE_COUNT`；此处是扫描范围，不是"机器有多少孔"）。
    # flow 默认是 `1:128`（通用默认，各头不同），实机 Samba recipe 覆盖为 `1:12`。
    ("喷嘴选择", "DropAnalysis.Nozzle", "1:12", ""),
]

# 刻度参考：SUSS 在 APO（打印规划，不进寻优）里对这些参数用的更窄默认区间。
# 只用来回答"网格该多密"，与寻优判据无关。
NARROW_REFERENCE = {
    "DropAnalysis.DMC.Samba.Pulse0.Voltage": ("20:1:32", "13 水平", "V"),
    "DropAnalysis.DMC.Samba.Pulse0.Width": ("1.4:0.2:4", "14 水平", "µs"),
    "DropAnalysis.DMC.Samba.Pulse0.Tail": ("1.4:0.2:4", "14 水平", "µs"),
    "DropAnalysis.DMC.Samba.Pulse1.Voltage": ("20:1:32", "13 水平", "V"),
    "DropAnalysis.DMC.Samba.Pulse1.Width": ("1.4:0.2:4", "14 水平", "µs"),
    "DropAnalysis.DMC.Samba.Pulse1.Tail": ("1.4:0.2:4", "14 水平", "µs"),
    "DropAnalysis.DMC.Samba.Slewrate": ("25:5:55", "7 水平", "V/µs"),
    "DropAnalysis.DMC.Samba.VoltageSupply": ("110:2:120", "6 水平", "V"),
}


# --------------------------------------------------------------------------- #
# 范围展开：严格复刻 ADA.cs GetSeparateParameterValues
# --------------------------------------------------------------------------- #
def expand_range(value: str) -> List[str]:
    """把 ADA/HMI 的范围语法展开为取值列表（与 ADA.cs 6016-6141 行为一致）。

    `;` 分段；段 `a:b:c` → 步长 b 的闭区间；段 `a:b` → step=1；单值原样。
    """
    out: List[str] = []
    text = str(value).strip()
    if not text:
        return out
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) >= 3:
            start, step, stop = float(parts[0]), float(parts[1]), float(parts[2])
            if step == 0:
                out.append(_fmt(start))
                continue
            v, guard = start, 0
            while ((step > 0 and v <= stop + 1e-9) or (step < 0 and v >= stop - 1e-9)):
                out.append(_fmt(v))
                v += step
                guard += 1
                if guard > 20000:
                    break
        elif len(parts) == 2:
            start, stop = float(parts[0]), float(parts[1])
            v = start
            while v <= stop + 1e-9:
                out.append(_fmt(v))
                v += 1
        else:
            out.append(_fmt(float(item)) if _isnum(item) else item)
    return out


def _isnum(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else ("%g" % v)


def _prod(xs) -> int:
    p = 1
    for x in xs:
        p *= x
    return p


# --------------------------------------------------------------------------- #
# 真实解析 95 ADA 的流程文件与 recipe（不是硬编码常量）
# --------------------------------------------------------------------------- #
def _attrs(tag: str) -> Dict[str, str]:
    """引号感知的属性解析（单/双引号都认）。"""
    return {n: (a if a != "" else b)
            for n, a, b in re.findall(r"""([A-Za-z][\w.]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""", tag)}


def parse_flow(path: str = ADA_FLOW) -> Dict[str, Dict[str, str]]:
    """解析 `Advanced-Drop-Analysis-<Head>.xml` 的 <Parameter> 定义。

    返回 {参数全名(去掉 .Checkbox): {"default","unit","description"}}。
    证据：ADA.cs 4882 `ConfigParameters()` 直接读这些参数名，
         6016 `GetSeparateParameterValues()` 读它们的值。
    """
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    out: Dict[str, Dict[str, str]] = {}
    for m in re.finditer(r"<Parameter\b([^>]*?)/?>", text, re.S):
        a = _attrs(m.group(1))
        name = a.get("Name", "")
        if not name.startswith("DropAnalysis.") or name.endswith(".Checkbox"):
            continue
        out[name] = {"default": a.get("Default", ""), "unit": a.get("Unit", ""),
                     "description": a.get("Description", "")}
    return out


def parse_recipe(path: str) -> Dict[str, str]:
    """解析 `*.recipe.xml` 的 <ParameterValues> → {参数全名: 覆盖值}。

    注意 recipe 里参数名可能带 `[0]` 下标（如 `DropAnalysis.Frequency[0]`），
    这里统一去掉下标以便与流程参数对齐。
    """
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    out: Dict[str, str] = {}
    for m in re.finditer(r"<Parameter\b([^>]*?)/?>", text, re.S):
        a = _attrs(m.group(1))
        name = a.get("Name", "")
        if "Value" not in a:
            continue
        out[re.sub(r"\[\d+\]$", "", name)] = a["Value"]
    return out


def _params_from_text(text: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for blk in re.findall(r"<parameter\b.*?</parameter>", text, re.S | re.I):
        fn = re.search(r"<FullName>([^<]*)</FullName>", blk)
        us = re.search(r"<UpdateSpeed>([^<]*)</UpdateSpeed>", blk)
        if fn and us:
            try:
                out[fn.group(1).strip()] = int(us.group(1).strip())
            except ValueError:
                pass
    return out


def parse_update_speeds(viewer_dir: str, head: str = "DMCSamba",
                        fallback_dirs: Optional[List[str]] = None
                        ) -> Dict[str, int]:
    """读 UpdateSpeed，严格按 ADA.cs:4683-4720 的加载顺序。

    ADA 先反序列化通用 `Parameters.xml`，再按**当前喷头**反序列化
    `Parameters_<HeadName>.xml` 并**追加**（喷头专用条目覆盖/补充通用条目）。
    所以必须只读"通用 + 本喷头"两个文件；若把所有 `Parameters_*.xml` 合并，
    无关喷头（如 XAAR1002 的 Frequency=4）会污染结果。
    """
    vdir = Path(viewer_dir)
    generic = vdir / "Parameters.xml"
    if not generic.is_file():
        raise FileNotFoundError("缺少通用参数文件：%s" % generic)
    merged = _params_from_text(generic.read_text(encoding="utf-8-sig", errors="replace"))

    specific = vdir / ("Parameters_%s.xml" % head)
    if not specific.is_file():
        # ADA 落盘的会话目录里带完整的一套喷头参数文件，优先用它
        for d in (fallback_dirs or []):
            cand = Path(d) / ("Parameters_%s.xml" % head)
            if cand.is_file():
                specific = cand
                break
    if specific.is_file():
        merged.update(_params_from_text(
            specific.read_text(encoding="utf-8-sig", errors="replace")))
    return merged


def scan_dimensions(ada_dir: str = ADA_DIR_C,
                    recipe: Optional[str] = None,
                    viewer_dir: Optional[str] = None,
                    head: str = "DMCSamba") -> List[ScanDim]:
    """从真实文件里列出**可扫描**维度（勾选后进入全析因的那些）。

    规则（来自 ADA.cs 4882-4922）：只有存在 `.Checkbox` 兄弟参数、
    且其类型为 Checkbox 的参数才是扫描维度；纯展示/内部参数排除。
    顺序按 `Parameters*.xml` 的 UpdateSpeed 降序（ADA.cs 4934-4937）。
    """
    # 注意：必须排除 *.Screen.xml（界面文件里没有 <Parameter> 定义），
    # 否则按字母序第一个匹配可能是 Screen 文件，导致解析出 0 个维度。
    flows = sorted(p for p in Path(ada_dir).glob("Advanced-Drop-Analysis-*.xml")
                   if not p.name.endswith(".Screen.xml"))
    if not flows:
        raise FileNotFoundError("找不到 ADA 流程文件：%s" % ada_dir)

    # 每个喷头各有自己的一套流程文件，绝不能混用：按 head 精确挑选
    want = head.lower().replace("-", "").replace(" ", "")
    picked = [p for p in flows
              if p.stem.replace("Advanced-Drop-Analysis-", "").lower()
              .replace("-", "").replace(" ", "") == want]
    if not picked:
        raise FileNotFoundError(
            "喷头 %r 没有对应的流程文件；可用：%s"
            % (head, ", ".join(p.stem.replace("Advanced-Drop-Analysis-", "") for p in flows)))
    flows = picked

    override = parse_recipe(recipe) if recipe else {}
    speeds = parse_update_speeds(
        viewer_dir or (NEW_LP50 + r"\Tools\DropAnalysis\Viewer"), head,
        fallback_dirs=[r"D:\Gitee\LP50\AutoADA\share\NOA89-test-01\Parameters"])

    dims: List[ScanDim] = []
    for fp in flows:
        text = fp.read_text(encoding="utf-8-sig", errors="replace")
        params: Dict[str, Dict[str, str]] = {}
        checkboxes = set()
        for m in re.finditer(r"<Parameter\b([^>]*?)/?>", text, re.S):
            a = _attrs(m.group(1))
            name = a.get("Name", "")
            if name.endswith(".Checkbox"):
                checkboxes.add(name[: -len(".Checkbox")])
            elif name.startswith("DropAnalysis."):
                params[name] = {"default": a.get("Default", ""),
                                "unit": a.get("Unit", ""),
                                "description": a.get("Description", "")}
        for name, d in params.items():
            if name not in checkboxes:
                continue                      # 没有 Checkbox → 不是扫描维度
            value = override.get(name, d["default"])   # 配方覆盖优先
            if not value:
                continue
            dims.append(ScanDim(
                label=d["description"] or name.split(".")[-1],
                name=name, range_text=value, unit=d["unit"],
                enabled=override.get(name + ".Checkbox") == "1",
                note="流程默认" if name not in override else "配方覆盖",
                update_speed=speeds.get(name, 3),
            ))
        break                                # 只取第一个（与单喷头运行一致）
    dims.sort(key=lambda d: -d.update_speed)   # ADA.cs:4934 降序
    return dims


def verify_against_flow(dims: List[ScanDim] = None, ada_dir: str = ADA_DIR_C,
                         head: str = "DMCSamba",
                         recipe: Optional[str] = None):
    """交叉校验：硬编码的默认范围是否与真实流程文件一致（基线 = V4730 4.7.3）。

    返回 `(errors, notes)`：
      · **errors** —— 真正的错误（代码常量与实机事实冲突），必须修；
      · **notes** —— 提示（配方合法地收窄了扫描范围等），只需人工确认。
    目的是让常量一旦与实机不符就**立刻暴露**，而不是静默失真。

    ⚠️ **三个基准层次，语义各不相同**（2026-11-03 踩坑后明确）：
      · **flow 默认值**——`Advanced-Drop-Analysis-<Head>.xml` 里的 `<Default>`，
        是**该参数在流程里的出厂默认**（不同喷头不同）；
      · **recipe 覆盖值**——`*.recipe.xml` 里真正生效的值，会**覆盖** flow 默认；
      · **实机物理量**——例如喷嘴**孔数**，只能是 12（Samba），不能是 128。

    而且覆盖值的**语义要分两类**，不能一刀切地"以覆盖值为准"：

      | 参数类别 | 例 | 覆盖值语义 | 判定口径 |
      |---|---|---|---|
      | **扫描范围** | `Nozzle` | `1:12` = **要测哪些孔** | 上界不得**超过物理孔数**，但可以是子集 |
      | **扫描区间** | `HeadTemperature` | `25:2.5:50` = 范围被收窄 | 覆盖值是**另一种合法配置**，不必然是"错" |

    踩坑经过：`1:128` 曾被当成 Samba 的实际孔数写进方案，
    而 `HeadTemperature` 的 `25:2.5:50` 只是配方把扫描区间收窄——两者都"与 flow 默认不同"，
    但只有前者是**事实错误**。详见 §8.6.8 勘误。

    因此本函数分两种判定：
      1. **喷嘴范围**（`Nozzle`）：解析出的**最大孔号不得大于该喷头的物理孔数**
         （`HEAD_NOZZLE_COUNT`）。子集是合法的（ADA 源码把它当 "Active Nozzle"
         序列解析），超出物理孔数才是错；
      2. **其余参数**：与 flow 默认比；覆盖值不同则**只提示、不算错**
         （配方可以合法地收窄扫描范围）——但会列出供人工确认。
    """
    fp = Path(ada_dir) / ("Advanced-Drop-Analysis-%s.xml" % head)
    try:
        flow = parse_flow(str(fp))
    except OSError:
        return ["无法读取流程文件 %s" % fp], []
    # recipe 覆盖值（下标已去掉，与 flow 的键对齐）
    over: Dict[str, str] = {}
    if recipe:
        try:
            over = parse_recipe(recipe)
        except OSError:
            over = {}
    n_phys = HEAD_NOZZLE_COUNT.get(head)
    problems: List[str] = []
    notes: List[str] = []
    for label, name, ada_def, unit in [(a, b, c, d) for (a, b, c, d, _e) in SAMBA_WAVEFORM] + PROCESS:
        key = name if name in flow else None
        if key is None:
            problems.append("流程文件中找不到参数：%s（%s）" % (name, label))
            continue
        flow_def = flow[key]["default"]
        if name in over:
            real = over[name]
            src = "配方覆盖值"
        else:
            real = flow_def
            src = "流程默认值"
        if name == "DropAnalysis.Nozzle":
            # 喷嘴扫描范围：上界不得超物理孔数；子集合法
            try:
                mx = max(int(x) for x in expand_range(ada_def))
            except (ValueError, TypeError):
                mx = None
            if mx is not None and n_phys and mx > n_phys:
                problems.append("%s：代码写 %r，最大孔号 %d 超过 %s 的物理孔数 %d"
                                % (label, ada_def, mx, head, n_phys))
            else:
                # 顺带记录：配置值 vs 物理孔数的差距（子集是正常的）
                try:
                    rx = max(int(x) for x in expand_range(real))
                except (ValueError, TypeError):
                    rx = None
                if rx is not None and n_phys and rx < n_phys:
                    notes.append("%s：%s 的%s %r 只覆盖到第 %d 孔，"
                                 "而物理孔数为 %d（其余孔本次不测，属正常）"
                                 % (label, head, src, real, rx, n_phys))
        elif real != ada_def:
            # 扫描区间类：配方可合法收窄，只提示
            notes.append("%s：代码写 %r（flow 默认），%s 的配方覆盖为 %r —— "
                         "配方收窄了扫描范围，属正常配置，请确认代码取值意图"
                         % (label, ada_def, head, real))
    return problems, notes


# --------------------------------------------------------------------------- #
# 扫描维度
# --------------------------------------------------------------------------- #
@dataclass
class ScanDim:
    label: str
    name: str
    range_text: str
    unit: str = ""
    enabled: bool = False
    note: str = ""
    update_speed: int = 3      # Tools\DropAnalysis\Viewer\Parameters*.xml 的 <UpdateSpeed>

    @property
    def levels(self) -> List[str]:
        return expand_range(self.range_text)

    @property
    def n(self) -> int:
        return len(self.levels)

    def row(self) -> str:
        return "| %s | %-46s | %-18s | %-5s | %d | %s |" % (
            "✓" if self.enabled else "□", self.name, self.range_text,
            self.unit or "—", self.n, self.note or "")


def _dim(label, name, rng, unit, enabled=False, note=""):
    return ScanDim(label=label, name=name, range_text=rng, unit=unit,
                   enabled=enabled, note=note)


@dataclass
class DOEPlan:
    title: str
    dims: List[ScanDim] = field(default_factory=list)
    #: 默认 `1:12` = **DMC Samba 实机喷嘴数**（规格书 12 jets；实机 recipe 覆盖值）。
    #: 不用 LP50 flow 的 `1:128` 通用默认——那是 KM512/Spectra 的头数，
    #: 照填到 Samba 上会扫到不存在的孔（2026-11-03 核实）。
    nozzle_range: str = "1:12"
    nozzle_enabled: bool = True
    intro: str = ""
    accept_hint: str = ""

    def table(self) -> str:
        lines = ["## " + self.title, ""]
        if self.intro:
            lines += [self.intro, ""]
        lines += ["| 勾选 | 参数全名 | 要填的范围/序列 | 单位 | 取值数 | 说明 |",
                  "|---|---|---|---|---|---|"]
        lines += [d.row() for d in self.dims]
        lines += ["| %s | %-46s | %-18s | %-5s | %d | %s |" % (
            "✓" if self.nozzle_enabled else "□", "DropAnalysis.Nozzle",
            self.nozzle_range, "—", len(expand_range(self.nozzle_range)),
            "喷嘴维度（只影响测多少孔）")]
        lines.append("")
        n_wave = _prod(d.n for d in self.dims if d.enabled)
        n_noz = len(expand_range(self.nozzle_range)) if self.nozzle_enabled else 1
        lines += [
            "**波形参数组合数** = %d 张/孔 × **%d 孔** = **%d 张 noz*.png**"
            % (n_wave, n_noz, n_wave * n_noz), "",
            "（未勾选的参数不复述：它们维持 recipe 原值，不参与扫描）", "",
            "输出目录 = `BaseFolder\\Directory\\<各参数前缀+取值>\\...\\noz*.png`，",
            "前缀见 `Tools\\DropAnalysis\\Viewer\\Parameters.xml` 的 `FolderPrefix`"
            "（Nozzle→`noz`，Pulse0 电压→`P0V`，宽度→`P0W`，尾隙→`P0T`），",
            "所以每个波形组合落在自己独立的子目录里，可与 DropQCS 判据逐组合对上。",
        ]
        if self.accept_hint:
            lines += ["", "**合格判据**：" + self.accept_hint]
        return "\n".join(lines)

    def recipe_xml(self, base_folder: str = r"C:\LP50\ScreenCaptures",
                   recipe: Optional[str] = None) -> str:
        """输出可直接照填进 ADA recipe 的 <ParameterValues> 片段。

        ⚠️ **参数名必须与实机 recipe 完全一致**（2026-11-03 核实）：
        实机 `ADA Fujifilm DMCSamba.recipe.xml` 里通用参数带 `[0]` 下标、
        喷头专属参数不带；而 KM512 / KM1024i 的 recipe 恰好相反。
        本方法用 `_recipe_param_names()` 从真实 recipe 反查实际名字，
        避免照填后 ADA 认不出参数名而**静默不生效**（扫一个都没跑却不报错）。
        """
        actual = _recipe_param_names(recipe or DEFAULT_RECIPE)
        lines = ['<!-- ============ %s ============ -->' % self.title,
                 '<Recipe xmlns:xsd="http://www.w3.org/2001/XMLSchema" '
                 'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">',
                 '  <Name>%s</Name>' % self.title,
                 '  <FlowFile>Advanced-Drop-Analysis-DMCSamba.xml</FlowFile>',
                 '  <ParameterValues>',
                 '    <Parameter Name="%s" Value="%s" />'
                 % (_recipe_name_for("DropAnalysis.BaseFolder", actual), base_folder),
                 '    <Parameter Name="%s" Value="%d" />'
                 % (_recipe_name_for("DropAnalysis.Nozzle.Checkbox", actual),
                    1 if self.nozzle_enabled else 0),
                 '    <Parameter Name="%s" Value="%s" />'
                 % (_recipe_name_for("DropAnalysis.Nozzle", actual), self.nozzle_range)]
        for d in self.dims:
            if d.enabled:
                lines += ['    <Parameter Name="%s" Value="%s" /> <!-- %s -->'
                          % (_recipe_name_for(d.name, actual), d.range_text, d.label),
                          '    <Parameter Name="%s" Value="1" />'
                          % _recipe_name_for(d.name + ".Checkbox", actual)]
            else:
                lines += ['    <!-- 未勾选（维持原值，不扫描）：%s = %s -->' % (d.label, d.range_text)]
        lines += ['  </ParameterValues>', '</Recipe>']
        return "\n".join(lines)


def _recipe_param_names(recipe_path: Optional[str] = None) -> Dict[str, str]:
    """从实机 recipe 里读出「参数全名 → recipe 中实际用的名字」。

    为什么需要（2026-11-03 核实）
    --------------------------
    同一个参数在不同 recipe 里的**下标风格并不统一**：

    | recipe | 通用参数（`DropAnalysis.X`） | 喷头专属（`DropAnalysis.DMC.Samba.*`） |
    |---|---|---|
    | ADA Fujifilm DMCSamba | 24/24 **带 `[0]`** | 0/28 带 |
    | ADA CanonC29 | 28/28 **带 `[0]`** | — |
    | ADA Spectra S-class | 26/26 **带 `[0]`** | — |
    | ADA KM512 | 0/26 带 | — |
    | ADA KM1024i | 0/26 带 | — |

    而 `parse_recipe()` 在读入时会把 `[0]` 这类下标去掉（见其 docstring），
    所以**读**不受影响；但**写**（`recipe_xml()`）若照抄另一种风格，
    照填进 ADA 后可能因参数名不匹配而**静默不生效**——扫描一个都没跑，
    却在界面上看不出任何报错。

    故本函数从真实 recipe 反查每个参数的实际名字，查不到就退回"按同类参数
    的多数风格推断"，并在推断时明确标注。
    """
    if recipe_path is None:
        return {}
    try:
        text = Path(recipe_path).read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return {}
    out: Dict[str, str] = {}
    for m in re.finditer(r"<Parameter\b([^>]*?)/?>", text, re.S):
        a = _attrs(m.group(1))
        name = a.get("Name", "")
        if not name.startswith("DropAnalysis."):
            continue
        bare = re.sub(r"\[\d+\]$", "", name)
        out.setdefault(bare, name)
    return out


#: 实机 recipe 的默认位置（DMCSamba）；只用于反查参数命名风格
DEFAULT_RECIPE = ADA_DIR_C + r"\ADA Fujifilm DMCSamba.recipe.xml"


def _recipe_name_for(bare: str, actual: Dict[str, str]) -> str:
    """把参数全名转成该 recipe 实际使用的名字。

    优先用实机 recipe 里查到的大小写与下标；查不到时按同类多数风格推断：
    喷头专属参数（中间含 `.` 分段，如 `DropAnalysis.DMC.Samba.X`）不加下标，
    通用参数（`DropAnalysis.X`）加 `[0]`。
    """
    if bare in actual:
        return actual[bare]
    parts = bare.split(".")
    # DropAnalysis.DMC.Samba.Pulse0.Voltage → 喷头专属（有一段是喷头命名空间）
    if len(parts) >= 4 and parts[2] not in ("Checkbox",):
        return bare
    return bare + "[0]"


# --------------------------------------------------------------------------- #
# 四阶段 DOE（screen → narrow → window → uniform）
# --------------------------------------------------------------------------- #
def plan_screen() -> DOEPlan:
    """阶段 1 —— 粗筛"可行波形域"。

    设计依据（全部来自本项目实测，2026-11-03 复核）
    ------------------------------------------------
    1. **电压 × 宽度必须正交**：现有 108 孔数据里 34V/38V 只在宽度=2.2 下测过，
       两者效应无法分离（PROJECT_RECORD §8.6.4.1 问题 4）。本阶段用**完全交叉**
       的稀疏水平（每维 4–5 个点，任意组合都出现），让两维效应可分离。
    2. **电压上限压到 40V**：现有数据里 40V 时速度 3.82–4.65 m/s 且出现阈值跳变
       （34→38V 反降、40V 跳升），说明**阈值效应就在 40V 附近**。
       不扫到 40V 就看不到阈值，也就采不到卫星滴样本。
    3. **尾隙往大扫**：卫星滴在高电压 / **宽尾隙**区出现
       （PROJECT_RECORD §8.6.4 勘误），尾隙下限不设到 2µs 附近就永远碰不到。
    4. **压力与温度固定在实测工作点**：本阶段要回答的是「波形域」问题，
       引入工艺变量会与波形效应混淆。实测工作点为
       **P ≈ -1 mbar / T ≈ 40–45 °C**（9 个会话实测，见 §8.6.4 勘误），
       两者都是"选窗旋钮"，此处固定以隔离波形效应。

    ⚠️ 压力取值说明：实测 -1 mbar **落在 ADA 默认扫描区间 -25…-10 之外**
    （机器 `IO.Ink.PressureSetpoint` 合法范围 -38…0，默认 -15）。
    本阶段按实测值填 -1，超出默认区间是**故意的**——那才是真实工作点。
    """
    dims = [
        _dim("Pulse0 电压", "DropAnalysis.DMC.Samba.Pulse0.Voltage",
             "20;26;32;40", "V", True,
             "4 水平；上限 40V=实测阈值区（现有数据 40V 时速度跳升）"),
        _dim("Pulse0 宽度", "DropAnalysis.DMC.Samba.Pulse0.Width",
             "1.4;2.0;2.6;3.4", "µs", True,
             "4 水平；与电压**完全交叉**，破解现有数据 34/38V 只在 W=2.2 的共线"),
        _dim("Pulse0 尾隙", "DropAnalysis.DMC.Samba.Pulse0.Tail",
             "1.4;2.6;4.0;6.0", "µs", True,
             "4 水平；下限 1.4µs + 上限 6µs，专为采到卫星滴区"),
        _dim("边沿速率", "DropAnalysis.DMC.Samba.Slewrate",
             "30;40;50", "V/µs", True, "3 水平；决定上升/下降沿斜率"),
        _dim("Pulse1-3 脉冲", "DropAnalysis.DMC.Samba.Pulse1.Voltage",
             "10:2.5:40", "V", False, "阶段1固定不扫：先确认单脉冲是否够"),
        _dim("点火频率", "DropAnalysis.Frequency", "1000", "Hz", False, "固定"),
        _dim("墨路压力", "DropAnalysis.InkPressure", "-1", "mbar", False,
             "固定在**实测工作点**（9 会话实测 -0.83…-1.15 mbar）；"
             "注意该值超出 ADA 默认区间 -25…-10，属故意"),
        _dim("喷头温度", "DropAnalysis.HeadTemperature", "42", "ºC", False,
             "固定在**实测工作点**（9 会话实测 40.07…45.17 °C）"),
        _dim("LED 延时", "DropAnalysis.LEDDelay", "150", "µs", False, "固定"),
    ]
    return DOEPlan(
        title="阶段1 · 粗筛可行波形域（电压×宽度正交 · 4 维稀疏网格）",
        dims=dims, nozzle_range="1;2;3;4", nozzle_enabled=True,
        intro="目的：用最少的实验量找出「能形成单滴、无卫星」的电压×宽度×尾隙×边沿区域。"
              "此阶段**不追求最优点**，只追求看清可行域边界，并让电压与宽度可分离归因。",
        accept_hint="`[NrDroplets] == 1`（无卫星）、`[SpeedAverage] > 0`、"
                    "`|[AngleAverage]| < 2°`（飞行方向正）。")


def plan_narrow() -> DOEPlan:
    """阶段 2 —— 围绕阶段1最优点局部加密。"""
    dims = [
        _dim("Pulse0 电压", "DropAnalysis.DMC.Samba.Pulse0.Voltage",
             "26;28;30;32", "V", True, "阶段1最优点 ±3 V 内 4 个水平"),
        _dim("Pulse0 宽度", "DropAnalysis.DMC.Samba.Pulse0.Width",
             "2.0;2.4;2.8;3.2", "µs", True, "阶段1最优点 ±0.8 µs 内 4 个水平"),
        _dim("Pulse0 尾隙", "DropAnalysis.DMC.Samba.Pulse0.Tail",
             "1.4;2.5;3.6;4.8", "µs", True, "4 个水平（该维度响应弱，保持稀疏）"),
        _dim("Pulse1 电压", "DropAnalysis.DMC.Samba.Pulse1.Voltage",
             "0;20;30", "V", True, "0=关断/20/30，判定多脉冲是否必要"),
        _dim("边沿速率", "DropAnalysis.DMC.Samba.Slewrate",
             "35;40;45", "V/µs", True, "3 个水平"),
        _dim("点火频率", "DropAnalysis.Frequency", "1000", "Hz", False, "固定"),
    ]
    return DOEPlan(
        title="阶段2 · 围绕最优点局部加密（含第二脉冲必要性判定）",
        dims=dims, nozzle_range="1;2;3;4;5;6", nozzle_enabled=True,
        intro="目的：在阶段1给出的可行域内**收窄**（水平数少于阶段1，范围更聚焦）；"
              "同时判定单脉冲是否足够。",
        accept_hint="同阶段1；并加 `[PrintheadVolumeTotalStdDev]` 小（孔间体积一致）。")


def plan_window() -> DOEPlan:
    """阶段 2b —— **把温度 / 压力扫出成滴窗口**（PROJECT_RECORD §9 的核心待办）。

    为什么单独设一个阶段
    ------------------
    现有 108 孔数据是**「成滴窗口切片」**：9 个会话全部落在"能稳定成滴"的窗口内，
    `InkPressure` 恒 -1 mbar、`HeadTemperature` 在 40–45 °C 窄幅浮动。
    后果（§8.6.4 勘误已确认）：
      · 温度、压力看起来像"常量列"，但它们**不是噪声**——它们是**选窗旋钮**，
        作用是"把系统按在能成滴的区域里"，所以数据里看不到它们的效应；
      · 温度→粘度→喷射、压力→弯月面→喷射这两条弱互关，被"全都在喷"的单一状态淹没了。

    所以必须**主动把这两个维度扫出窗口**，让边界暴露出来：
      · **低压 / 低温** → 进不喷 / 断线边界；
      · **高压 / 高温** → 进卫星滴 / 拖尾边界。

    ⚠️ 压力取值必须**覆盖实测工作点 -1 mbar**（9 会话实测 -0.83…-1.15），
    否则新数据与现有 108 孔不在同一工作点，无法合并。
    机器 `IO.Ink.PressureSetpoint` 合法范围 **-38…0 mbar**（默认 -15），
    故取 `-1;-6;-12;-20` 覆盖从工作点到强负压的两侧。

    温度同理：实测工作点 40–45 °C，取 `36;42;47;52` 向两侧展开。
    机器侧温度上限见 `IO.HeadHeating`，此处取 52 °C 留出余量。
    """
    dims = [
        _dim("墨路压力", "DropAnalysis.InkPressure",
             "-1;-6;-12;-20", "mbar", True,
             "**独立扫描维度**：含实测工作点 -1，向强负压推进找不喷/断线边界"),
        _dim("喷头温度", "DropAnalysis.HeadTemperature",
             "36;42;47;52", "ºC", True,
             "**独立扫描维度**：含实测工作点 42，向上找卫星/拖尾边界、向下找断线边界"),
        _dim("Pulse0 电压", "DropAnalysis.DMC.Samba.Pulse0.Voltage",
             "28;30;32;34", "V", True, "阶段1/2 选定的波形附近 4 水平（随工艺一起变）"),
        _dim("Pulse0 宽度", "DropAnalysis.DMC.Samba.Pulse0.Width",
             "2.2;2.5;2.8", "µs", True, "3 水平（与电压完全交叉）"),
        _dim("Pulse0 尾隙", "DropAnalysis.DMC.Samba.Pulse0.Tail",
             "1.4;2.5;3.6", "µs", True, "3 水平（下限 1.4µs 专找卫星滴）"),
        _dim("边沿速率", "DropAnalysis.DMC.Samba.Slewrate",
             "35;40;45", "V/µs", True, "3 水平"),
        _dim("点火频率", "DropAnalysis.Frequency", "1000", "Hz", False, "固定"),
        _dim("LED 延时", "DropAnalysis.LEDDelay", "150", "µs", False, "固定"),
    ]
    return DOEPlan(
        title="阶段2b · 工艺窗口扫描（压力 / 温度扫出成滴窗口）",
        dims=dims, nozzle_range="1;2;3;4", nozzle_enabled=True,
        intro="目的：把 InkPressure 与 HeadTemperature 从「常量列」变成**独立扫描维度**，"
              "并主动扫出成滴窗口的边界——低压低温进不喷/断线区，高压高温进卫星滴/拖尾区。"
              "此阶段的产出是**边界**，不是最优点。",
        accept_hint="同阶段1；并记录 `[NrDroplets] >= 2`（卫星滴）与 `[NrDroplets] == 0`"
                    "（不喷）各自的边界位置。")


def plan_delay() -> DOEPlan:
    """阶段 2c —— **LED 延时扫描：把「喷口成滴形貌」拍成时间序列**（2026-11-03 新增）。

    为什么需要这个阶段
    ------------------
    论文（Kim2022）用**高速 CCD 序列**拍同一滴的成滴过程：
    40→400 µs、5 µs 步长、73 帧/波形，据此判形态（液柱拉长→断裂→卫星滴）。
    LP50/ADA 的原生做法不同：`m_dDelay2 = m_dDelay1 + Dropview.LedDelayIncr`
    （`LedDelayIncr` 默认 30 µs）做**双闪两点差分**，一次给一个速度值。

    但 **`DropAnalysis.LEDDelay` 本身就是 ADA 的正式可扫描参数**
    （`Tools/DropAnalysis/Viewer/Parameters.xml`：`FullName=DropAnalysis.LEDDelay`
    → `LP50=Vision.LedDelay`，前置动作 `SetLEDDelay` 写 `m_dDelay1`），
    且 `Vision.LedDelay` 在 ADA 的无下标白名单 `parsWithoutIndex` 内
    （`Advanced-Drop-Analysis.cs:749`），量程 **0–1000 µs**、小步长 0.1 µs。
    ⇒ **扫这个参数就能拿到延时序列**，覆盖论文 0–220 µs 的成滴形貌窗口。

    实测佐证（PROJECT_RECORD §5.1 B 类）
    ------------------------------------
    · 实机图 `ScreenCaptures\\2025-08-04_09.40.52_Measurement\\DropQCS1,2,3,4,5.png`
      （`LEDDelay=80 µs`）里，5 条**从画面顶端延伸下来的液柱**清晰可见，末端挂主滴；
    · 同图 XML 里 ADA 把液柱单列为一条 `Volume=0.66 pl`、包围盒 114 px 高、
      **`Speed` 为空** 的记录（主滴另计 5.07 pl）→ "细长、无速度"= 成滴形貌本体；
    · 现有 108 孔 Samba 数据全部用 `LEDDelay=180/220 µs`，属**液滴已脱离**的时段，
      所以那批图里看不到形貌——**是采样窗口问题，不是能力问题**。

    本阶段的设计取舍
    ----------------
    · **延时范围 0–220 µs**：与论文的 40–220 µs 观察窗对齐，且 ≤ 实机验证过的 220 µs；
      0 附近对应"液滴刚脱离喷口"，正是卫星滴刚分离的时刻。
    · **波形全部固定**在实测工作点（40 V / 1.9 µs / 尾隙 4.5 µs / Slewrate 40 /
      供电 120 V，对应 `NOA89-test-03`、`-05` 的实际设定），因为本阶段要回答的是
      **"形貌随延时怎么演化"**，不是"哪个波形最好"——引入波形变量会混淆两者。
    · ⚠️ **每个延时点样本量小**：延时点是不同次发射的液滴（不是同一滴的轨迹），
      且延时越靠近 0，能被 ADA 判定为"液滴"的像素越少。
      ⇒ 判据必须**按延时点分别看**，不能直接套 DropQCS 的单滴合格率。
    """
    dims = [
        _dim("LED 延时", "DropAnalysis.LEDDelay",
             "0;10;20;40;60;80;110;140;170;200;220", "µs", True,
             "**独立扫描维度**：覆盖 0–220 µs 成滴形貌窗口（论文用 40–220 µs / 5 µs 步长）；"
             "11 水平，密步在 0–80 µs（液柱拉长→断裂区），疏步在 110–220 µs"),
        _dim("Pulse0 电压", "DropAnalysis.DMC.Samba.Pulse0.Voltage",
             "40", "V", False, "固定在实测工作点（NOA89-test-03/05 实测 Pulse0Voltage=40）"),
        _dim("Pulse0 宽度", "DropAnalysis.DMC.Samba.Pulse0.Width",
             "1.9", "µs", False, "固定在实测工作点（NOA89-test-03/05 实测 Pulse0Width=1.9）"),
        _dim("Pulse0 尾隙", "DropAnalysis.DMC.Samba.Pulse0.Tail",
             "4.5", "µs", False, "固定在实测工作点（NOA89-test-03/05/09 实测 Pulse0Tail=4.5）"),
        _dim("边沿速率", "DropAnalysis.DMC.Samba.Slewrate",
             "40", "V/µs", False, "固定在实测工作点（Slewrate=40）"),
        _dim("供电电压", "DropAnalysis.DMC.Samba.VoltageSupply",
             "120", "V", False, "固定在实测工作点（VoltageSupply=120）"),
        _dim("墨路压力", "DropAnalysis.InkPressure", "-1", "mbar", False, "实测工作点"),
        _dim("喷头温度", "DropAnalysis.HeadTemperature", "42", "ºC", False, "实测工作点"),
        _dim("点火频率", "DropAnalysis.Frequency", "1000", "Hz", False, "固定"),
    ]
    return DOEPlan(
        title="阶段2c · 成滴形貌延时扫描（LEDDelay 0–220 µs 时间序列）",
        dims=dims, nozzle_range="1;2;3;4", nozzle_enabled=True,
        intro="目的：把**喷口成滴形貌**（液柱拉长→断裂→卫星滴分离）拍成**延时序列**，"
              "补上论文用高速 CCD 做的、现有 108 孔数据完全没有的观测维度。"
              "波形与工艺**全部固定**在实测工作点——本阶段只回答"
              "「形貌随延时怎么演化」，不回答「哪个波形最好」。",
        accept_hint="**按延时点分别看**（不能套单滴合格率）：① `Nr. of droplets >= 2` "
                    "的**首次出现延时**= 卫星滴分离时刻；② 出现时刻随电压/尾隙的移动 = "
                    "尾隙抑制卫星滴的量化证据；③ 包围盒高度随延时的增长斜率 = 液柱拉长速率。")


def plan_uniform() -> DOEPlan:
    """阶段 3 —— 波形定型后的**全头重复性**扫描。

    ⚠️ 2026-11-03 修正：本阶段的**前提原先是错的**，已重写。

    原设计假定 Samba 有 128 个喷嘴（照抄了 LP50 flow 的通用默认 `1:128`），
    认为「现有 108 孔只覆盖 12 孔、仅占 9%」，所以要「扫全量 128 孔」。
    实机核实结论（三个独立来源一致）：
      · 规格书 *DMC Samba Specification*：**Addressable Nozzles – 12 jets**；
      · 实机 recipe `ADA Fujifilm DMCSamba.recipe.xml`：`DropAnalysis.Nozzle[0] = 1:12`
        （旧机台 C:\\LP50 同值；`1:128` 是 KM512 / Spectra S-class 的通用默认）；
      · 实测数据：9 个会话 × 12 孔 = 108 行，**每个会话都完整覆盖 1..12 孔**。
    ⇒ Samba 全头就是 12 孔，**覆盖率早已是 100%**，不存在「待扫全头」这回事。

    本阶段改为回答真正没答过的问题：**同一波形、同一喷嘴，重复测的差异有多大？**
    只有拿到重复性噪声底，才能判断：
      · 孔间差异是**系统性空间梯度**（可建模、可补偿）
        —— 实测已见强梯度：同一波形内 孔号 vs 速度 Spearman ρ 均值 **+0.764**（p<0.001）；
      · 还是随机抖动（不可补偿，只能靠统计容差）。
    现在已有 40V/1.9 波形重复 4 次、40V/2.2 重复 2 次的数据可作初步估计，
    但样本太少——本阶段补足**重复次数**与**重复间隔**。
    """
    dims = [
        _dim("Pulse0 电压", "DropAnalysis.DMC.Samba.Pulse0.Voltage", "29", "V", False, "阶段2选定值"),
        _dim("Pulse0 宽度", "DropAnalysis.DMC.Samba.Pulse0.Width", "2.4", "µs", False, "阶段2选定值"),
        _dim("Pulse0 尾隙", "DropAnalysis.DMC.Samba.Pulse0.Tail", "3.0", "µs", False, "阶段2选定值"),
        _dim("边沿速率", "DropAnalysis.DMC.Samba.Slewrate", "40", "V/µs", False, "阶段2选定值"),
        _dim("墨路压力", "DropAnalysis.InkPressure", "-1", "mbar", False, "实测工作点"),
        _dim("喷头温度", "DropAnalysis.HeadTemperature", "42", "ºC", False, "实测工作点"),
    ]
    return DOEPlan(
        title="阶段3 · 波形定型后的全头重复性（12 孔 × N 次重复）",
        dims=dims, nozzle_range="1:12", nozzle_enabled=True,
        intro="目的：波形已定，**全头 12 孔**逐孔重复测，用于分离"
              "「孔间系统性差异」与「同一孔的重复性噪声」。"
              "现有 108 孔数据已覆盖全头 12 孔（覆盖率 100%），"
              "缺的不是孔数而是**重复次数**——同一波形最多只重复过 4 次。"
              "本阶段**不扫波形**，只重复测同一波形。",
        accept_hint="DropQCS 默认：`[NrDroplets] != 1` → Reject；"
                    "`[SpeedAverage]` 偏离 `[PrintheadSpeedAverage]` 超过 20% → Reject。"
                    "整头一致性另看 `[PrintheadSpeedStdDev]` / `[PrintheadVolumeTotalStdDev]`。"
                    "**重复性**：同一孔多次测量的速度标准差 → 与孔间标准差比较，"
                    "决定是否值得做喷嘴级补偿。")


# --------------------------------------------------------------------------- #
# DropQCS 判据
# --------------------------------------------------------------------------- #
def dropqcs_rules(v_target: float = 3.5, speed_margin: float = 0.20,
                  angle_max: float = 2.0) -> str:
    """生成"找最优喷射波形"用的 DropQCS 规则集。

    变量名与 `Rules\\Examples\\*.RulesConfig.xml` 完全一致（大小写敏感）。
    顺序即优先级：先保证"有滴"，再保证"单滴无卫星"，再看速度、方向。

    ⚠️ **目标速度必须按实测标定**：默认 5.0 m/s 会让合格率普遍为 0
    （实测速度分布 1.13–6.99 m/s，share 会话均值 2.8–4.7）。
    默认取 **3.5**——实测下 NOA89-test-05/06/07/08 四个会话合格率均为 100%，
    test-01 为 91.7%（--target-speed 4.5 时）。
    **每次换墨后必须用 ADA 实测重新确定。**
    """
    lo, hi = v_target * (1 - speed_margin), v_target * (1 + speed_margin)
    return f"""<?xml version="1.0" encoding="utf-8"?>
<!-- Copyright © SUSS MicroTec Netherlands B.V., All rights reserved -->
<RulesSet xmlns:xsd="http://www.w3.org/2001/XMLSchema"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <Rule Name="OptimalWaveformScreen">
    <!-- ================= 单孔判据（NozzlesRule，逐条求值） ================= -->
    <NozzlesRule>
      <!-- 1) 无滴 → 淘汰 -->
      <Condition Nozzles="all" Expression="[NrDroplets] &lt; 1">
        <Reject/>
      </Condition>
    </NozzlesRule>
    <NozzlesRule>
      <!-- 2) 多滴 = 卫星/卫星雨 → 淘汰（NrDroplets==2 表示主滴+1 个卫星） -->
      <Condition Nozzles="all" Expression="[NrDroplets] &gt; 1">
        <Reject/>
      </Condition>
    </NozzlesRule>
    <NozzlesRule>
      <!-- 3) 速度高于目标上限 -->
      <Condition Nozzles="all" Expression="[SpeedAverage] &gt; {hi}">
        <Reject/>
      </Condition>
    </NozzlesRule>
    <NozzlesRule>
      <!-- 4) 速度低于目标下限 -->
      <Condition Nozzles="all" Expression="[SpeedAverage] &lt; {lo}">
        <Reject/>
      </Condition>
    </NozzlesRule>
    <NozzlesRule>
      <!-- 5) 飞行方向偏离 -->
      <Condition Nozzles="all" Expression="[AngleAverage] &lt; -{angle_max} or [AngleAverage] &gt; {angle_max}">
        <Reject/>
      </Condition>
    </NozzlesRule>
    <NozzlesRule>
      <!-- 6) 以上全部通过 → 接受 -->
      <Condition Nozzles="all" Expression="[Result]==false">
        <Accept/>
      </Condition>
    </NozzlesRule>

    <!-- ================= 整头判据（HeadsRule） ================= -->
    <HeadsRule>
      <Condition Expression="[PercentageAnalyzed] == 100 and [NrRejectedNozzles] &gt; ([NrNozzles] / 2)">
        <Reject/>
        <Warning Message="More than 50% of the nozzles have been rejected by the rules engine"/>
        <Stop/>
      </Condition>
    </HeadsRule>
    <HeadsRule>
      <Condition Expression="[PercentageAnalyzed] == 100">
        <Prompt Message="[NrRejectedNozzles] out of [NrNozzles] nozzles rejected.[LineBreak]The results can be analyzed in the Advanced Drop Analysis (Viewer).">
          <Stop/>
        </Prompt>
        <Stop/>
      </Condition>
    </HeadsRule>
  </Rule>
</RulesSet>
"""


# --------------------------------------------------------------------------- #
def _dims_table(recipe: Optional[str] = None, head: str = "DMCSamba") -> str:
    """真实解析 95 ADA 流程（+ 可选 recipe）得到的扫描维度表。"""
    try:
        dims = scan_dimensions(recipe=recipe, head=head)
    except FileNotFoundError as exc:
        return "## 无法解析 ADA 流程\n\n%s\n" % exc

    lines = ["## %s 可扫描维度（真实解析自 95 ADA 流程，基线 V4730 / HMI 4.7.3%s）"
             % (head, " + recipe 覆盖" if recipe else ""), "",
             "筛选规则：存在 `.Checkbox` 兄弟参数 → 勾选后进入全析因扫描"
             "（ADA.cs:4882-4922 `IsSelected`）；按 UpdateSpeed 降序（ADA.cs:4934-4937）。", "",
             "| # | UpdateSpeed | 勾选 | 参数全名 | 当前值 | 单位 | 水平数 | 来源 | 说明 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, d in enumerate(dims, 1):
        lines.append("| %d | %d | %s | `%s` | `%s` | %s | %d | %s | %s |"
                     % (i, d.update_speed, "✓" if d.enabled else "□",
                        d.name, d.range_text, d.unit or "—", d.n,
                        d.note, d.label))
    n_on = sum(1 for d in dims if d.enabled)
    lines += ["", "共 **%d** 个可扫描维度，其中 **%d** 个已在 recipe 中勾选；"
              "勾选组合数 = %s。"
              % (len(dims), n_on,
                 f"{_prod(d.n for d in dims if d.enabled):,}" if n_on else "1（无勾选）")]

    lines += ["", "### 网格密度提醒", "",
              "流程默认区间是「看全程边界」用的极密网格。几个关键项：", "",
              "| 参数 | 默认展开 | 水平数 |", "|---|---|---|"]
    for d in dims:
        if d.name.endswith(("Pulse0.Voltage", "Pulse0.Width", "Pulse0.Tail",
                            "Frequency", "Nozzle")):
            lines.append("| %s | `%s` | %d |"
                         % (d.name.split(".")[-1], d.range_text, d.n))
    return "\n".join(lines)


PLANS = {"screen": plan_screen, "narrow": plan_narrow,
         "window": plan_window, "delay": plan_delay, "uniform": plan_uniform}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ADA + DropQCS 找最优喷射波形 — 实验设计辅助（不计算耗时）")
    ap.add_argument("--plan", choices=sorted(PLANS), help="输出某阶段 DOE 计划")
    ap.add_argument("--config", choices=sorted(PLANS), help="输出可直接照填的 ADA recipe 片段")
    ap.add_argument("--rules", action="store_true", help="输出 DropQCS 判据文件内容")
    ap.add_argument("--dims", action="store_true", help="真实解析并列出全部可扫描维度")
    ap.add_argument("--head", default="DMCSamba",
                    help="喷头（DMCSamba / KM512 / KM1024i / SpectraSClass / CanonC29）")
    ap.add_argument("--list-heads", action="store_true", help="列出该版本可用的喷头流程")
    ap.add_argument("--recipe", help="指定 ADA recipe 以显示配方覆盖后的值")
    ap.add_argument("--verify", action="store_true",
                    help="交叉校验：代码内的默认范围 vs 真实流程文件")
    ap.add_argument("--target-speed", type=float, default=3.5,
                    help="判据中的目标液滴速度 m/s（**必须按实测标定**；"
                         "默认 3.5，实测速度分布 1.13–6.99 m/s，用 5.0 会让合格率普遍为 0）")
    ap.add_argument("--speed-margin", type=float, default=0.20, help="速度容差百分比")
    ap.add_argument("--angle-max", type=float, default=2.0, help="飞行方向容差(度)")
    args = ap.parse_args(argv)

    did = False
    if args.list_heads:
        for p in sorted(Path(ADA_DIR_C).glob("Advanced-Drop-Analysis-*.xml")):
            if not p.name.endswith(".Screen.xml"):
                print("  %s" % p.stem.replace("Advanced-Drop-Analysis-", ""))
        print("\n（基线：%s）" % NEW_LP50)
        did = True
    if args.dims:
        print(_dims_table(args.recipe, args.head))
        did = True
    if args.verify:
        # 默认用 Samba 的实机 recipe 参与校验：它才有「覆盖值」这一层
        errors, notes = verify_against_flow(head=args.head,
                                            recipe=args.recipe or DEFAULT_RECIPE)
        print("## 交叉校验：代码常量 vs 真实流程文件 + 配方覆盖（%s / 基线 V4730 4.7.3）" % args.head)
        print("   判定口径：喷嘴扫描范围的上界不得超过该喷头**物理孔数**；")
        print("            其余参数配方可合法收窄扫描范围，只提示不算错。")
        print("   物理孔数：%s" % HEAD_NOZZLE_COUNT.get(args.head, "未知"))
        print("   配方：%s" % (args.recipe or DEFAULT_RECIPE))
        if errors:
            print("\n发现 %d 处**错误**（与实机事实冲突，必须同步更新代码常量）：\n" % len(errors))
            for p in errors:
                print("  ✗ " + p)
        else:
            print("\n✓ 全部一致（%d 项）" % (len(SAMBA_WAVEFORM) + len(PROCESS)))
        if notes:
            print("\n另有 %d 处**提示**（非错误，请确认取值意图）：\n" % len(notes))
            for p in notes:
                print("  · " + p)
        did = True
    if args.plan:
        print(PLANS[args.plan]().table())
        did = True
    if args.config:
        print(PLANS[args.config]().recipe_xml())
        did = True
    if args.rules:
        print(dropqcs_rules(args.target_speed, args.speed_margin, args.angle_max))
        did = True
    if not did:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())