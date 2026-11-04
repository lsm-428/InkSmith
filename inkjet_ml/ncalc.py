# -*- coding: utf-8 -*-
"""DropQCS 规则表达式求值器（纯 Python，无需 .NET / NCalc.dll）。

为什么不用真的 NCalc.dll
-----------------------
`NCalc.dll` 是 **.NET 库**（V4730 基线的 `Release\\NCalc.dll`，190 KB），
只在**新基线**那台机台上有（旧机台没有），用 pythonnet 调它需要本机装 .NET Runtime
—— 而本机既无 pythonnet 也无 .NET Runtime。依赖链从"纯 Python"变成"需要 .NET"，
换机/装环境都会踩坑。

实测官方 8 套规则 + Default 用到的**全部 13 条表达式**，语法面只是一个小子集：
    变量 `[Xxx]` / 比较 `== != < > <= >=` / 逻辑 `and or &&` / 算术 `* / + -`
    / 括号 / 常量（bool int float）/ 18 个纯数学内置函数
没有字符串操作、没有正则、没有集合运算 —— 用 Python 的 `ast` 解析 + 白名单求值
即可覆盖，**零新依赖**（`ast` + `math` 都是标准库）。

与 NCalc 的语义差异（已知且已处理）
----------------------------------
1. **`&&` / `||`**：NCalc 接受 C 风格写法，Python 不认 → 预处理成 `and` / `or`。
2. **`true` / `false`**：NCalc 用小写，Python 用 `True` / `False` → 预处理。
3. **`[Result]`**：NCalc 里是**上一条规则的动作结果**（Accept/Reject），是规则链的
   状态而非纯表达式 → 由调用方（规则执行器）作为变量传入，本模块只管求值。
4. **类型提升**：NCalc 是 .NET double 语义；Python 的 `/` 同样是浮点除，
   `int / int` 得 float —— 与 NCalc 一致（`[NrNozzles] / 2` 得 64.0）。

安全性
------
表达式来自机台配置文件，但**仍按不可信输入处理**：用 `ast.parse` 解析后逐节点
白名单校验，**只允许**字面量 / 名字（变量与内置函数）/ 一元二元比较逻辑算术运算 /
函数调用（仅限白名单函数）。任何其它节点（属性访问、导入、lambda、下标、比较链
之外的调用等）一律拒绝并给出明确报错。
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Dict, Mapping, Optional

__all__ = ["NCalcError", "eval_expr", "parse_expr", "BUILTIN_FUNCTIONS",
           "EXPR_VAR_RE"]


class NCalcError(ValueError):
    """表达式语法错误 / 变量缺失 / 用了不允许的语法。"""


#: NCalc 的 18 个内置函数（官方 `Create_ruleset.htm` 的 "build in functions" 表）
#: —— 全是纯数学函数，无字符串/集合操作。
BUILTIN_FUNCTIONS: Dict[str, Any] = {
    "Abs": abs,
    "Acos": math.acos,
    "Asin": math.asin,
    "Atan": math.atan,
    "Ceiling": math.ceil,
    "Cos": math.cos,
    "Exp": math.exp,
    "Floor": math.floor,
    "Log": math.log,
    "Log10": math.log10,
    "Max": max,
    "Min": min,
    "Pow": pow,
    "Round": round,
    "Sign": lambda x: (x > 0) - (x < 0),
    "Sin": math.sin,
    "Sqrt": math.sqrt,
    "Tan": math.tan,
}

#: 表达式里的变量：`[Name]` 形式（NCalc 的写法）
EXPR_VAR_RE = re.compile(r"\[[A-Za-z_][A-Za-z0-9_]*\]")

# 允许的 AST 节点类型（白名单 —— 其它一律拒绝）
_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare,
    ast.Name, ast.Load, ast.Constant, ast.Call,
    ast.And, ast.Or, ast.Not,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)


def _preprocess(expr: str) -> str:
    """把 NCalc 写法转成合法 Python 表达式。

    - `[VarName]` → `_v_VarName`（合法标识符，且不会与用户变量名冲突）
    - `&&` → `and`、`||` → `or`
    - `true`/`false` → `True`/`False`（只替换独立单词，避免动到标识符内部）
    """
    s = expr.strip()
    if not s:
        raise NCalcError("表达式为空")
    # 变量：方括号 → 合法标识符
    s = EXPR_VAR_RE.sub(lambda m: "_v_" + m.group(0)[1:-1], s)
    # 逻辑运算符
    s = s.replace("&&", " and ").replace("||", " or ")
    # 布尔字面量（独立单词才替换）
    s = re.sub(r"\btrue\b", "True", s)
    s = re.sub(r"\bfalse\b", "False", s)
    return s


def parse_expr(expr: str) -> ast.Expression:
    """解析并校验表达式，返回 AST。失败抛 `NCalcError`。"""
    py = _preprocess(expr)
    try:
        tree = ast.parse(py, mode="eval")
    except SyntaxError as e:
        raise NCalcError("表达式语法错误：%s（原文：%s）" % (e.msg, expr.strip())) from e

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise NCalcError(
                "表达式含不允许的语法 %s（原文：%s）"
                % (type(node).__name__, expr.strip()))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise NCalcError("只允许直接调用内置函数（原文：%s）" % expr.strip())
            if node.func.id not in BUILTIN_FUNCTIONS:
                raise NCalcError(
                    "未知内置函数 %s（可用：%s）"
                    % (node.func.id, ", ".join(sorted(BUILTIN_FUNCTIONS))))
            if node.keywords:
                raise NCalcError("内置函数不支持关键字参数（原文：%s）" % expr.strip())
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, bool)):
            raise NCalcError("只支持数值/布尔常量（原文：%s）" % expr.strip())
    return tree


def eval_expr(expr: str, variables: Optional[Mapping[str, Any]] = None,
              strict: bool = True) -> Any:
    """求值一条 NCalc 表达式。

    `variables` 里的键**不带方括号**（如 `NrDroplets`）；表达式里写 `[NrDroplets]`。

    `strict=True`（默认）：变量缺失 → 抛 `NCalcError`
    （不返回 None，避免"缺变量静默判假"）。
    `strict=False`：变量缺失 → 该变量求值为 `None`（Python 语义：与数值比较会
    抛 TypeError，由调用方决定怎么处理）。用于**分两级求值**的场景：
    孔级规则先跑（缺整头级变量），整头级规则在汇总后补上整头统计量再跑。
    """
    tree = parse_expr(expr)
    env: Dict[str, Any] = dict(BUILTIN_FUNCTIONS)
    if variables:
        for k, v in variables.items():
            env["_v_" + k] = v

    # 收集表达式里引用到的变量名（`_v_` 前缀）
    used = {n.id[3:] for n in ast.walk(tree)
            if isinstance(n, ast.Name) and n.id.startswith("_v_")}
    missing = sorted(u for u in used if ("_v_" + u) not in env)
    if missing and strict:
        raise NCalcError("缺少变量：%s" % ", ".join("[%s]" % m for m in missing))
    for u in missing:
        env["_v_" + u] = None

    try:
        code = compile(tree, filename="<ncalc>", mode="eval")
        return eval(code, {"__builtins__": {}}, env)   # noqa: S307 - AST 已白名单校验
    except NCalcError:
        raise
    except ZeroDivisionError as e:
        raise NCalcError("表达式求值时除零：%s" % e) from e
    except (ValueError, TypeError, OverflowError) as e:
        raise NCalcError("表达式求值失败：%s（原文：%s）" % (e, expr.strip())) from e
