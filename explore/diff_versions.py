# -*- coding: utf-8 -*-
"""逐喷头对比两版 ADA 流程参数，列出全部差异（V4730 为准的修改依据）。

A = C:\LP50                                  (HMI 4.4.5)
B = D:\Oleton\Inkjet\LP50 V4730 4-2026 & Canon FAT  (HMI 4.7.3)  ← 新基线
"""
import re
from pathlib import Path

A = Path(r"C:\LP50")
B = Path(r"D:\Oleton\Inkjet\LP50 V4730 4-2026 & Canon FAT")


def attrs(tag):
    return {n: (a if a != "" else b)
            for n, a, b in re.findall(r"""([A-Za-z][\w.]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""", tag)}


def parse_flow(p):
    params, cbs = {}, set()
    t = p.read_text(encoding="utf-8-sig", errors="replace")
    for m in re.finditer(r"<Parameter\b([^>]*?)/?>", t, re.S):
        a = attrs(m.group(1))
        n = a.get("Name", "")
        if n.endswith(".Checkbox"):
            cbs.add(n[: -len(".Checkbox")])
        elif n.startswith("DropAnalysis."):
            params[n] = a
    return params, cbs


def parse_recipe(p):
    if not p.is_file():
        return {}
    t = p.read_text(encoding="utf-8-sig", errors="replace")
    out = {}
    for m in re.finditer(r"<Parameter\b([^>]*?)/?>", t, re.S):
        a = attrs(m.group(1))
        n = re.sub(r"\[\d+\]$", "", a.get("Name", ""))
        if "Value" in a:
            out[n] = a["Value"]
    return out


def flows_of(root):
    d = root / "Recipes" / "95 ADA"
    return {p.stem.replace("Advanced-Drop-Analysis-", ""): p
            for p in sorted(d.glob("Advanced-Drop-Analysis-*.xml"))
            if not p.name.endswith(".Screen.xml")}


FA, FB = flows_of(A), flows_of(B)
print("A(4.4.5) 流程: %s" % sorted(FA))
print("B(4.7.3) 流程: %s" % sorted(FB))
print()

RA = {p.stem: parse_recipe(p) for p in (A / "Recipes" / "95 ADA").glob("*.recipe.xml")}
RB = {p.stem: parse_recipe(p) for p in (B / "Recipes" / "95 ADA").glob("*.recipe.xml")}

for head in sorted(set(FA) | set(FB)):
    if head not in FB:
        print("### %s : 仅 4.4.5 有，4.7.3 已无" % head)
        continue
    if head not in FA:
        print("### %s : ★4.7.3 新增" % head)
        pa, _ = parse_flow(FB[head])
        _, cb = parse_flow(FB[head])
        dims = [n for n in pa if n in cb]
        print("    参数=%d 可扫维度=%d" % (len(pa), len(dims)))
        continue

    pa, ca = parse_flow(FA[head])
    pb, cb = parse_flow(FB[head])
    da = {n: a for n, a in pa.items() if n in ca}
    db = {n: a for n, a in pb.items() if n in cb}
    print("### %s   4.4.5: %d 参数/%d 可扫   →   4.7.3: %d 参数/%d 可扫"
          % (head, len(pa), len(da), len(pb), len(db)))

    # 默认值变化
    changed = [(n, da[n].get("Default", ""), db[n].get("Default", ""))
               for n in sorted(set(da) & set(db))
               if da[n].get("Default", "") != db[n].get("Default", "")]
    if changed:
        print("    -- 流程默认变化 --")
        for n, x, y in changed:
            print("       %-56s %-16s → %s" % (n, x, y))
    # 新增可扫维度（不逐条列 KM1024i 的 105 个）
    added = sorted(set(db) - set(da))
    if added:
        groups = {}
        for n in added:
            key = re.sub(r"\{[^}]+\}|\d+", "*", n)
            groups.setdefault(key, []).append(n)
        print("    -- 4.7.3 新增可扫维度 %d 个（按模式归并）--" % len(added))
        for k, v in sorted(groups.items()):
            print("       %-56s x%d   e.g. %s" % (k, len(v), v[0]))
    removed = sorted(set(da) - set(db))
    if removed:
        print("    -- 4.7.3 移除 %d 个：%s" % (len(removed), ", ".join(removed[:6])))
    # recipe 覆盖差异
    ra = next((v for k, v in RA.items() if head.lower() in k.lower()), {})
    rb = next((v for k, v in RB.items() if head.lower() in k.lower()), {})
    rc = [(n, ra[n], rb[n]) for n in sorted(set(ra) & set(rb)) if ra[n] != rb[n]
          and (n.endswith(".Checkbox") is False) and ra.get(n) and rb.get(n)]
    cb_diff = [(n, ra.get(n), rb.get(n)) for n in sorted(set(ra) & set(rb))
               if n.endswith(".Checkbox") and ra[n] != rb[n]]
    if rc or cb_diff:
        print("    -- recipe 覆盖差异 --")
        for n, x, y in rc:
            print("       %-56s %-16s → %s" % (n, x, y))
        if cb_diff:
            print("       Checkbox 变化 %d 处" % len(cb_diff))
    print()