"""打印模拟器参数计算器：喷头数据库 + 实测铺展 -> film 模式 σ。

数据出处：InteractivePrintingSimulator/printheads.json（喷头规格库）
物理链：drop volume pL -> 球形直径 -> 实测铺展直径 -> 高斯等效 σ=直径/4
      -> σ[px] = σ[µm] * dpi / 25400

用法：
  uv run python calc_sigma.py                      # 默认 KM1024iSHE，铺展 50µm
  uv run python calc_sigma.py --spread-um 45 --dpi 1080
"""
import argparse
import json
from pathlib import Path

HEADS_DB = Path(__file__).parent.parent / "InteractivePrintingSimulator" / "printheads.json"


def sphere_diameter_um(volume_pl):
    """pL 液滴的球形直径（µm）。1 pL = 1e-12 L = 1e3 µm³。"""
    return (6 * volume_pl * 1e3 / 3.141592653589793) ** (1 / 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="KM1024iSHE")
    p.add_argument("--spread-um", type=float, default=50.0,
                   help="实测墨滴在基材上的铺展直径（µm）")
    p.add_argument("--dpi", type=int, default=None, help="只看这一个 dpi")
    args = p.parse_args()

    # 文件是 JS 赋值格式：printheadDatabase = [...]
    text = HEADS_DB.read_text()
    text = text[text.index("["):text.rindex("]") + 1]
    heads = json.loads(text)
    head = next((h for h in heads if h.get("Model") == args.head), None)
    if head is None:
        raise SystemExit(f"数据库 {HEADS_DB} 里找不到喷头 {args.head}")

    vol = float(head["drop volume pL"])
    d_sphere = sphere_diameter_um(vol)
    sigma_um = args.spread_um / 4  # 高斯等效：99.7% 能量在 6σ ≈ 1.5×直径 内
    dot_gain = args.spread_um / d_sphere

    print(f"喷头：{args.head}（{head['Manufacturer']}）")
    print(f"  墨滴体积 {vol} pL -> 球形直径 {d_sphere:.1f} µm")
    print(f"  实测铺展直径 {args.spread_um} µm -> dotGain(直径比) ≈ {dot_gain:.2f}")
    print(f"  高斯等效 σ = 铺展直径/4 = {sigma_um:.2f} µm")
    print(f"  基材：非吸收膜，100% 固含 -> 线性体积模型（FilmSimulator，无 exp 饱和）")
    print()
    dpis = [args.dpi] if args.dpi else [508, 720, 1080, 1440, 2880, 5000]
    print(f"{'dpi':>6} {'像素µm':>8} {'σ[px]':>8} {'铺展直径[px]':>12}")
    for dpi in dpis:
        um_px = 25400 / dpi
        print(f"{dpi:>6} {um_px:>8.2f} {sigma_um / um_px:>8.3f} "
              f"{args.spread_um / um_px:>12.2f}")
    print()
    d = args.dpi or 720
    sigma_px = sigma_um * d / 25400.0
    # 原训练结晶下限（铁律2 / PITFALLS #11/#52：σ<1.4px 退火躺平）；NN 移除后
    # 该阈值改读为**评测/铺展语义**：σ[px]<1.4 时相邻落墨不再融合，one-pass
    # 单遍膜呈麻点（微观锯齿）。本脚本的用途是「σ 出处可溯」，低于该值必须
    # 显式警告，不能让人照着打印的命令跑。
    if sigma_px < 1.4:
        print(f"⚠ σ = {sigma_px:.2f}px @ {d}dpi **低于铺展下限 1.4px**：墨滴"
              f"不融合，one-pass 单遍膜会是麻点（原训练下限 #11 同源）。"
              f"提高 dpi 或降低 --spread-um 使其 ≥1.4px，"
              f"例如 --dpi 1440 --spread-um 25（=1.42px）。")
    print("评测/推理命令（把 --dpi 换成实际打印分辨率，物理参数即 CLI 参数）：")
    print(f"  uv run python eval.py batch --sigma-um {sigma_um:.2f} --dpi {d}")
    print(f"  uv run python eval.py infer 图.jpg --sigma-um {sigma_um:.2f} "
          f"--dpi {d} --out out/photo")


if __name__ == "__main__":
    main()
