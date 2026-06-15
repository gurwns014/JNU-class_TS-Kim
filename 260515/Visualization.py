"""
[Visualization.py] 참고 그래프 스타일
=========================================
참고 이미지와 동일한 레이아웃:
  - 상단: Cold(메인) + Hot(보조) 온도 프로파일
  - 3구간 라벨 (양방향 화살표): ① Subcooled  ② Boiling  ③ Superheated
  - 천이점 큰 ○ 마커 (Onset of Boiling, End of Boiling)
  - 입구/출구 큰 ○ + T_in, T_out 텍스트
  - 우측 정보 박스 (P, q'', m_dot, D, L)
  - 하단: Key Results 표 + Node Results 표 + Legend
"""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.gridspec import GridSpec
matplotlib.rcParams["axes.unicode_minus"] = False


# 색상 정의 (참고 이미지와 동일)
COLOR_COLD     = "#1f77b4"   # 파랑 (subcooled)
COLOR_BOIL     = "#d62728"   # 빨강 (boiling onset)
COLOR_SUP      = "#2ca02c"   # 초록 (superheated)
COLOR_HOT      = "#b22222"   # 진빨강 (Hot 보조선)
COLOR_REGION1  = "#1f77b4"
COLOR_REGION2  = "#d62728"
COLOR_REGION3  = "#2ca02c"


# ============================================================
def _extract(result):
    nd = result["node_data"]
    return {
        "L":           result["L"],
        "N":           result["N"],
        "mode":        result.get("mode", "counter"),
        "model":       result.get("model", "?"),
        "x_pos":       [n["x_pos"]      for n in nd],
        "T_hot":       [n["T_hot"]      for n in nd],
        "T_cold":      [n["T_cold"]     for n in nd],
        "x_cold":      [n["x_cold"]     for n in nd],
        "x_hot":       [n["x_hot"]      for n in nd],
        "regime_cold": [n["regime_cold"]for n in nd],
        "regime_hot":  [n["regime_hot"] for n in nd],
        "h_cold":      [n["h_cold"]     for n in nd],
        "h_hot":       [n["h_hot"]      for n in nd],
        "P_cold":      [n["P_cold"]     for n in nd],
        "P_hot":       [n["P_hot"]      for n in nd],
        "H_cold":      [n["H_cold"]     for n in nd],
        "q_cell":      [n["q_cell"]     for n in nd],
        "q_flux":      [n["q_flux"]     for n in nd],
    }


def _find_transitions(x, regime):
    """천이 인덱스 찾기 (subcooled→two_phase, two_phase→superheated)"""
    onset_idx = None
    end_idx = None
    for i in range(1, len(regime)):
        if onset_idx is None and regime[i] == "two_phase" \
           and regime[i-1] in ("subcooled", "init"):
            onset_idx = i
        if end_idx is None and regime[i] == "superheated" \
           and regime[i-1] == "two_phase":
            end_idx = i
    return onset_idx, end_idx


def _get_sat_temp(side="cold"):
    try:
        from Solver import saturation_props
        from Data_model import get_fixed_conditions
        c = get_fixed_conditions()
        key = "cold_inlet" if side == "cold" else "hot_inlet"
        sat = saturation_props(c[key]["fluid"], c[key]["P_in"])
        return sat["T_sat"]
    except Exception:
        return None


def _get_operating_info():
    """우측 정보 박스용 운전 조건"""
    try:
        from Data_model import get_fixed_conditions
        c = get_fixed_conditions()
        return {
            "P_cold":      c["cold_inlet"]["P_in"],
            "mdot_cold":   c["cold_inlet"]["m_dot"],
            "P_hot":       c["hot_inlet"]["P_in"],
            "mdot_hot":    c["hot_inlet"]["m_dot"],
            "D_h":         c["geometry"]["D_h"],
        }
    except Exception:
        return None


# ============================================================
def plot_cold_journey(result, fig=None, show_hot=True,
                       T_unit="C", save_csv=False):
    """
    참고 그래프 스타일 - 메인 함수.

    T_unit: "C" (섭씨) 또는 "K" (켈빈)
    show_hot: True면 Hot 곡선도 같이 그림
    """
    arr = _extract(result)
    Lp = arr["L"]

    # 온도 단위 변환
    def conv(T):
        return T - 273.15 if T_unit == "C" else T
    unit = "°C" if T_unit == "C" else "K"

    x_arr  = arr["x_pos"]
    T_cold = [conv(t) for t in arr["T_cold"]]
    T_hot  = [conv(t) for t in arr["T_hot"]]
    rg_c   = arr["regime_cold"]

    T_sat_K = _get_sat_temp("cold")
    T_sat_v = conv(T_sat_K) if T_sat_K else None

    # 천이점
    onset_i, end_i = _find_transitions(x_arr, rg_c)
    onset_x = x_arr[onset_i] if onset_i else None
    end_x   = x_arr[end_i]   if end_i   else None

    # ── Figure layout (참고 이미지처럼 큰 그래프 + 하단 3분할 영역) ──
    if fig is None:
        fig = plt.figure(figsize=(13, 9))
    fig.clear()

    gs = GridSpec(2, 3, figure=fig,
                   height_ratios=[3.2, 1.0],
                   width_ratios=[1.0, 1.4, 1.0],
                   hspace=0.32, wspace=0.25,
                   left=0.07, right=0.97, top=0.92, bottom=0.05)

    ax_main = fig.add_subplot(gs[0, :])     # 상단 전체
    ax_key  = fig.add_subplot(gs[1, 0])     # Key Results
    ax_tbl  = fig.add_subplot(gs[1, 1])     # Node table
    ax_leg  = fig.add_subplot(gs[1, 2])     # Legend

    # ── 메인 플롯 ──
    ax_main.set_title("Cold Water / Steam Temperature Profile",
                       fontsize=13, fontweight="bold", pad=42)

    # Cold 곡선: regime별 색 다르게
    seg_x_sub, seg_T_sub = [], []
    seg_x_2p,  seg_T_2p  = [], []
    seg_x_sup, seg_T_sup = [], []
    for xi, Ti, ri in zip(x_arr, T_cold, rg_c):
        if ri == "subcooled":
            seg_x_sub.append(xi); seg_T_sub.append(Ti)
        elif ri == "two_phase":
            seg_x_2p.append(xi); seg_T_2p.append(Ti)
        elif ri == "superheated":
            seg_x_sup.append(xi); seg_T_sup.append(Ti)

    if seg_x_sub:
        ax_main.plot(seg_x_sub, seg_T_sub, ".-", color=COLOR_COLD,
                     ms=5, lw=1.8, zorder=3)
    if seg_x_2p:
        ax_main.plot(seg_x_2p, seg_T_2p, ".-", color=COLOR_BOIL,
                     ms=5, lw=1.8, zorder=3)
    if seg_x_sup:
        ax_main.plot(seg_x_sup, seg_T_sup, ".-", color=COLOR_SUP,
                     ms=5, lw=1.8, zorder=3)

    # Hot 곡선 (보조, 점선)
    if show_hot:
        ax_main.plot(x_arr, T_hot, "--", color=COLOR_HOT, lw=2,
                     alpha=0.7, zorder=2)

    # T_sat 가로 점선
    if T_sat_v is not None:
        ax_main.axhline(T_sat_v, color="gray", lw=1, ls="--", alpha=0.6,
                          zorder=1)

    # 입구 큰 ○ (Cold)
    ax_main.plot(x_arr[0], T_cold[0], "o", ms=14,
                  mfc="white", mec=COLOR_COLD, mew=2.2, zorder=5)

    # 출구 큰 ○ (Cold)
    out_color = (COLOR_SUP if rg_c[-1] == "superheated"
                  else COLOR_BOIL if rg_c[-1] == "two_phase"
                  else COLOR_COLD)
    ax_main.plot(x_arr[-1], T_cold[-1], "o", ms=14,
                  mfc="white", mec=out_color, mew=2.2, zorder=5)

    # Onset of Boiling — 큰 ○ + vertical line
    if onset_i is not None:
        T_on = T_cold[onset_i]
        ax_main.plot(onset_x, T_on, "o", ms=16,
                      mfc="white", mec=COLOR_BOIL, mew=2.5, zorder=5)
        ax_main.axvline(onset_x, color=COLOR_BOIL, lw=1, ls="--",
                         alpha=0.4, zorder=1)

    # End of Boiling — 큰 ○ + vertical line
    if end_i is not None:
        T_end = T_cold[end_i]
        ax_main.plot(end_x, T_end, "o", ms=16,
                      mfc="white", mec=COLOR_SUP, mew=2.5, zorder=5)
        ax_main.axvline(end_x, color=COLOR_SUP, lw=1, ls="--",
                         alpha=0.4, zorder=1)

    ax_main.set_xlabel("Position [m]", fontsize=11)
    ax_main.set_ylabel(f"Temperature [{unit}]", fontsize=11)
    ax_main.set_xlim(-Lp*0.02, Lp*1.04)
    ax_main.grid(alpha=0.3, ls=":")
    ax_main.set_axisbelow(True)

    # ── Y축 여백 강제 확장 (라벨용) ──
    ymin, ymax = ax_main.get_ylim()
    yrange = ymax - ymin
    ax_main.set_ylim(ymin - yrange*0.18, ymax + yrange*0.22)
    ymin, ymax = ax_main.get_ylim()
    yrange = ymax - ymin

    # ── T_sat 텍스트 (T_sat 라인 위에) ──
    if T_sat_v is not None:
        # T_sat 라인의 가운데에 두기 (단, 비등 구간이 있으면 그 가운데)
        if onset_x and end_x:
            x_sat_text = (onset_x + end_x) / 2
        else:
            x_sat_text = Lp * 0.5
        ax_main.text(x_sat_text, T_sat_v + yrange*0.018,
                      f"$T_{{sat}}$ = {T_sat_v:.2f} {unit}",
                      fontsize=10, color="#333", ha="center",
                      style="italic",
                      bbox=dict(boxstyle="round,pad=0.15",
                                facecolor="white", edgecolor="none",
                                alpha=0.8))

    # ── 입구/출구 라벨 (그래프 안쪽 아래) ──
    ax_main.text(x_arr[0] + Lp*0.012, T_cold[0] - yrange*0.04,
                  f"$T_{{in}}$ = {T_cold[0]:.1f} {unit}",
                  fontsize=10, color=COLOR_COLD,
                  ha="left", va="top", fontweight="bold")
    ax_main.text(x_arr[-1] - Lp*0.012, T_cold[-1] + yrange*0.025,
                  f"$T_{{out}}$ = {T_cold[-1]:.1f} {unit}",
                  fontsize=10, color=out_color,
                  ha="right", va="bottom", fontweight="bold")

    # ── Hot 라벨 (Hot 곡선 위에, 영역 라벨과 겹치지 않게) ──
    if show_hot:
        # Hot,in 라벨: 입구쪽 그래프 안쪽 약간 위
        ax_main.text(x_arr[0] + Lp*0.012, T_hot[0] - yrange*0.025,
                      f"$T_{{hot,in}}$ = {T_hot[0]:.1f} {unit}",
                      fontsize=9, color=COLOR_HOT, ha="left", va="top",
                      style="italic", alpha=0.85,
                      bbox=dict(boxstyle="round,pad=0.15",
                                facecolor="white", edgecolor="none",
                                alpha=0.75))
        # Hot,out 라벨: 출구쪽 그래프 안쪽 약간 아래
        ax_main.text(x_arr[-1] - Lp*0.012, T_hot[-1] - yrange*0.025,
                      f"$T_{{hot,out}}$ = {T_hot[-1]:.1f} {unit}",
                      fontsize=9, color=COLOR_HOT, ha="right", va="top",
                      style="italic", alpha=0.85,
                      bbox=dict(boxstyle="round,pad=0.15",
                                facecolor="white", edgecolor="none",
                                alpha=0.75))

    # ── 천이점 텍스트 (아래쪽 빈 공간) ──
    y_trans_text = ymin + yrange*0.04
    if onset_i is not None:
        ax_main.text(onset_x, y_trans_text,
                      f"Onset of Boiling\n$x$ = {onset_x:.2f} m",
                      fontsize=9, color=COLOR_BOIL,
                      ha="center", va="bottom", fontweight="bold")
    if end_i is not None:
        ax_main.text(end_x, y_trans_text,
                      f"End of Boiling\n$x$ = {end_x:.2f} m",
                      fontsize=9, color=COLOR_SUP,
                      ha="center", va="bottom", fontweight="bold")

    # ── 3구간 라벨 (위쪽 바깥, 양방향 화살표) ──
    y_arrow = ymax - yrange*0.05
    y_label = ymax - yrange*0.10
    region_min_width = Lp * 0.025   # 너무 좁은 영역은 라벨 생략

    # 구간 경계 결정 (전이점 없으면 끝까지)
    if onset_x is None and end_x is None:
        # 단상만 있는 경우
        x_start_sub, x_end_sub = 0, Lp
        x_start_2p,  x_end_2p  = None, None
        x_start_sup, x_end_sup = None, None
    elif end_x is None:
        # subcooled + two-phase 만
        x_start_sub, x_end_sub = 0, onset_x
        x_start_2p,  x_end_2p  = onset_x, Lp
        x_start_sup, x_end_sup = None, None
    elif onset_x is None:
        x_start_sub = x_end_sub = None
        x_start_2p,  x_end_2p  = 0, end_x
        x_start_sup, x_end_sup = end_x, Lp
    else:
        x_start_sub, x_end_sub = 0, onset_x
        x_start_2p,  x_end_2p  = onset_x, end_x
        x_start_sup, x_end_sup = end_x, Lp

    def _draw_region_label(xs, xe, text, color):
        if xs is None or xe is None: return
        if (xe - xs) < region_min_width: return
        ax_main.annotate("", xy=(xs, y_arrow), xytext=(xe, y_arrow),
                          arrowprops=dict(arrowstyle="<->", color=color,
                                          lw=1.5))
        ax_main.text((xs+xe)/2, y_label, text,
                      fontsize=10, color=color, ha="center", va="top",
                      fontweight="bold")

    _draw_region_label(x_start_sub, x_end_sub,
                        "①  Subcooled\nLiquid Region", COLOR_COLD)
    _draw_region_label(x_start_2p, x_end_2p,
                        "②  Boiling Region\n(Saturated at $T_{sat}$)",
                        COLOR_BOIL)
    _draw_region_label(x_start_sup, x_end_sup,
                        "③  Superheated\nVapor Region", COLOR_SUP)

    # 우측 정보 박스
    op = _get_operating_info()
    if op:
        info_text = (
            f"$P$ = {op['P_cold']/1e6:.2f} MPa\n"
            f"$\\dot{{m}}$ = {op['mdot_cold']:.3f} kg/s\n"
            f"$D_{{inner}}$ = {op['D_h']*1000:.1f} mm\n"
            f"$L$ = {Lp:.2f} m\n"
            f"Model: {arr['model']}"
        )
        ax_main.text(0.985, 0.04, info_text,
                      transform=ax_main.transAxes,
                      fontsize=9.5, ha="right", va="bottom",
                      family="monospace",
                      bbox=dict(boxstyle="round,pad=0.5",
                                facecolor="white", edgecolor="#444",
                                linewidth=1))

    # ── 하단 1: Key Results ──
    ax_key.axis("off")
    ax_key.set_title("Key Results", fontsize=11, fontweight="bold",
                      loc="left", pad=5)
    out_phase = "Superheated" if rg_c[-1] == "superheated" \
                else "Two-phase" if rg_c[-1] == "two_phase" else "Subcooled"
    Q_tot = sum(arr["q_cell"])
    key_lines = [
        f"• Saturation Temperature $T_{{sat}}$  : {T_sat_v:.2f} {unit}"
        if T_sat_v else "• Saturation Temperature   : N/A",
        f"• Onset of Boiling $x$        : "
        f"{onset_x:.3f} m" if onset_x is not None else "• Onset of Boiling           : N/A",
        f"• End of Boiling $x$          : "
        f"{end_x:.3f} m" if end_x is not None else "• End of Boiling             : N/A",
        f"• Outlet Temperature        : {T_cold[-1]:.2f} {unit}  ({out_phase})",
        f"• Outlet Quality            : {arr['x_cold'][-1]:.3f}",
        f"• Total Length              : {Lp:.2f} m",
        f"• Total Heat Duty $Q$         : {Q_tot/1000:.2f} kW",
    ]
    for i, line in enumerate(key_lines):
        ax_key.text(0.02, 0.92 - i*0.13, line,
                     transform=ax_key.transAxes,
                     fontsize=9, family="DejaVu Sans", va="top")

    # ── 하단 2: Node Table ──
    ax_tbl.axis("off")
    ax_tbl.set_title("Node Results (sample rows)",
                       fontsize=11, fontweight="bold",
                       loc="left", pad=5)
    nd = result["node_data"]
    # 균등 5점 샘플
    N = len(nd)
    if N <= 6:
        sample_idx = list(range(N))
    else:
        sample_idx = [0,
                       N//5,
                       2*N//5,
                       3*N//5,
                       4*N//5,
                       N-1]
    rows = []
    rows.append(["pos[m]", "T_c[K]", "x_c", "phase"])
    for i in sample_idx:
        n = nd[i]
        phase = n["regime_cold"]
        rows.append([
            f"{n['x_pos']:6.3f}",
            f"{n['T_cold']:7.2f}",
            f"{n['x_cold']:6.3f}",
            phase[:11],
        ])
    # 표 그리기
    n_rows = len(rows); n_cols = 4
    col_x = [0.02, 0.28, 0.50, 0.72]
    for j, hdr in enumerate(rows[0]):
        ax_tbl.text(col_x[j], 0.85, hdr,
                     transform=ax_tbl.transAxes,
                     fontsize=9, fontweight="bold", va="top")
    for i, row in enumerate(rows[1:]):
        for j, cell in enumerate(row):
            ax_tbl.text(col_x[j], 0.72 - i*0.12, cell,
                         transform=ax_tbl.transAxes,
                         fontsize=8.5, family="monospace", va="top")

    # ── 하단 3: Legend ──
    ax_leg.axis("off")
    ax_leg.set_title("Legend (Phase)", fontsize=11, fontweight="bold",
                       loc="left", pad=5)
    legend_items = [
        (COLOR_COLD, "Subcooled liquid ($T < T_{sat}$)"),
        (COLOR_BOIL, "Boiling region ($T = T_{sat}$)"),
        (COLOR_SUP,  "Superheated vapor ($T > T_{sat}$)"),
    ]
    if show_hot:
        legend_items.append((COLOR_HOT, "Hot side (dashed)"))
    for i, (color, label) in enumerate(legend_items):
        y = 0.85 - i*0.18
        ax_leg.plot([0.05, 0.18], [y, y], "-",
                     color=color, lw=2.5,
                     transform=ax_leg.transAxes,
                     clip_on=False)
        ax_leg.text(0.22, y, label,
                     transform=ax_leg.transAxes,
                     fontsize=9, va="center")

    # CSV 저장 옵션
    if save_csv:
        save_node_csv(result, "cold_nodes.csv")

    return fig


# ============================================================
def save_node_csv(result, path="cold_nodes.csv"):
    """노드 데이터 CSV 저장"""
    import csv
    nd = result["node_data"]
    keys = ["node", "x_pos", "T_hot", "T_cold", "x_cold", "x_hot",
            "regime_cold", "regime_hot", "P_cold", "P_hot",
            "H_cold", "H_hot", "U", "h_hot", "h_cold",
            "q_cell", "q_flux"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for n in nd:
            w.writerow({k: n.get(k, "") for k in keys})
    return path


# ============================================================
def print_summary(result):
    nd = result["node_data"]
    Lp = result["L"]
    mode = result.get("mode", "counter")
    print("=" * 65)
    print(f"  Summary — {result.get('model','?')} correlation, "
          f"L={Lp:.3f}m, N={result['N']}, mode={mode}")
    print("=" * 65)
    print(f"  converged: {result['converged']}")
    print(f"  Cold inlet  : T = {nd[0]['T_cold']:.2f} K  ({nd[0]['T_cold']-273.15:.2f} °C)")
    print(f"  Cold outlet : T = {nd[-1]['T_cold']:.2f} K  ({nd[-1]['T_cold']-273.15:.2f} °C), "
          f"x = {nd[-1]['x_cold']:.3f}")
    print(f"  Hot  inlet  : T = {nd[0]['T_hot']:.2f} K")
    print(f"  Hot  outlet : T = {nd[-1]['T_hot']:.2f} K")
    from collections import Counter
    rc = Counter(n["regime_cold"] for n in nd if n["regime_cold"] != "init")
    print(f"  Cold regime distribution: {dict(rc)}")
    Q = sum(n["q_cell"] for n in nd)
    print(f"  Total heat duty Q = {Q/1000:.2f} kW")


# ============================================================
def save_plots(result, prefix="result", dpi=150,
                T_unit="C", show_hot=True):
    """그래프 PNG + CSV 동시 저장"""
    fig = plot_cold_journey(result, T_unit=T_unit, show_hot=show_hot)
    f_png = f"{prefix}_temperature_profile.png"
    fig.savefig(f_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    f_csv = save_node_csv(result, f"{prefix}_nodes.csv")
    return f_png, f_csv


# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from Data_model import FIXED_CONDITIONS
    FIXED_CONDITIONS["cold_inlet"]["m_dot"] = 0.05

    from Solver import solve_counter_current

    print("[Visualization] running test...")
    result = solve_counter_current(
        L=6.0, N=150,
        geom_extra={"A_flow_hot": 1e-3, "A_flow_cold": 1e-3,
                    "P_w_hot": 0.628, "P_w_cold": 0.628},
        boil_corr="chen", shoot_tol=3.0, shoot_max=30
    )
    print_summary(result)
    f_png, f_csv = save_plots(result, prefix="cold_water")
    print(f"\n  Saved: {f_png}, {f_csv}")

    FIXED_CONDITIONS["cold_inlet"]["m_dot"] = 5.0
