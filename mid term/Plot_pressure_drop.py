"""
Plot_pressure_drop.py
=====================
Plot cumulative pressure drop along the channel length.
  - x-axis: position L [m]
  - y-axis: dP = P_in - P(x) [kPa]
  - Two curves on one plot: Hot, Cold

For Cold side in the two-phase region, the user-selected two-phase
pressure drop model (from Pressure_drop.py) is used.
For all single-phase regions, the friction gradient from
Solver's friction_factor is used (Darcy-Weisbach).

Hot side is single-phase throughout (no boiling), so its dP comes
from the Solver result directly.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["axes.unicode_minus"] = False

from Pressure_drop import dpdz_two_phase
from Physics_engine import friction_factor

try:
    from CoolProp.CoolProp import PropsSI
except ImportError:
    PropsSI = None


def _saturated_props(fluid, P):
    """Get saturated liquid + vapor properties at pressure P."""
    rho_l = PropsSI('D', 'P', P, 'Q', 0, fluid)
    rho_g = PropsSI('D', 'P', P, 'Q', 1, fluid)
    mu_l  = PropsSI('V', 'P', P, 'Q', 0, fluid)
    mu_g  = PropsSI('V', 'P', P, 'Q', 1, fluid)
    T_sat = PropsSI('T', 'P', P, 'Q', 0, fluid)
    sigma = PropsSI('I', 'T', T_sat, 'Q', 0, fluid)
    return dict(rho_l=rho_l, rho_g=rho_g,
                mu_l=mu_l,  mu_g=mu_g,  sigma=sigma)


def _single_phase_dpdz(G, D_h, rho, mu):
    """Single-phase Darcy-Weisbach gradient."""
    Re = G * D_h / mu
    f  = friction_factor(Re)
    return f * G * G / (2.0 * rho * D_h)


def compute_pressure_drop_profiles(result, A_flow_hot, A_flow_cold,
                                     mdot_hot, mdot_cold,
                                     fluid_hot="Water", fluid_cold="Water",
                                     P_hot=15e6, P_cold=6e6,
                                     two_phase_model="friedel"):
    """
    Compute cumulative pressure drop dP(x) = P_in - P(x) for Hot and Cold sides.

    Returns:
        x_arr   : positions [m]
        dP_hot  : cumulative pressure drop on hot side [Pa]
        dP_cold : cumulative pressure drop on cold side [Pa]
    """
    nd = result["node_data"]
    L  = result["L"]
    N  = result["N"]
    dx = L / N

    # Properties (saturated — used in two-phase region)
    sat_cold = _saturated_props(fluid_cold, P_cold)

    # For single-phase regions, we use the local fluid state from CoolProp
    # using P and T of each node.
    G_hot  = mdot_hot  / A_flow_hot
    G_cold = mdot_cold / A_flow_cold

    # Geometry: D_h from FIXED_CONDITIONS
    from Data_model import get_fixed_conditions
    D_h = get_fixed_conditions()["geometry"]["D_h"]

    x_arr   = []
    dP_hot  = []
    dP_cold = []

    cum_hot  = 0.0
    cum_cold = 0.0

    for i, n in enumerate(nd):
        x_arr.append(n["x_pos"])

        if i == 0:
            dP_hot.append(0.0)
            dP_cold.append(0.0)
            continue

        # ── Hot side: always single-phase (water/water case) ──
        # Use local properties at this node
        T_h = nd[i]["T_hot"]
        P_h = nd[i]["P_hot"]
        try:
            rho_h = PropsSI('D', 'P', P_h, 'T', T_h, fluid_hot)
            mu_h  = PropsSI('V', 'P', P_h, 'T', T_h, fluid_hot)
            grad_h = _single_phase_dpdz(G_hot, D_h, rho_h, mu_h)
        except Exception:
            grad_h = 0.0
        cum_hot += grad_h * dx
        dP_hot.append(cum_hot)

        # ── Cold side: region-dependent ──
        regime = n["regime_cold"]
        if regime == "two_phase":
            x_quality = n["x_cold"]
            grad_c = dpdz_two_phase(two_phase_model,
                                      x_quality, G_cold, D_h, sat_cold)
        else:
            # single-phase liquid or vapor — local props
            T_c = n["T_cold"]
            P_c = n["P_cold"]
            try:
                rho_c = PropsSI('D', 'P', P_c, 'T', T_c, fluid_cold)
                mu_c  = PropsSI('V', 'P', P_c, 'T', T_c, fluid_cold)
                grad_c = _single_phase_dpdz(G_cold, D_h, rho_c, mu_c)
            except Exception:
                grad_c = 0.0
        cum_cold += grad_c * dx
        dP_cold.append(cum_cold)

    return x_arr, dP_hot, dP_cold


def plot_pressure_drop(result, A_flow_hot=1e-3, A_flow_cold=1e-3,
                         mdot_hot=None, mdot_cold=None,
                         fluid_hot="Water", fluid_cold="Water",
                         P_hot=15e6, P_cold=6e6,
                         two_phase_model="friedel",
                         show_temperature=True, T_unit="C",
                         fig=None):
    """
    Plot Hot/Cold cumulative pressure drop vs position on one figure.

    If show_temperature=True, temperatures are overlaid on a twin y-axis.
        T_unit : "C" or "K"
    """
    # Get default mdot from FIXED_CONDITIONS if not given
    from Data_model import get_fixed_conditions
    c = get_fixed_conditions()
    if mdot_hot  is None: mdot_hot  = c["hot_inlet"]["m_dot"]
    if mdot_cold is None: mdot_cold = c["cold_inlet"]["m_dot"]

    x_arr, dP_h, dP_c = compute_pressure_drop_profiles(
        result, A_flow_hot, A_flow_cold, mdot_hot, mdot_cold,
        fluid_hot, fluid_cold, P_hot, P_cold,
        two_phase_model=two_phase_model
    )

    # Convert Pa -> kPa
    dP_h_kPa = [p/1000 for p in dP_h]
    dP_c_kPa = [p/1000 for p in dP_c]

    # Counter-current data from solve_counter_current:
    #   x_pos = 0  → Hot inlet  (hot),  Cold outlet (warm)
    #   x_pos = L  → Hot outlet (warm), Cold inlet  (cold ← 530K)
    # So we plot everything vs x_pos directly — no flipping needed.
    # Hot ΔP grows 0→L (inlet to outlet), already left→right.
    # Cold ΔP also accumulated 0→L in solver, but physically cold
    # enters at x=L, so we flip Cold ΔP to show it growing from x=L.
    L = result["L"]
    x_hot_axis  = list(x_arr)               # Hot: ΔP 0 at x=0, max at x=L
    x_cold_axis = [L - x for x in x_arr]    # Cold: ΔP 0 at x=L (inlet), max at x=0 (outlet)
    dP_c_kPa_disp = list(reversed(dP_c_kPa))  # reverse so x=L is 0, x=0 is max

    if fig is None:
        fig = plt.figure(figsize=(11, 6.5))
    fig.clear()
    ax = fig.add_subplot(1, 1, 1)

    # --- pressure drop curves (left axis) ---
    line_dPh, = ax.plot(x_hot_axis, dP_h_kPa, "-", color="#b22222", lw=2.2,
                          marker="o", ms=4, markevery=max(1, len(x_arr)//20),
                          label="Hot $\\Delta P$  (0→L)")
    line_dPc, = ax.plot(x_cold_axis, dP_c_kPa_disp, "-", color="#1565c0", lw=2.2,
                          marker="s", ms=4, markevery=max(1, len(x_arr)//20),
                          label=f"Cold $\\Delta P$  (L→0)  [{two_phase_model}]")

    ax.set_xlabel("Position [m]", fontsize=11)
    ax.set_ylabel("Pressure drop  $\\Delta P$  [kPa]", fontsize=11)
    ax.grid(alpha=0.3, ls=":")

    # --- two-phase region shading ---
    nd = result["node_data"]
    onset_x = end_x = None
    rg_c = [n.get("regime_cold","") for n in nd]
    for i in range(1, len(rg_c)):
        if onset_x is None and rg_c[i] == "two_phase" \
           and rg_c[i-1] in ("subcooled", "init"):
            onset_x = nd[i]["x_pos"]
        if end_x is None and rg_c[i] == "superheated" \
           and rg_c[i-1] == "two_phase":
            end_x = nd[i]["x_pos"]
    if onset_x is not None:
        right = end_x if end_x is not None else x_arr[-1]
        # Cold enters at x=L, so two-phase region is shown as-is (no flip)
        ax.axvspan(onset_x, right, alpha=0.12, color="orange")

    # --- final value annotations (pressure) ---
    if dP_h_kPa:
        ax.text(x_hot_axis[-1], dP_h_kPa[-1],
                f"  {dP_h_kPa[-1]:.2f} kPa",
                fontsize=9, color="#b22222", va="center", ha="left")
        ax.text(x_cold_axis[-1], dP_c_kPa_disp[-1],
                f"  {dP_c_kPa_disp[-1]:.2f} kPa",
                fontsize=9, color="#1565c0", va="center", ha="left")

    # --- temperature overlay (twin right axis) ---
    handles = [line_dPh, line_dPc]
    if show_temperature:
        def _conv(T):
            return T - 273.15 if T_unit == "C" else T
        T_h = [_conv(n["T_hot"])  for n in nd]
        T_c = [_conv(n["T_cold"]) for n in nd]
        x_pos = [n["x_pos"] for n in nd]
        unit = "°C" if T_unit == "C" else "K"

        # No flip — data is already correct:
        #   x=0: T_cold=warm (outlet), T_hot=hot (inlet)
        #   x=L: T_cold=cold (inlet),  T_hot=warm (outlet)
        ax_T = ax.twinx()
        line_Th, = ax_T.plot(x_pos, T_h, "--", color="#d62728", lw=1.6,
                               alpha=0.75, label="Hot T")
        line_Tc, = ax_T.plot(x_pos, T_c, "--", color="#2ca02c", lw=1.6,
                               alpha=0.85, label="Cold T")
        ax_T.set_ylabel(f"Temperature [{unit}]", fontsize=11, color="#444")
        ax_T.tick_params(axis="y", labelcolor="#444")
        handles += [line_Th, line_Tc]

        try:
            T_sat_K = PropsSI('T', 'P', P_cold, 'Q', 0, fluid_cold)
            T_sat_v = _conv(T_sat_K)
            ax_T.axhline(T_sat_v, color="#888", lw=0.9, ls=":", alpha=0.7)
            ax_T.text(L*0.55, T_sat_v + 1.5,
                       f"$T_{{sat}}$ = {T_sat_v:.1f} {unit}",
                       fontsize=8.5, color="#555", style="italic")
        except Exception:
            pass

    # Inlet arrows on top
    ymax_left = max(dP_h_kPa + dP_c_kPa_disp)
    # Hot enters at x=0 → arrow points right
    ax.annotate("", xy=(L*0.15, ymax_left*1.10), xytext=(0, ymax_left*1.10),
                 arrowprops=dict(arrowstyle="->", color="#b22222", lw=1.5),
                 annotation_clip=False)
    ax.text(L*0.075, ymax_left*1.13, "Hot in", color="#b22222",
             fontsize=9, ha="center", fontweight="bold")

    # Cold enters at x=L → arrow points left
    ax.annotate("", xy=(L*0.85, ymax_left*1.10), xytext=(L, ymax_left*1.10),
                 arrowprops=dict(arrowstyle="->", color="#1565c0", lw=1.5),
                 annotation_clip=False)
    ax.text(L*0.925, ymax_left*1.13, "Cold in", color="#1565c0",
             fontsize=9, ha="center", fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    if onset_x is not None:
        handles.append(Patch(facecolor="orange", alpha=0.25,
                              label="Cold two-phase region"))
    ax.legend(handles=handles, loc="center left", fontsize=9, framealpha=0.9)

    ax.set_title("Cumulative Pressure Drop & Temperature  (Counter-Current)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig, (x_arr, dP_h, dP_c)


# ============================================================
# Convenience runner — one call does everything
# ============================================================
def save_pressure_csv(result, x_arr, dP_h, dP_c, path="pressure_drop.csv"):
    """
    Save pressure drop + temperature data to CSV.
    Columns: x_pos, T_hot_C, T_cold_C, regime_cold,
             dP_hot_kPa, dP_cold_kPa
    """
    import csv
    nd = result["node_data"]
    rows = []
    L = result["L"]
    dP_c_disp = list(reversed(dP_c))   # Cold ΔP: 0 at x=L (inlet)
    x_cold_axis = [L - x for x in x_arr]

    for i, n in enumerate(nd):
        rows.append({
            "x_pos_m":       round(n["x_pos"], 6),
            "T_hot_C":       round(n["T_hot"] - 273.15, 4),
            "T_cold_C":      round(n["T_cold"] - 273.15, 4),
            "regime_cold":   n.get("regime_cold", ""),
            "x_quality":     round(n.get("x_cold", 0), 6),
            "dP_hot_kPa":    round(dP_h[i] / 1000, 6),
            "dP_cold_kPa":   round(dP_c_disp[i] / 1000, 6),
            "h_cold_W_m2K":  round(n.get("h_cold", 0), 4),
            "h_hot_W_m2K":   round(n.get("h_hot", 0), 4),
            "q_cell_W":      round(n.get("q_cell", 0), 4),
        })

    fieldnames = ["x_pos_m", "T_hot_C", "T_cold_C", "regime_cold",
                  "x_quality", "dP_hot_kPa", "dP_cold_kPa",
                  "h_cold_W_m2K", "h_hot_W_m2K", "q_cell_W"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return path


def run_and_plot(L=1.0, N=100,
                  A_flow_hot=1e-3, A_flow_cold=1e-3,
                  P_w_hot=0.628, P_w_cold=0.628,
                  P_hot=15e6, P_cold=6e6,
                  boil_corr="chen", two_phase_model="friedel",
                  show_temperature=True, T_unit="C",
                  save_path=None, save_csv=None):
    """
    Run simulation (counter-current) + plot in one call.
      save_path : PNG 저장 경로 (예: "result.png")
      save_csv  : CSV 저장 경로 (예: "result.csv"),  None이면 저장 안 함
    """
    from Solver import solve_counter_current

    result = solve_counter_current(
        L=L, N=N,
        geom_extra={"A_flow_hot": A_flow_hot, "A_flow_cold": A_flow_cold,
                    "P_w_hot":    P_w_hot,    "P_w_cold":    P_w_cold},
        boil_corr=boil_corr,
        shoot_tol=3.0, shoot_max=30
    )
    fig, (x_arr, dP_h, dP_c) = plot_pressure_drop(
        result,
        A_flow_hot=A_flow_hot, A_flow_cold=A_flow_cold,
        P_hot=P_hot, P_cold=P_cold,
        two_phase_model=two_phase_model,
        show_temperature=show_temperature, T_unit=T_unit
    )
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  PNG saved: {save_path}")
    if save_csv:
        save_pressure_csv(result, x_arr, dP_h, dP_c, path=save_csv)
        print(f"  CSV saved: {save_csv}")

    return fig, result, (x_arr, dP_h, dP_c)


# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    fig, result, _ = run_and_plot(L=1.0, N=100,
                                    boil_corr="chen",
                                    two_phase_model="friedel",
                                    save_path="pressure_drop.png",
                                    save_csv="pressure_drop.csv")

    nd = result["node_data"]
    print(f"\n  L = {result['L']} m  ({result['N']} nodes)")
    print(f"  Hot   in/out: {nd[0]['T_hot']-273.15:.2f} / "
          f"{nd[-1]['T_hot']-273.15:.2f} °C")
    print(f"  Cold  in/out: {nd[-1]['T_cold']-273.15:.2f} / "
          f"{nd[0]['T_cold']-273.15:.2f} °C")
