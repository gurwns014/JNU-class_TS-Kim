"""
[Task 2] Physics_engine.py
- node에서 열전달 방정식 계산
- 조건 받아서 h, U 도출
- Data_model의 물성치만 사용 (직접 CoolProp 호출 X)
- GUI로 검증

[변경된 조건 — 양쪽 모두 Water]
  Hot  (Water):  P = 15.0 MPa,  T_in = 600 K
  Cold (Water):  P =  6.0 MPa,  T_in = 530 K
"""

import math
import tkinter as tk
from tkinter import ttk, messagebox
from Data_model import get_fixed_conditions, get_state


# ============================================================
# 물리 계산 함수들 (단일 노드 기준)
# ============================================================
def reynolds(rho, V, D_h, mu):
    """Re = ρVD/μ"""
    return rho * V * D_h / mu


def prandtl(Cp, mu, k):
    """Pr = μCp/k"""
    return mu * Cp / k


def velocity(m_dot, rho, A_flow):
    """V = m_dot / (ρA)"""
    return m_dot / (rho * A_flow)


def nusselt_dittus_boelter(Re, Pr, mode="heating"):
    """
    Dittus-Boelter 상관식:
      Nu = 0.0243 · Re^0.8 · Pr^0.4   (Heating: 유체가 가열될 때, Cold side)
      Nu = 0.0265 · Re^0.8 · Pr^0.3   (Cooling: 유체가 냉각될 때, Hot side)
    층류 (Re < 2300) → Nu = 4.36 (q'' = const)
    """
    if Re < 2300:
        return 4.36
    if mode == "heating":
        return 0.0243 * Re**0.8 * Pr**0.4
    else:  # cooling
        return 0.0265 * Re**0.8 * Pr**0.3


def htc(Nu, k, D_h):
    """h = Nu · k / D_h"""
    return Nu * k / D_h


def overall_U(h_hot, h_cold, t_wall, k_wall):
    """
    총괄 열전달계수
      1/U = 1/h_hot + t_wall/k_wall + 1/h_cold
    """
    return 1.0 / (1.0/h_hot + t_wall/k_wall + 1.0/h_cold)


def friction_factor(Re):
    """마찰계수 (smooth pipe)"""
    if Re < 2300:
        return 64.0 / Re
    # Petukhov / Swamee-Jain 계열 — smooth pipe
    return 0.25 / (math.log10(5.74 / Re**0.9))**2


# ============================================================
# 단일 노드 열전달 평가 (핵심 함수)
# ============================================================
def evaluate_node(fluid, P, T, m_dot, A_flow, D_h, mode="heating"):
    """
    한 노드에서: 물성치 → V, Re, Pr → Nu → h
    반환: dict (모든 중간값 포함)
    """
    state = get_state(fluid, P, T)
    V  = velocity(m_dot, state["rho"], A_flow)
    Re = reynolds(state["rho"], V, D_h, state["mu"])
    Pr = prandtl(state["Cp"], state["mu"], state["k"])
    Nu = nusselt_dittus_boelter(Re, Pr, mode=mode)
    h  = htc(Nu, state["k"], D_h)
    f  = friction_factor(Re)

    return {
        "fluid": fluid, "P": P, "T": T,
        "rho": state["rho"], "Cp": state["Cp"],
        "k":   state["k"],   "mu": state["mu"], "H": state["H"],
        "V": V, "Re": Re, "Pr": Pr, "Nu": Nu,
        "h": h, "f": f, "mode": mode,
    }


def compute_U_for_pair(hot_node, cold_node, t_wall, k_wall):
    """Hot/Cold 두 노드의 h 결과로 U 계산"""
    return overall_U(hot_node["h"], cold_node["h"], t_wall, k_wall)


# ============================================================
# GUI (검증용)
# ============================================================
class PhysicsEngineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("[Task 2] Physics Engine — h, U 검증 (Water/Water)")
        self.root.geometry("1000x800")
        self._build()
        self._auto_test()  # 시작 시 자동으로 입구조건 평가

    def _build(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # 입력
        in_frame = ttk.LabelFrame(main, text="단일 노드 평가 입력", padding=8)
        in_frame.pack(fill=tk.X, pady=4)

        # Hot row
        hot_row = ttk.Frame(in_frame); hot_row.pack(fill=tk.X, pady=2)
        ttk.Label(hot_row, text="[HOT]", width=8, foreground="red").pack(side=tk.LEFT)
        self.hot_entries = self._make_node_inputs(hot_row,
                                                   defaults=("Water", "15.0", "600", "1.5"))

        # Cold row
        cold_row = ttk.Frame(in_frame); cold_row.pack(fill=tk.X, pady=2)
        ttk.Label(cold_row, text="[COLD]", width=8, foreground="blue").pack(side=tk.LEFT)
        self.cold_entries = self._make_node_inputs(cold_row,
                                                    defaults=("Water", "6.0", "530", "5.0"))

        # 기하
        geom_row = ttk.Frame(in_frame); geom_row.pack(fill=tk.X, pady=4)
        ttk.Label(geom_row, text="A_flow [m²]").pack(side=tk.LEFT)
        self.A_flow_entry = ttk.Entry(geom_row, width=10)
        self.A_flow_entry.insert(0, "1e-3")
        self.A_flow_entry.pack(side=tk.LEFT, padx=4)

        ttk.Label(geom_row, text="D_h [mm]").pack(side=tk.LEFT, padx=(10, 2))
        self.Dh_entry = ttk.Entry(geom_row, width=8)
        self.Dh_entry.insert(0, "2.0")
        self.Dh_entry.pack(side=tk.LEFT)

        ttk.Label(geom_row, text="t_wall [mm]").pack(side=tk.LEFT, padx=(10, 2))
        self.tw_entry = ttk.Entry(geom_row, width=8)
        self.tw_entry.insert(0, "1.0")
        self.tw_entry.pack(side=tk.LEFT)

        ttk.Label(geom_row, text="k_wall [W/mK]").pack(side=tk.LEFT, padx=(10, 2))
        self.kw_entry = ttk.Entry(geom_row, width=8)
        self.kw_entry.insert(0, "20.0")
        self.kw_entry.pack(side=tk.LEFT)

        # 버튼
        btnf = ttk.Frame(main); btnf.pack(fill=tk.X, pady=4)
        ttk.Button(btnf, text="🔥 Hot/Cold 노드 평가 + U 계산",
                   command=self.compute).pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        ttk.Button(btnf, text="🔄 입구조건 자동 채우기",
                   command=self._fill_inlet).pack(side=tk.LEFT, padx=2)

        # 결과
        result_frame = ttk.LabelFrame(main, text="결과", padding=8)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        self.result_text = tk.Text(result_frame, font=("Consolas", 10), state=tk.DISABLED)
        self.result_text.pack(fill=tk.BOTH, expand=True)

    def _make_node_inputs(self, parent, defaults):
        fluid_def, P_def, T_def, m_def = defaults
        ttk.Label(parent, text="Fluid").pack(side=tk.LEFT)
        cb = ttk.Combobox(parent, values=["Water"], width=8)
        cb.set(fluid_def); cb.pack(side=tk.LEFT, padx=2)

        ttk.Label(parent, text="P[MPa]").pack(side=tk.LEFT, padx=(8, 2))
        pe = ttk.Entry(parent, width=8); pe.insert(0, P_def); pe.pack(side=tk.LEFT)

        ttk.Label(parent, text="T[K]").pack(side=tk.LEFT, padx=(8, 2))
        te = ttk.Entry(parent, width=8); te.insert(0, T_def); te.pack(side=tk.LEFT)

        ttk.Label(parent, text="m_dot[kg/s]").pack(side=tk.LEFT, padx=(8, 2))
        me = ttk.Entry(parent, width=8); me.insert(0, m_def); me.pack(side=tk.LEFT)

        return {"fluid": cb, "P": pe, "T": te, "m_dot": me}

    def _fill_inlet(self):
        c = get_fixed_conditions()
        # Hot
        self.hot_entries["fluid"].set(c["hot_inlet"]["fluid"])
        self._set_entry(self.hot_entries["P"],     f"{c['hot_inlet']['P_in']/1e6}")
        self._set_entry(self.hot_entries["T"],     f"{c['hot_inlet']['T_in']}")
        self._set_entry(self.hot_entries["m_dot"], f"{c['hot_inlet']['m_dot']}")
        # Cold
        self.cold_entries["fluid"].set(c["cold_inlet"]["fluid"])
        self._set_entry(self.cold_entries["P"],     f"{c['cold_inlet']['P_in']/1e6}")
        self._set_entry(self.cold_entries["T"],     f"{c['cold_inlet']['T_in']}")
        self._set_entry(self.cold_entries["m_dot"], f"{c['cold_inlet']['m_dot']}")
        # 기하
        self._set_entry(self.Dh_entry, f"{c['geometry']['D_h']*1000}")
        self._set_entry(self.tw_entry, f"{c['geometry']['t_wall']*1000}")
        self._set_entry(self.kw_entry, f"{c['geometry']['k_wall']}")

    def _set_entry(self, entry, val):
        entry.delete(0, tk.END); entry.insert(0, val)

    def _read_node(self, entries):
        return {
            "fluid":  entries["fluid"].get(),
            "P":      float(entries["P"].get()) * 1e6,
            "T":      float(entries["T"].get()),       # K
            "m_dot":  float(entries["m_dot"].get()),
        }

    def compute(self):
        try:
            hot  = self._read_node(self.hot_entries)
            cold = self._read_node(self.cold_entries)
            A_flow = float(self.A_flow_entry.get())
            D_h    = float(self.Dh_entry.get()) / 1000
            t_wall = float(self.tw_entry.get()) / 1000
            k_wall = float(self.kw_entry.get())

            # Hot은 cooling (열을 잃음), Cold는 heating (열을 얻음)
            hot_node  = evaluate_node(hot["fluid"],  hot["P"],  hot["T"],
                                       hot["m_dot"],  A_flow, D_h, mode="cooling")
            cold_node = evaluate_node(cold["fluid"], cold["P"], cold["T"],
                                       cold["m_dot"], A_flow, D_h, mode="heating")
            U = compute_U_for_pair(hot_node, cold_node, t_wall, k_wall)

            self._show_result(hot_node, cold_node, U, t_wall, k_wall)

        except Exception as e:
            messagebox.showerror("오류", str(e))

    def _show_result(self, h, c, U, t_wall, k_wall):
        # 저항 분해
        R_hot  = 1.0 / h["h"]
        R_wall = t_wall / k_wall
        R_cold = 1.0 / c["h"]
        R_tot  = R_hot + R_wall + R_cold

        # 비율
        pct_hot  = R_hot  / R_tot * 100
        pct_wall = R_wall / R_tot * 100
        pct_cold = R_cold / R_tot * 100

        text = (
            "═══════════════════════════════════════════════\n"
            "  [HOT 노드 — cooling]\n"
            "═══════════════════════════════════════════════\n"
            f"   Fluid = {h['fluid']:<8s}  P = {h['P']/1e6:.2f} MPa  "
            f"T = {h['T']:.2f} K ({h['T']-273.15:.2f} °C)\n"
            f"   ρ  = {h['rho']:>10.4f} kg/m³    Cp = {h['Cp']:>10.2f} J/kgK\n"
            f"   k  = {h['k']:>10.6f} W/mK     μ  = {h['mu']:>10.4e} Pa·s\n"
            f"   V  = {h['V']:>10.4f} m/s\n"
            f"   Re = {h['Re']:>10.1f}    Pr = {h['Pr']:>8.4f}    "
            f"Nu = {h['Nu']:>8.4f}\n"
            f"   ★ h_hot = {h['h']:>10.2f} W/m²K\n"
            f"   f  = {h['f']:.5f}\n"
            "\n"
            "═══════════════════════════════════════════════\n"
            "  [COLD 노드 — heating]\n"
            "═══════════════════════════════════════════════\n"
            f"   Fluid = {c['fluid']:<8s}  P = {c['P']/1e6:.2f} MPa  "
            f"T = {c['T']:.2f} K ({c['T']-273.15:.2f} °C)\n"
            f"   ρ  = {c['rho']:>10.4f} kg/m³    Cp = {c['Cp']:>10.2f} J/kgK\n"
            f"   k  = {c['k']:>10.6f} W/mK     μ  = {c['mu']:>10.4e} Pa·s\n"
            f"   V  = {c['V']:>10.4f} m/s\n"
            f"   Re = {c['Re']:>10.1f}    Pr = {c['Pr']:>8.4f}    "
            f"Nu = {c['Nu']:>8.4f}\n"
            f"   ★ h_cold = {c['h']:>9.2f} W/m²K\n"
            f"   f  = {c['f']:.5f}\n"
            "\n"
            "═══════════════════════════════════════════════\n"
            "  [총괄 열전달계수 U]\n"
            "═══════════════════════════════════════════════\n"
            f"   R_hot  = 1/h_hot       = {R_hot:>10.6e} m²K/W  ({pct_hot:>5.2f}%)\n"
            f"   R_wall = t/k_wall      = {R_wall:>10.6e} m²K/W  ({pct_wall:>5.2f}%)\n"
            f"   R_cold = 1/h_cold      = {R_cold:>10.6e} m²K/W  ({pct_cold:>5.2f}%)\n"
            f"   R_tot                  = {R_tot:>10.6e} m²K/W\n"
            f"   ★★ U = 1/R_tot         = {U:>10.4f} W/m²K\n"
            "═══════════════════════════════════════════════\n"
            "  ※ Water/Water 양쪽 모두 압축액 — 난류 영역 (Dittus-Boelter)\n"
        )
        self.result_text.configure(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.configure(state=tk.DISABLED)

    def _auto_test(self):
        """시작 시 입구조건으로 자동 평가 → 즉시 검증"""
        self._fill_inlet()
        self.compute()


# ============================================================
if __name__ == "__main__":
    # 콘솔 검증 출력
    print("=" * 60)
    print("  [Task 2] Physics Engine 검증 — Water/Water")
    print("=" * 60)
    c = get_fixed_conditions()
    A_flow = 1e-3   # 임시값 — 실제는 Optimizer에서 설정

    hot = evaluate_node(c['hot_inlet']['fluid'],
                        c['hot_inlet']['P_in'], c['hot_inlet']['T_in'],
                        c['hot_inlet']['m_dot'],
                        A_flow, c['geometry']['D_h'], mode="cooling")
    cold = evaluate_node(c['cold_inlet']['fluid'],
                         c['cold_inlet']['P_in'], c['cold_inlet']['T_in'],
                         c['cold_inlet']['m_dot'],
                         A_flow, c['geometry']['D_h'], mode="heating")
    U = compute_U_for_pair(hot, cold,
                           c['geometry']['t_wall'], c['geometry']['k_wall'])

    print(f"  Hot  (Water 15MPa, 600K):")
    print(f"    V = {hot['V']:.4f} m/s   Re = {hot['Re']:.1f}   Pr = {hot['Pr']:.4f}")
    print(f"    Nu = {hot['Nu']:.2f}   h = {hot['h']:.2f} W/m²K")
    print(f"  Cold (Water  6MPa, 530K):")
    print(f"    V = {cold['V']:.4f} m/s   Re = {cold['Re']:.1f}   Pr = {cold['Pr']:.4f}")
    print(f"    Nu = {cold['Nu']:.2f}   h = {cold['h']:.2f} W/m²K")
    print(f"  ★ U = {U:.2f} W/m²K")

    # GUI
    root = tk.Tk()
    app = PhysicsEngineGUI(root)
    root.mainloop()