"""
[Task 3] Nodal_solver.py
- 대향류(Counter-Current) 단일 노드 계산 모듈
- 별도 물리 계산 사용 금지 → Physics_engine 의 함수만 사용
- 에너지 보존, 변수 업데이트만 수행
- 압력 안전 가드 추가
- GUI로 검증

[변경된 조건 — 양쪽 모두 Water]
  Hot  (Water):  P = 15.0 MPa,  T_in = 600 K
  Cold (Water):  P =  6.0 MPa,  T_in = 530 K
"""

import tkinter as tk
from tkinter import ttk, messagebox

from Data_model import get_fixed_conditions, get_state, T_from_PH
from Physics_engine import evaluate_node, compute_U_for_pair


# ============================================================
# 단일 셀(노드) 진행 — 에너지보존만 수행
# ============================================================
def advance_single_cell(hot_state, cold_state, geom, dx):
    """
    한 셀에서 (x → x+dx) 진행하며 에너지보존으로 다음 상태 계산.

    입력 hot_state, cold_state:
        {"fluid", "P", "T", "m_dot"}
    geom:
        {"A_flow_hot", "A_flow_cold", "D_h", "t_wall", "k_wall",
         "P_w_hot", "P_w_cold"}
    dx: 셀 길이 [m]
    """
    # 1) 현재 셀 입구에서 h, U 계산 (Physics_engine 호출만)
    hot_node = evaluate_node(
        hot_state["fluid"], hot_state["P"], hot_state["T"],
        hot_state["m_dot"], geom["A_flow_hot"], geom["D_h"],
        mode="cooling"
    )
    cold_node = evaluate_node(
        cold_state["fluid"], cold_state["P"], cold_state["T"],
        cold_state["m_dot"], geom["A_flow_cold"], geom["D_h"],
        mode="heating"
    )
    U = compute_U_for_pair(hot_node, cold_node,
                           geom["t_wall"], geom["k_wall"])

    # 2) 셀 전열면적
    P_w_avg = 0.5 * (geom["P_w_hot"] + geom["P_w_cold"])
    dA = P_w_avg * dx

    # 3) 열전달
    dT = hot_state["T"] - cold_state["T"]
    q_cell = U * dA * dT

    # 4) 엔탈피 변화 (Counter-current)
    H_hot_new  = hot_node["H"]  - q_cell / hot_state["m_dot"]
    H_cold_new = cold_node["H"] - q_cell / cold_state["m_dot"]

    # 5) 압력 강하 (Darcy-Weisbach)
    rho_h, rho_c = hot_node["rho"],  cold_node["rho"]
    V_h,   V_c   = hot_node["V"],    cold_node["V"]
    f_h,   f_c   = hot_node["f"],    cold_node["f"]

    dP_hot  = f_h * (dx / geom["D_h"]) * rho_h * V_h**2 / 2.0
    dP_cold = f_c * (dx / geom["D_h"]) * rho_c * V_c**2 / 2.0

    P_hot_new  = hot_state["P"]  - dP_hot
    P_cold_new = cold_state["P"] + dP_cold   # Counter-current: cold는 반대 방향이라 +

    # ★★★ 안전 가드: 압력이 너무 낮아지면 명확한 에러 ★★★
    P_MIN = 1e5   # 1 bar 미만이면 비현실적
    if P_hot_new < P_MIN:
        raise ValueError(
            f"⚠️ Hot 압력이 너무 낮아짐 (P={P_hot_new/1e6:.4f} MPa).\n"
            f"   → A_flow_hot 을 키우거나(예: 1e-3), dx를 줄이세요.\n"
            f"   현재: A_flow_hot={geom['A_flow_hot']}, V_hot={V_h:.2f} m/s"
        )
    if P_cold_new < P_MIN:
        raise ValueError(
            f"⚠️ Cold 압력이 너무 낮아짐 (P={P_cold_new/1e6:.4f} MPa).\n"
            f"   → A_flow_cold 를 키우세요."
        )

    # 6) 엔탈피 → 온도 역산
    T_hot_new  = T_from_PH(hot_state["fluid"],  P_hot_new,  H_hot_new)
    T_cold_new = T_from_PH(cold_state["fluid"], P_cold_new, H_cold_new)

    hot_next = {
        "fluid": hot_state["fluid"], "m_dot": hot_state["m_dot"],
        "P": P_hot_new,  "T": T_hot_new,
    }
    cold_next = {
        "fluid": cold_state["fluid"], "m_dot": cold_state["m_dot"],
        "P": P_cold_new, "T": T_cold_new,
    }
    info = {
        "h_hot":  hot_node["h"],  "h_cold": cold_node["h"], "U": U,
        "Re_hot": hot_node["Re"], "Re_cold": cold_node["Re"],
        "q_cell": q_cell, "dA": dA, "dT": dT,
        "dP_hot": dP_hot, "dP_cold": dP_cold,
        "f_hot":  f_h, "f_cold": f_c,
    }
    return hot_next, cold_next, info


# ============================================================
# GUI
# ============================================================
class NodalSolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("[Task 3] Nodal Solver — 단일 셀 검증 (Water/Water)")
        self.root.geometry("1080x830")
        self._build()
        self._auto_run()

    def _build(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Hot
        hot = ttk.LabelFrame(main, text="HOT 입구 (Water)", padding=6)
        hot.pack(fill=tk.X, pady=2)
        self.hot_E = self._make_state_inputs(hot, ("Water", "15.0", "600", "1.5"))

        # Cold
        cold = ttk.LabelFrame(main, text="COLD 입구 (Water)", padding=6)
        cold.pack(fill=tk.X, pady=2)
        self.cold_E = self._make_state_inputs(cold, ("Water", "6.0", "530", "5.0"))

        # 기하 + 셀 길이
        geom = ttk.LabelFrame(main, text="기하 / 셀 길이", padding=6)
        geom.pack(fill=tk.X, pady=2)
        self.geom_E = {}
        items = [
            ("A_flow_hot",  "Hot 유로 면적 [m²]",   "1e-3"),
            ("A_flow_cold", "Cold 유로 면적 [m²]",  "1e-3"),
            ("P_w_hot",     "Hot 젖은 둘레 [m]",    "0.628"),
            ("P_w_cold",    "Cold 젖은 둘레 [m]",   "0.628"),
            ("D_h",         "D_h [mm]",            "2.0"),
            ("t_wall",      "t_wall [mm]",         "1.0"),
            ("k_wall",      "k_wall [W/mK]",       "20.0"),
            ("dx",          "셀 길이 dx [m]",      "0.01"),
        ]
        for i, (key, label, default) in enumerate(items):
            row = i // 2
            col = i % 2
            f = ttk.Frame(geom); f.grid(row=row, column=col, sticky="w", padx=4, pady=1)
            ttk.Label(f, text=label, width=22).pack(side=tk.LEFT)
            e = ttk.Entry(f, width=12); e.insert(0, default); e.pack(side=tk.LEFT)
            self.geom_E[key] = e

        # 버튼
        btnf = ttk.Frame(main); btnf.pack(fill=tk.X, pady=4)
        ttk.Button(btnf, text="▶ 단일 셀 진행 (1 step)",
                   command=self.step_once).pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        ttk.Button(btnf, text="🔄 입구조건 채우기",
                   command=self._fill_inlet).pack(side=tk.LEFT, padx=2)
        ttk.Button(btnf, text="🔁 100 셀 연속 진행 (테스트)",
                   command=self.run_100_cells).pack(side=tk.LEFT, padx=2)

        # 결과
        rf = ttk.LabelFrame(main, text="결과", padding=6)
        rf.pack(fill=tk.BOTH, expand=True, pady=4)
        self.result_text = tk.Text(rf, font=("Consolas", 10), state=tk.DISABLED)
        self.result_text.pack(fill=tk.BOTH, expand=True)

    def _make_state_inputs(self, parent, defaults):
        fluid_def, P_def, T_def, m_def = defaults
        f = ttk.Frame(parent); f.pack(fill=tk.X, pady=1)
        ttk.Label(f, text="Fluid").pack(side=tk.LEFT)
        cb = ttk.Combobox(f, values=["Water"], width=8)
        cb.set(fluid_def); cb.pack(side=tk.LEFT, padx=2)
        ttk.Label(f, text="P[MPa]").pack(side=tk.LEFT, padx=(8, 2))
        pe = ttk.Entry(f, width=8); pe.insert(0, P_def); pe.pack(side=tk.LEFT)
        ttk.Label(f, text="T[K]").pack(side=tk.LEFT, padx=(8, 2))
        te = ttk.Entry(f, width=8); te.insert(0, T_def); te.pack(side=tk.LEFT)
        ttk.Label(f, text="m_dot[kg/s]").pack(side=tk.LEFT, padx=(8, 2))
        me = ttk.Entry(f, width=8); me.insert(0, m_def); me.pack(side=tk.LEFT)
        return {"fluid": cb, "P": pe, "T": te, "m_dot": me}

    def _fill_inlet(self):
        c = get_fixed_conditions()
        self.hot_E["fluid"].set(c["hot_inlet"]["fluid"])
        self._set(self.hot_E["P"],     c['hot_inlet']['P_in']/1e6)
        self._set(self.hot_E["T"],     c['hot_inlet']['T_in'])
        self._set(self.hot_E["m_dot"], c['hot_inlet']['m_dot'])
        self.cold_E["fluid"].set(c["cold_inlet"]["fluid"])
        self._set(self.cold_E["P"],     c['cold_inlet']['P_in']/1e6)
        self._set(self.cold_E["T"],     c['cold_inlet']['T_in'])
        self._set(self.cold_E["m_dot"], c['cold_inlet']['m_dot'])
        self._set(self.geom_E["D_h"],    c['geometry']['D_h']*1000)
        self._set(self.geom_E["t_wall"], c['geometry']['t_wall']*1000)
        self._set(self.geom_E["k_wall"], c['geometry']['k_wall'])

    def _set(self, e, v):
        e.delete(0, tk.END); e.insert(0, str(v))

    def _read_state(self, E):
        return {
            "fluid":  E["fluid"].get(),
            "P":      float(E["P"].get()) * 1e6,
            "T":      float(E["T"].get()),       # K
            "m_dot":  float(E["m_dot"].get()),
        }

    def _read_geom(self):
        G = self.geom_E
        return {
            "A_flow_hot":  float(G["A_flow_hot"].get()),
            "A_flow_cold": float(G["A_flow_cold"].get()),
            "P_w_hot":     float(G["P_w_hot"].get()),
            "P_w_cold":    float(G["P_w_cold"].get()),
            "D_h":         float(G["D_h"].get()) / 1000,
            "t_wall":      float(G["t_wall"].get()) / 1000,
            "k_wall":      float(G["k_wall"].get()),
        }, float(G["dx"].get())

    def step_once(self):
        try:
            hot  = self._read_state(self.hot_E)
            cold = self._read_state(self.cold_E)
            geom, dx = self._read_geom()
            hot_n, cold_n, info = advance_single_cell(hot, cold, geom, dx)
            self._show_step(hot, cold, hot_n, cold_n, info, dx)
        except Exception as e:
            messagebox.showerror("오류", str(e))

    def run_100_cells(self):
        try:
            hot  = self._read_state(self.hot_E)
            cold = self._read_state(self.cold_E)
            geom, dx = self._read_geom()

            history = [(0.0, hot.copy(), cold.copy())]
            stopped_at = None

            for i in range(100):
                try:
                    hot, cold, _ = advance_single_cell(hot, cold, geom, dx)
                    history.append(((i+1)*dx, hot.copy(), cold.copy()))
                except ValueError as e:
                    stopped_at = (i, str(e))
                    break

            txt = ["━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                   f"  연속 진행 결과 (실제 진행: {len(history)-1} 셀, "
                   f"길이 = {(len(history)-1)*dx:.3f} m)",
                   "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                   f"  {'x[m]':>8s} {'T_hot[K]':>11s} {'T_cold[K]':>11s} "
                   f"{'P_hot[MPa]':>11s} {'P_cold[MPa]':>11s}"]
            step = max(1, (len(history)-1) // 10)
            for x, h, c in history[::step]:
                txt.append(f"  {x:>8.4f} {h['T']:>11.2f} {c['T']:>11.2f} "
                           f"{h['P']/1e6:>11.4f} {c['P']/1e6:>11.4f}")
            txt.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            txt.append(f"  T_hot:  {history[0][1]['T']:.2f} → "
                       f"{history[-1][1]['T']:.2f} K  "
                       f"(Δ = {history[-1][1]['T']-history[0][1]['T']:+.2f} K)")
            txt.append(f"  T_cold: {history[0][2]['T']:.2f} → "
                       f"{history[-1][2]['T']:.2f} K  "
                       f"(Δ = {history[-1][2]['T']-history[0][2]['T']:+.2f} K)")

            if stopped_at:
                txt.append("")
                txt.append(f"⚠️ {stopped_at[0]}번째 셀에서 중단:")
                txt.append(stopped_at[1])

            self._set_text("\n".join(txt))

        except Exception as e:
            messagebox.showerror("오류", str(e))

    def _show_step(self, h, c, hn, cn, info, dx):
        text = (
            "═══════════════════════════════════════════════\n"
            f"  단일 셀 진행 (dx = {dx} m)\n"
            "═══════════════════════════════════════════════\n"
            "  [상태 변화]\n"
            f"   Hot:  T  {h['T']:>8.4f} → {hn['T']:>8.4f} K  "
            f"(Δ = {hn['T']-h['T']:+.4f} K)\n"
            f"         P  {h['P']/1e6:>8.4f} → {hn['P']/1e6:>8.4f} MPa\n"
            f"   Cold: T  {c['T']:>8.4f} → {cn['T']:>8.4f} K  "
            f"(Δ = {cn['T']-c['T']:+.4f} K)\n"
            f"         P  {c['P']/1e6:>8.4f} → {cn['P']/1e6:>8.4f} MPa\n"
            "\n"
            "  [열전달]\n"
            f"   h_hot  = {info['h_hot']:>10.2f} W/m²K   "
            f"Re_hot  = {info['Re_hot']:>9.1f}\n"
            f"   h_cold = {info['h_cold']:>10.2f} W/m²K   "
            f"Re_cold = {info['Re_cold']:>9.1f}\n"
            f"   ★ U   = {info['U']:>10.2f} W/m²K\n"
            f"   ΔT    = {info['dT']:>10.4f} K\n"
            f"   dA    = {info['dA']:>10.6e} m²\n"
            f"   q_cell= {info['q_cell']:>10.4f} W\n"
            "\n"
            "  [압력강하]\n"
            f"   ΔP_hot  = {info['dP_hot']:>10.4f} Pa   (f={info['f_hot']:.5f})\n"
            f"   ΔP_cold = {info['dP_cold']:>10.4f} Pa   (f={info['f_cold']:.5f})\n"
            "\n"
            "  [에너지 보존 검증]\n"
            f"   q_cell = U·dA·ΔT = {info['q_cell']:.4f} W\n"
            f"   ✅ Hot 잃은 열 = Cold 흡수 열 = q_cell\n"
            "═══════════════════════════════════════════════\n"
            "  ※ Counter-current: x↑ 방향 진행 시\n"
            "    Hot은 식고, Cold는 식음 (즉, -x쪽으로 갈수록 Cold가 데워짐)\n"
        )
        self._set_text(text)

    def _set_text(self, text):
        self.result_text.configure(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.configure(state=tk.DISABLED)

    def _auto_run(self):
        self._fill_inlet()
        self.step_once()


# ============================================================
if __name__ == "__main__":
    # 콘솔 검증
    print("=" * 60)
    print("  [Task 3] Nodal Solver 검증 — Water/Water")
    print("=" * 60)
    c = get_fixed_conditions()
    hot  = {"fluid": c['hot_inlet']['fluid'],
            "P": c['hot_inlet']['P_in'],   "T": c['hot_inlet']['T_in'],
            "m_dot": c['hot_inlet']['m_dot']}
    cold = {"fluid": c['cold_inlet']['fluid'],
            "P": c['cold_inlet']['P_in'],  "T": c['cold_inlet']['T_in'],
            "m_dot": c['cold_inlet']['m_dot']}
    geom = {
        "A_flow_hot": 1e-3, "A_flow_cold": 1e-3,
        "P_w_hot":   0.628, "P_w_cold":   0.628,
        "D_h":       c['geometry']['D_h'],
        "t_wall":    c['geometry']['t_wall'],
        "k_wall":    c['geometry']['k_wall'],
    }
    dx = 0.01

    hot_n, cold_n, info = advance_single_cell(hot, cold, geom, dx)
    print(f"  T_hot:  {hot['T']:.2f} → {hot_n['T']:.4f} K")
    print(f"  T_cold: {cold['T']:.2f} → {cold_n['T']:.4f} K")
    print(f"  q_cell = {info['q_cell']:.2f} W,  U = {info['U']:.2f} W/m²K")
    print(f"  Re_hot = {info['Re_hot']:.0f},  Re_cold = {info['Re_cold']:.0f}")
    print(f"  ΔP_hot = {info['dP_hot']:.2f} Pa,  ΔP_cold = {info['dP_cold']:.2f} Pa")

    root = tk.Tk()
    app = NodalSolverGUI(root)
    root.mainloop()