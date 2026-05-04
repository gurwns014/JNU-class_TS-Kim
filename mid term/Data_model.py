"""
[Task 1] Data_model.py
- VHTR-sCO2 사이클 열교환기 설계
- 고정조건 보관 + CoolProp 물성치 호출 함수
- 하드코딩, 수식, 루프 금지 (오로지 데이터/물성치 호출만)
- GUI로 검증 가능

[변경된 조건 — 양쪽 모두 Water]
  Hot  (Water):  P = 15.0 MPa,  T_in = 600 K
  Cold (Water):  P =  6.0 MPa,  T_in = 530 K

※ 양쪽 모두 압축액(subcooled water) 영역
   - 15 MPa 포화온도 ≈ 615.31 K  → Hot 600 K 는 압축액
   -  6 MPa 포화온도 ≈ 548.73 K  → Cold 530 K 는 압축액
"""

import tkinter as tk
from tkinter import ttk, messagebox

try:
    from CoolProp.CoolProp import PropsSI
except ModuleNotFoundError as e:
    PropsSI = None
    _COOLPROP_IMPORT_ERROR = e


# ============================================================
# 고정조건
# ============================================================
FIXED_CONDITIONS = {
    "geometry": {
        "D_h":      2e-3,           # 수력직경 [m]   = 2 mm
        "t_wall":   1e-3,           # 벽 두께 [m]    = 1 mm
        "k_wall":   20.0,           # 벽 열전도율 [W/mK]
    },
    "hot_inlet": {     # Water (고온부)
        "fluid":  "Water",
        "T_in":   600.0,            # K  (변경됨)
        "m_dot":  1.5,              # kg/s
        "P_in":   15.0e6,           # 15.0 MPa  (변경됨)
    },
    "cold_inlet": {    # Water (저온부)
        "fluid":  "Water",
        "T_in":   530.0,            # K  (변경됨)
        "m_dot":  5.0,              # kg/s
        "P_in":   6.0e6,            # 6.0 MPa  (변경됨)
    },
    "target": {
        # Cold 출구 목표 온도. Hot 입구가 600 K이므로 그 이하만 가능.
        # 일단 540 K (압축액 영역 내)로 설정 — GUI/Optimizer에서 조정 가능.
        "T_cold_out": 540.0,        # K
    },
}


# ============================================================
# 물성치 호출 함수 (CoolProp 래퍼)
# ============================================================
def require_coolprop():
    if not COOLPROP_AVAILABLE:
        raise ImportError(
            "CoolProp 패키지가 설치되어 있지 않습니다. `pip install CoolProp` 후 다시 실행하세요."
        ) from _COOLPROP_IMPORT_ERROR


def get_property(fluid, prop, P, T):
    """
    단일 물성치 호출.
    fluid: "Water" 등
    prop : "D"(밀도), "C"(Cp), "L"(k), "V"(μ), "H"(엔탈피)
    P [Pa], T [K]
    """
    require_coolprop()
    return PropsSI(prop, 'P', P, 'T', T, fluid)


def get_state(fluid, P, T):
    """한 점에서의 모든 물성치를 dict로 반환"""
    require_coolprop()
    return {
        "rho": PropsSI('D', 'P', P, 'T', T, fluid),  # 밀도 [kg/m³]
        "Cp":  PropsSI('C', 'P', P, 'T', T, fluid),  # 비열 [J/kgK]
        "k":   PropsSI('L', 'P', P, 'T', T, fluid),  # 열전도율 [W/mK]
        "mu":  PropsSI('V', 'P', P, 'T', T, fluid),  # 점성계수 [Pa·s]
        "H":   PropsSI('H', 'P', P, 'T', T, fluid),  # 엔탈피 [J/kg]
    }


def T_from_PH(fluid, P, H):
    """엔탈피로부터 온도 역산 (마칭에서 사용)"""
    require_coolprop()
    return PropsSI('T', 'P', P, 'H', H, fluid)


def get_fixed_conditions():
    """고정조건 dict 반환"""
    return FIXED_CONDITIONS


# ============================================================
# GUI (검증용)
# ============================================================
class DataModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("[Task 1] Data Model — 고정조건 + 물성치 검증")
        self.root.geometry("900x740")
        self._build()
        self._show_conditions()

    def _build(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # 상단: 고정조건 표시
        cond_frame = ttk.LabelFrame(main, text="고정조건 (Fixed Conditions)", padding=8)
        cond_frame.pack(fill=tk.X, pady=4)
        self.cond_text = tk.Text(cond_frame, height=15, font=("Consolas", 10),
                                 state=tk.DISABLED)
        self.cond_text.pack(fill=tk.BOTH, expand=True)

        # 중간: 물성치 호출 테스트
        test_frame = ttk.LabelFrame(main, text="물성치 호출 검증", padding=8)
        test_frame.pack(fill=tk.X, pady=4)

        row = ttk.Frame(test_frame); row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Fluid", width=8).pack(side=tk.LEFT)
        self.fluid_cb = ttk.Combobox(row, values=["Water"], width=10)
        self.fluid_cb.set("Water")
        self.fluid_cb.pack(side=tk.LEFT, padx=4)

        ttk.Label(row, text="P [MPa]").pack(side=tk.LEFT, padx=(10, 2))
        self.p_entry = ttk.Entry(row, width=10); self.p_entry.insert(0, "15.0")
        self.p_entry.pack(side=tk.LEFT)

        ttk.Label(row, text="T [K]").pack(side=tk.LEFT, padx=(10, 2))
        self.t_entry = ttk.Entry(row, width=10); self.t_entry.insert(0, "600")
        self.t_entry.pack(side=tk.LEFT)

        ttk.Button(row, text="🔍 물성치 호출", command=self.test_property).pack(side=tk.LEFT, padx=10)

        # 결과
        result_frame = ttk.LabelFrame(main, text="결과", padding=8)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        self.result_text = tk.Text(result_frame, font=("Consolas", 10), state=tk.DISABLED)
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # 빠른 검증 버튼
        btnf = ttk.Frame(main); btnf.pack(fill=tk.X, pady=4)
        ttk.Button(btnf, text="✅ Hot 입구 (Water 15MPa, 600K)",
                   command=lambda: self._check_inlet("hot_inlet")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btnf, text="✅ Cold 입구 (Water 6MPa, 530K)",
                   command=lambda: self._check_inlet("cold_inlet")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btnf, text="✅ Cold 목표 출구",
                   command=self._check_target).pack(side=tk.LEFT, padx=2)

    def _show_conditions(self):
        c = FIXED_CONDITIONS
        require_coolprop()

        # 포화온도 정보 (참고용)
        T_sat_hot  = PropsSI('T', 'P', c['hot_inlet']['P_in'],  'Q', 0, 'Water')
        T_sat_cold = PropsSI('T', 'P', c['cold_inlet']['P_in'], 'Q', 0, 'Water')

        text = (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  Geometry\n"
            f"   D_h         = {c['geometry']['D_h']*1000:.2f} mm\n"
            f"   t_wall      = {c['geometry']['t_wall']*1000:.2f} mm\n"
            f"   k_wall      = {c['geometry']['k_wall']:.2f} W/mK\n"
            "\n"
            "  Hot Inlet (Water)  ★ 변경됨\n"
            f"   T_in        = {c['hot_inlet']['T_in']:.2f} K  "
                f"({c['hot_inlet']['T_in']-273.15:.2f} °C)\n"
            f"   m_dot       = {c['hot_inlet']['m_dot']:.2f} kg/s\n"
            f"   P_in        = {c['hot_inlet']['P_in']/1e6:.2f} MPa  "
                f"(T_sat = {T_sat_hot:.2f} K → 압축액)\n"
            "\n"
            "  Cold Inlet (Water)  ★ 변경됨\n"
            f"   T_in        = {c['cold_inlet']['T_in']:.2f} K  "
                f"({c['cold_inlet']['T_in']-273.15:.2f} °C)\n"
            f"   m_dot       = {c['cold_inlet']['m_dot']:.2f} kg/s\n"
            f"   P_in        = {c['cold_inlet']['P_in']/1e6:.2f} MPa  "
                f"(T_sat = {T_sat_cold:.2f} K → 압축액)\n"
            "\n"
            "  Target\n"
            f"   T_cold_out  = {c['target']['T_cold_out']:.2f} K  "
                f"({c['target']['T_cold_out']-273.15:.2f} °C)\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        self.cond_text.configure(state=tk.NORMAL)
        self.cond_text.delete(1.0, tk.END)
        self.cond_text.insert(tk.END, text)
        self.cond_text.configure(state=tk.DISABLED)

    def test_property(self):
        try:
            fluid = self.fluid_cb.get()
            P = float(self.p_entry.get()) * 1e6      # MPa → Pa
            T = float(self.t_entry.get())            # K
            state = get_state(fluid, P, T)
            self._set_result(self._format_state(fluid, P, T, state))
        except Exception as e:
            messagebox.showerror("오류", str(e))

    def _check_inlet(self, key):
        c = FIXED_CONDITIONS[key]
        state = get_state(c["fluid"], c["P_in"], c["T_in"])
        self._set_result(self._format_state(c["fluid"], c["P_in"], c["T_in"], state,
                                             header=f"[{key} — {c['fluid']} 입구]"))

    def _check_target(self):
        c = FIXED_CONDITIONS["cold_inlet"]
        T_target = FIXED_CONDITIONS["target"]["T_cold_out"]
        state = get_state(c["fluid"], c["P_in"], T_target)
        self._set_result(self._format_state(c["fluid"], c["P_in"], T_target, state,
                                             header="[Cold 목표 출구]"))

    def _format_state(self, fluid, P, T, state, header=None):
        lines = []
        if header:
            lines.append(header)
        lines.extend([
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  Fluid : {fluid}",
            f"  P     = {P/1e6:.4f} MPa",
            f"  T     = {T:.4f} K  ({T-273.15:.4f} °C)",
            "─────────────────────────────────────",
            f"  ρ  (밀도)      = {state['rho']:>12.4f} kg/m³",
            f"  Cp (비열)      = {state['Cp']:>12.4f} J/kgK",
            f"  k  (열전도율)  = {state['k']:>12.6f} W/mK",
            f"  μ  (점성계수)  = {state['mu']:>12.6e} Pa·s",
            f"  H  (엔탈피)    = {state['H']:>12.2f} J/kg",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ])
        return "\n".join(lines)

    def _set_result(self, text):
        self.result_text.configure(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.configure(state=tk.DISABLED)


# ============================================================
if __name__ == "__main__":
    # 콘솔 검증 출력
    print("=" * 50)
    print("  [Task 1] Data Model 검증 — Water/Water")
    print("=" * 50)
    c = get_fixed_conditions()
    print(f"  D_h    = {c['geometry']['D_h']*1000} mm")
    print(f"  Hot    : {c['hot_inlet']['fluid']} @ "
          f"{c['hot_inlet']['T_in']} K, "
          f"{c['hot_inlet']['m_dot']} kg/s, "
          f"{c['hot_inlet']['P_in']/1e6} MPa")
    print(f"  Cold   : {c['cold_inlet']['fluid']} @ "
          f"{c['cold_inlet']['T_in']} K, "
          f"{c['cold_inlet']['m_dot']} kg/s, "
          f"{c['cold_inlet']['P_in']/1e6} MPa")
    print(f"  Target : T_cold_out = {c['target']['T_cold_out']} K "
          f"({c['target']['T_cold_out']-273.15:.2f}°C)")
    print()
    print("  [Hot 입구 물성치]")
    s = get_state(c['hot_inlet']['fluid'], c['hot_inlet']['P_in'], c['hot_inlet']['T_in'])
    for k, v in s.items():
        print(f"    {k:4s} = {v:.4e}")
    print()
    print("  [Cold 입구 물성치]")
    s = get_state(c['cold_inlet']['fluid'], c['cold_inlet']['P_in'], c['cold_inlet']['T_in'])
    for k, v in s.items():
        print(f"    {k:4s} = {v:.4e}")

    # GUI 실행
    root = tk.Tk()
    app = DataModelGUI(root)
    root.mainloop()