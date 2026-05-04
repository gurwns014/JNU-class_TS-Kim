"""
[Task 4] Optimizer.py
- 공간 전진(Space Marching) 루프 + 출구 압력 가정 및 수렴
- Nodal_solver의 advance_single_cell만 호출 (별도 물리계산 금지)
- 노드별 데이터 배열로 저장
- 최종 길이 L 반환
- GUI로 검증

[변경된 조건 — 양쪽 모두 Water]
  Hot  (Water):  P = 15.0 MPa,  T_in = 600 K
  Cold (Water):  P =  6.0 MPa,  T_in = 530 K

대향류 (Counter-Current):
    HOT  (Water):  x=0 입구(600K) ───→ x=L 출구
    COLD (Water):  x=L 입구(530K) ←─── x=0 출구(목표)

알고리즘:
    1. 출구 압력 가정 (P_out = P_in, 압력강하 무시한 가정값으로 시작)
    2. 길이 L과 노드 수 N으로 dx = L/N 분할
    3. Cold 출구 온도 T_cold_x0를 가정 → x=0에서 시작
    4. x=0 → x=L 방향으로 advance_single_cell 반복 호출
    5. x=L에서 계산된 Cold T가 실제 Cold 입구(530K)와 일치할 때까지
       Cold 출구 가정값을 수렴 (Inner Loop: Shooting Method)
    6. Outer Loop: T_cold_x0 = 목표 만족하는 L을 Bisection으로 탐색
"""

import csv
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from Data_model import get_fixed_conditions
from Nodal_solver import advance_single_cell


# ============================================================
# Inner Loop: 길이 L에서 Shooting Method로 정상 해 도출
# ============================================================
def march_with_guess(L, N, T_cold_x0_guess, geom_extra):
    """
    Cold x=0 출구 온도를 T_cold_x0_guess로 가정하고
    x=0 → x=L 방향으로 공간 전진.

    대향류:
        x=0 :  Hot 입구 (600 K),    Cold 출구 (가정값)
        x=L :  Hot 출구 (계산),     Cold 입구 (530 K — 비교용)

    반환: (T_cold_at_xL, node_data, err_msg or None)
    """
    c = get_fixed_conditions()
    fc_hot  = c["hot_inlet"]
    fc_cold = c["cold_inlet"]

    dx = L / N

    # 출구 압력 가정 = 입구 압력 (압력강하 무시)
    # x=0 시점 :
    #   Hot  은 x=0 이 입구 → P_hot_in 그대로
    #   Cold 는 x=0 이 출구 → P_cold_in 그대로 가정 (강하 무시)
    hot = {
        "fluid":  fc_hot["fluid"],
        "P":      fc_hot["P_in"],
        "T":      fc_hot["T_in"],
        "m_dot":  fc_hot["m_dot"],
    }
    cold = {
        "fluid":  fc_cold["fluid"],
        "P":      fc_cold["P_in"],
        "T":      T_cold_x0_guess,    # ← 가정값 (x=0 Cold 출구)
        "m_dot":  fc_cold["m_dot"],
    }

    geom = {
        "A_flow_hot":  geom_extra["A_flow_hot"],
        "A_flow_cold": geom_extra["A_flow_cold"],
        "P_w_hot":     geom_extra["P_w_hot"],
        "P_w_cold":    geom_extra["P_w_cold"],
        "D_h":         c["geometry"]["D_h"],
        "t_wall":      c["geometry"]["t_wall"],
        "k_wall":      c["geometry"]["k_wall"],
    }

    # 노드별 데이터 저장
    node_data = []
    node_data.append({
        "node": 0, "x": 0.0,
        "T_hot":  hot["T"],   "P_hot":  hot["P"],
        "T_cold": cold["T"],  "P_cold": cold["P"],
        "U": 0.0, "h_hot": 0.0, "h_cold": 0.0,
        "Re_hot": 0.0, "Re_cold": 0.0,
        "q_cell": 0.0, "dT": hot["T"]-cold["T"],
    })

    # 공간 전진 루프
    for i in range(N):
        try:
            hot_next, cold_next, info = advance_single_cell(hot, cold, geom, dx)
        except ValueError as e:
            return None, node_data, str(e)

        hot, cold = hot_next, cold_next

        node_data.append({
            "node": i+1, "x": (i+1)*dx,
            "T_hot":  hot["T"],   "P_hot":  hot["P"],
            "T_cold": cold["T"],  "P_cold": cold["P"],
            "U":      info["U"],
            "h_hot":  info["h_hot"], "h_cold": info["h_cold"],
            "Re_hot": info["Re_hot"], "Re_cold": info["Re_cold"],
            "q_cell": info["q_cell"], "dT": info["dT"],
        })

    # x=L 에서 Cold 온도 — 실제 Cold 입구(530K) 와 비교
    T_cold_at_xL = cold["T"]
    return T_cold_at_xL, node_data, None


def shoot_for_cold_inlet(L, N, geom_extra,
                         tol=0.01, max_iter=50):
    """
    Inner loop (Shooting + Bisection):
        T_cold_x0 (Cold 출구 가정값)을 조정해서
        x=L 에서 Cold 온도가 실제 Cold 입구(530 K)와 일치하도록.

    물리 제약:
        Cold 출구 온도 ∈ (Cold 입구, Hot 입구) = (530, 600)

    반환: (T_cold_x0_converged, node_data, ok)
    """
    c = get_fixed_conditions()
    T_cold_in_actual = c["cold_inlet"]["T_in"]  # 530 K

    # Bisection 범위
    T_lo = c["cold_inlet"]["T_in"] + 0.1
    T_hi = c["hot_inlet"]["T_in"]  - 0.1

    last_node_data = None
    T_guess = 0.5*(T_lo + T_hi)

    for _ in range(max_iter):
        T_guess = 0.5 * (T_lo + T_hi)
        T_at_xL, node_data, err_msg = march_with_guess(L, N, T_guess, geom_extra)
        last_node_data = node_data

        if T_at_xL is None:
            # 발산 → Cold 출구 가정값을 줄여본다
            T_hi = T_guess
            continue

        diff = T_at_xL - T_cold_in_actual

        if abs(diff) < tol:
            return T_guess, node_data, True

        # x=L 에서 Cold 온도가 실제 입구보다 높으면
        # → Cold 출구 가정값을 낮춰야 함
        if diff > 0:
            T_hi = T_guess
        else:
            T_lo = T_guess

    return T_guess, last_node_data, False


# ============================================================
# Outer Loop: 길이 L 을 Bisection 으로 탐색
# ============================================================
def optimize_length(N, geom_extra,
                    L_min=0.1, L_max=80.0,
                    tol=0.01, max_iter=40,
                    progress_cb=None):
    """
    Outer loop:
        Cold 출구 온도(=T_cold_x0)가 목표와 일치하는 L 탐색.

    반환: dict(L, T_cold_out, node_data, converged, history, N)
    """
    c = get_fixed_conditions()
    T_target = c["target"]["T_cold_out"]   # 540 K (변경됨)

    history = []
    L_lo, L_hi = L_min, L_max
    last = None
    L_sol = 0.5*(L_lo + L_hi)

    for it in range(1, max_iter+1):
        L_mid = 0.5*(L_lo + L_hi)
        T_x0, node_data, ok = shoot_for_cold_inlet(L_mid, N, geom_extra)

        diff = T_x0 - T_target
        history.append((it, L_mid, T_x0, diff, ok))

        if progress_cb:
            progress_cb(it, L_mid, T_x0, diff)

        last = (L_mid, T_x0, node_data, ok)
        L_sol = L_mid

        if abs(diff) < tol and ok:
            break

        # T_x0(Cold 출구) > 목표 → L 줄이기
        # T_x0(Cold 출구) < 목표 → L 늘리기
        if diff > 0:
            L_hi = L_mid
        else:
            L_lo = L_mid

    L_final, T_x0_final, nd_final, ok_final = last
    return {
        "L":          L_final,
        "T_cold_out": T_x0_final,
        "node_data":  nd_final,
        "converged":  ok_final and abs(T_x0_final - T_target) < tol,
        "history":    history,
        "N":          N,
    }


# ============================================================
# CSV 저장 (Main.py 에서도 사용)
# ============================================================
def save_node_data(node_data, path):
    keys = ["node", "x", "T_hot", "T_cold", "P_hot", "P_cold",
            "U", "h_hot", "h_cold", "Re_hot", "Re_cold", "q_cell", "dT"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for n in node_data:
            w.writerow({k: n.get(k, "") for k in keys})


# ============================================================
# GUI
# ============================================================
class OptimizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("[Task 4] Optimizer — 길이 L 최적화 (Water/Water)")
        self.root.geometry("1120x870")
        self._last_result = None
        self._build()

    def _build(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # 입력 영역
        in_frame = ttk.LabelFrame(main, text="최적화 입력", padding=8)
        in_frame.pack(fill=tk.X, pady=4)

        # 노드/탐색 설정
        row1 = ttk.Frame(in_frame); row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="노드 수 N").pack(side=tk.LEFT)
        self.N_entry = ttk.Entry(row1, width=8); self.N_entry.insert(0, "100")
        self.N_entry.pack(side=tk.LEFT, padx=4)

        ttk.Label(row1, text="L 탐색 범위 [m]").pack(side=tk.LEFT, padx=(15, 2))
        self.Lmin_entry = ttk.Entry(row1, width=8); self.Lmin_entry.insert(0, "0.1")
        self.Lmin_entry.pack(side=tk.LEFT)
        ttk.Label(row1, text="~").pack(side=tk.LEFT, padx=2)
        self.Lmax_entry = ttk.Entry(row1, width=8); self.Lmax_entry.insert(0, "80.0")
        self.Lmax_entry.pack(side=tk.LEFT)

        ttk.Label(row1, text="수렴 tol [K]").pack(side=tk.LEFT, padx=(15, 2))
        self.tol_entry = ttk.Entry(row1, width=8); self.tol_entry.insert(0, "0.01")
        self.tol_entry.pack(side=tk.LEFT)

        ttk.Label(row1, text="최대 반복").pack(side=tk.LEFT, padx=(15, 2))
        self.maxiter_entry = ttk.Entry(row1, width=8); self.maxiter_entry.insert(0, "40")
        self.maxiter_entry.pack(side=tk.LEFT)

        # 기하 설정
        row2 = ttk.Frame(in_frame); row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="A_flow_hot [m²]").pack(side=tk.LEFT)
        self.Ah_entry = ttk.Entry(row2, width=10); self.Ah_entry.insert(0, "1e-3")
        self.Ah_entry.pack(side=tk.LEFT, padx=4)

        ttk.Label(row2, text="A_flow_cold [m²]").pack(side=tk.LEFT, padx=(10, 2))
        self.Ac_entry = ttk.Entry(row2, width=10); self.Ac_entry.insert(0, "1e-3")
        self.Ac_entry.pack(side=tk.LEFT)

        ttk.Label(row2, text="P_w_hot [m]").pack(side=tk.LEFT, padx=(10, 2))
        self.Pwh_entry = ttk.Entry(row2, width=10); self.Pwh_entry.insert(0, "0.628")
        self.Pwh_entry.pack(side=tk.LEFT)

        ttk.Label(row2, text="P_w_cold [m]").pack(side=tk.LEFT, padx=(10, 2))
        self.Pwc_entry = ttk.Entry(row2, width=10); self.Pwc_entry.insert(0, "0.628")
        self.Pwc_entry.pack(side=tk.LEFT)

        # 버튼
        btnf = ttk.Frame(main); btnf.pack(fill=tk.X, pady=4)
        ttk.Button(btnf, text="🚀 최적화 실행 (L 탐색)",
                   command=self.run_optimize).pack(side=tk.LEFT, padx=2,
                                                    expand=True, fill=tk.X)
        ttk.Button(btnf, text="💾 노드 데이터 CSV 저장",
                   command=self.save_csv).pack(side=tk.LEFT, padx=2)

        # 결과 영역
        rf = ttk.LabelFrame(main, text="결과", padding=8)
        rf.pack(fill=tk.BOTH, expand=True, pady=4)
        self.result_text = tk.Text(rf, font=("Consolas", 10), state=tk.DISABLED)
        self.result_text.pack(fill=tk.BOTH, expand=True)

    def _read_geom(self):
        return {
            "A_flow_hot":  float(self.Ah_entry.get()),
            "A_flow_cold": float(self.Ac_entry.get()),
            "P_w_hot":     float(self.Pwh_entry.get()),
            "P_w_cold":    float(self.Pwc_entry.get()),
        }

    def run_optimize(self):
        try:
            N        = int(self.N_entry.get())
            L_min    = float(self.Lmin_entry.get())
            L_max    = float(self.Lmax_entry.get())
            tol      = float(self.tol_entry.get())
            max_iter = int(self.maxiter_entry.get())
            geom     = self._read_geom()

            self._set_text("⏳ 최적화 진행 중... (Bisection iteration)\n")
            self.root.update()

            # 진행 콜백 (실시간 업데이트)
            progress_lines = []
            def progress(it, L, T, err):
                progress_lines.append(
                    f"  iter {it:3d} | L={L:>9.4f} m | "
                    f"T_cold_out={T:>9.4f} K ({T-273.15:>7.4f}°C) | "
                    f"err={err:>+8.4f} K"
                )
                self._set_text("⏳ 최적화 진행 중...\n\n" + "\n".join(progress_lines))
                self.root.update()

            result = optimize_length(N, geom,
                                     L_min=L_min, L_max=L_max,
                                     tol=tol, max_iter=max_iter,
                                     progress_cb=progress)
            self._last_result = result
            self._show_result(result)

        except Exception as e:
            messagebox.showerror("오류", str(e))

    def save_csv(self):
        if not self._last_result:
            messagebox.showinfo("정보", "먼저 최적화를 실행하세요.")
            return
        try:
            path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv")],
                initialfile="node_data.csv"
            )
            if not path:
                return
            save_node_data(self._last_result["node_data"], path)
            messagebox.showinfo("저장", f"CSV 저장 완료:\n{path}")
        except Exception as e:
            messagebox.showerror("오류", str(e))

    def _show_result(self, result):
        c  = get_fixed_conditions()
        L  = result["L"]
        nd = result["node_data"]
        T_x0 = result["T_cold_out"]
        target = c["target"]["T_cold_out"]

        # Outer Bisection History
        hist_lines = ["  [Outer Bisection History]",
                      "  iter |   L [m]   |  T_cold_out [K]  |   err [K]   |",
                      "  ─────┼───────────┼──────────────────┼─────────────┤"]
        for it, Lm, T, d, ok in result["history"]:
            mark = "✓" if ok else " "
            hist_lines.append(
                f"   {it:3d} | {Lm:>9.4f} | {T:>10.4f}  ({T-273.15:>6.2f}°C) "
                f"| {d:>+8.4f} {mark}"
            )

        # 노드 샘플 (5점)
        N = result["N"]
        sample_idx = [0, N//4, N//2, 3*N//4, N]
        sample_lines = [
            "  [노드 데이터 샘플]",
            f"  {'node':>4} {'x[m]':>8} {'T_hot[K]':>11} "
            f"{'T_cold[K]':>11} {'P_hot[MPa]':>11} {'P_cold[MPa]':>12} "
            f"{'U[W/m²K]':>10} {'q[W]':>10}",
            "  " + "─"*86
        ]
        for idx in sample_idx:
            if idx >= len(nd):
                continue
            n = nd[idx]
            sample_lines.append(
                f"  {n['node']:>4} {n['x']:>8.4f} "
                f"{n['T_hot']:>11.4f} {n['T_cold']:>11.4f} "
                f"{n['P_hot']/1e6:>11.4f} {n['P_cold']/1e6:>12.4f} "
                f"{n['U']:>10.2f} {n['q_cell']:>10.4f}"
            )

        # 총 열전달량
        Q_total = sum(n['q_cell'] for n in nd)

        text = (
            "═══════════════════════════════════════════════════════════════\n"
            f"  [Task 4] Optimizer 결과 — Water/Water\n"
            "═══════════════════════════════════════════════════════════════\n"
            f"  ★ 최종 길이 L      = {L:.5f} m\n"
            f"  ★ 노드 수 N         = {N}\n"
            f"  ★ Cold 출구 온도    = {T_x0:.4f} K  ({T_x0-273.15:.4f} °C)\n"
            f"     목표              = {target:.4f} K  ({target-273.15:.4f} °C)\n"
            f"     오차              = {T_x0-target:+.4f} K\n"
            f"  ★ Hot  출구 온도    = {nd[-1]['T_hot']:.4f} K  "
            f"({nd[-1]['T_hot']-273.15:.4f} °C)\n"
            f"  ★ 총 열전달량 Q     = {Q_total:.2f} W  ({Q_total/1000:.2f} kW)\n"
            f"  ★ 수렴 여부         = {'✅ 수렴' if result['converged'] else '⚠️ 미수렴'}\n"
            "───────────────────────────────────────────────────────────────\n"
            + "\n".join(hist_lines) + "\n"
            "───────────────────────────────────────────────────────────────\n"
            + "\n".join(sample_lines) + "\n"
            "═══════════════════════════════════════════════════════════════\n"
        )
        self._set_text(text)

    def _set_text(self, text):
        self.result_text.configure(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.configure(state=tk.DISABLED)


# ============================================================
if __name__ == "__main__":
    # 콘솔 검증
    print("=" * 60)
    print("  [Task 4] Optimizer 검증 — Water/Water")
    print("=" * 60)

    geom_extra = {
        "A_flow_hot":  1e-3,
        "A_flow_cold": 1e-3,
        "P_w_hot":     0.628,
        "P_w_cold":    0.628,
    }

    def cb(it, L, T, err):
        print(f"  iter {it:3d} | L={L:>9.4f} m | "
              f"T_cold_out={T:>9.4f} K | err={err:>+8.4f} K")

    result = optimize_length(N=100, geom_extra=geom_extra,
                              L_min=0.1, L_max=80.0,
                              tol=0.01, max_iter=40,
                              progress_cb=cb)

    print()
    print(f"  ★ 최종 길이 L     = {result['L']:.5f} m")
    print(f"  ★ Cold 출구 온도  = {result['T_cold_out']:.4f} K "
          f"({result['T_cold_out']-273.15:.4f} °C)")
    print(f"  ★ Hot  출구 온도  = {result['node_data'][-1]['T_hot']:.4f} K "
          f"({result['node_data'][-1]['T_hot']-273.15:.4f} °C)")
    Q_total = sum(n['q_cell'] for n in result['node_data'])
    print(f"  ★ 총 열전달량 Q   = {Q_total:.2f} W ({Q_total/1000:.2f} kW)")
    print(f"  ★ 수렴 여부       = {result['converged']}")

    # GUI 실행
    root = tk.Tk()
    app = OptimizerGUI(root)
    root.mainloop()