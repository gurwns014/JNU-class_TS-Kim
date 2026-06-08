"""
[Task 5] Main.py
- 시뮬레이션 실행 (Optimizer 함수 호출)
- 데이터 저장 (CSV)
- 데이터 로드 및 시각화 (matplotlib)
- GUI로 통합 제어

[변경된 조건 — 양쪽 모두 Water]
  Hot  (Water):  P = 15.0 MPa,  T_in = 600 K
  Cold (Water):  P =  6.0 MPa,  T_in = 530 K
"""

import os
import csv
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk
)

# 한글 폰트 설정 (윈도우/맥/리눅스 자동 처리)
import platform
_sys = platform.system()
if _sys == "Windows":
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
elif _sys == "Darwin":
    matplotlib.rcParams["font.family"] = "AppleGothic"
else:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

from Data_model import get_fixed_conditions
from Optimizer import optimize_length, save_node_data


# ============================================================
# 시뮬레이션 실행 (함수 호출 = Optimizer.optimize_length)
# ============================================================
def run_simulation(N, geom_extra,
                   L_min=0.1, L_max=80.0,
                   tol=0.01, max_iter=40,
                   progress_cb=None):
    """
    Optimizer 호출 → 결과 반환
    """
    return optimize_length(
        N=N, geom_extra=geom_extra,
        L_min=L_min, L_max=L_max,
        tol=tol, max_iter=max_iter,
        progress_cb=progress_cb
    )


# ============================================================
# CSV 저장/로드
# ============================================================
def save_results_csv(result, path):
    """노드별 데이터 + 메타정보 CSV 저장"""
    save_node_data(result["node_data"], path)


def load_node_data_csv(path):
    """CSV 로드 → node_data list 반환"""
    node_data = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            n = {}
            for k, v in row.items():
                if v == "" or v is None:
                    n[k] = 0.0
                else:
                    try:
                        n[k] = float(v)
                    except ValueError:
                        n[k] = v
            node_data.append(n)
    return node_data


# ============================================================
# 시각화 (matplotlib)
# ============================================================
def plot_results(node_data, fig=None, title_suffix=""):
    """
    4-패널 그래프:
      ① 온도 분포 (T_hot, T_cold vs x)
      ② 압력 분포 (P_hot, P_cold vs x)
      ③ U, h 분포 vs x
      ④ 노드별 q_cell vs x
    """
    if fig is None:
        fig = plt.figure(figsize=(13, 9))
    fig.clear()

    x       = [n["x"]       for n in node_data]
    T_hot   = [n["T_hot"]   for n in node_data]
    T_cold  = [n["T_cold"]  for n in node_data]
    P_hot   = [n["P_hot"]/1e6 for n in node_data]
    P_cold  = [n["P_cold"]/1e6 for n in node_data]
    U_arr   = [n["U"]       for n in node_data]
    h_hot   = [n["h_hot"]   for n in node_data]
    h_cold  = [n["h_cold"]  for n in node_data]
    q_cell  = [n["q_cell"]  for n in node_data]

    # ① 온도
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(x, T_hot,  "r-",  marker="o", ms=3, label="T_hot (Water)")
    ax1.plot(x, T_cold, "b-",  marker="s", ms=3, label="T_cold (Water)")
    ax1.fill_between(x, T_cold, T_hot, alpha=0.10, color="orange",
                     label="ΔT (구동력)")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("Temperature [K]")
    ax1.set_title(f"① 온도 분포{title_suffix}")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(alpha=0.3)

    # ② 압력
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(x, P_hot,  "r-", marker="o", ms=3, label="P_hot")
    ax2.plot(x, P_cold, "b-", marker="s", ms=3, label="P_cold")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("Pressure [MPa]")
    ax2.set_title("② 압력 분포 (Darcy-Weisbach)")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(alpha=0.3)

    # ③ U, h
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(x[1:], U_arr[1:],   "g-",  marker="o", ms=3, label="U (총괄)")
    ax3.plot(x[1:], h_hot[1:],   "r--", marker="^", ms=3, label="h_hot")
    ax3.plot(x[1:], h_cold[1:],  "b--", marker="v", ms=3, label="h_cold")
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("Heat transfer coeff. [W/m²K]")
    ax3.set_title("③ 열전달계수 분포")
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.set_yscale("log")

    # ④ 노드별 열전달량
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.bar(x[1:], q_cell[1:], width=(x[1]-x[0])*0.8 if len(x)>1 else 0.01,
            color="orange", alpha=0.75, edgecolor="darkorange",
            label="q_cell")
    ax4.set_xlabel("x [m]")
    ax4.set_ylabel("q_cell [W]")
    ax4.set_title(f"④ 노드별 열전달량 (Σ = {sum(q_cell):.1f} W)")
    ax4.legend(loc="best", fontsize=9)
    ax4.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    return fig


# ============================================================
# GUI
# ============================================================
class MainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("[Task 5] Main — VHTR HX 시뮬레이션 (Water/Water)")
        self.root.geometry("1400x900")
        self._last_result = None
        self._loaded_data = None
        self._build()

    def _build(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        # 좌측: 컨트롤 패널
        left = ttk.Frame(main, width=380)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left.pack_propagate(False)

        # 우측: 그래프 + 결과
        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── 좌측: 입력 ─────────────────────────────
        cond = ttk.LabelFrame(left, text="고정조건 (참고)", padding=6)
        cond.pack(fill=tk.X, pady=2)
        c = get_fixed_conditions()
        cond_text = (
            f"  Hot  : Water @ {c['hot_inlet']['P_in']/1e6:.1f} MPa, "
            f"{c['hot_inlet']['T_in']} K\n"
            f"         m_dot = {c['hot_inlet']['m_dot']} kg/s\n"
            f"  Cold : Water @ {c['cold_inlet']['P_in']/1e6:.1f} MPa, "
            f"{c['cold_inlet']['T_in']} K\n"
            f"         m_dot = {c['cold_inlet']['m_dot']} kg/s\n"
            f"  Target T_cold_out = {c['target']['T_cold_out']} K\n"
            f"  D_h={c['geometry']['D_h']*1000} mm  "
            f"t_w={c['geometry']['t_wall']*1000} mm  "
            f"k_w={c['geometry']['k_wall']} W/mK"
        )
        tk.Label(cond, text=cond_text, justify=tk.LEFT,
                 font=("Consolas", 9)).pack(anchor="w")

        # 입력 파라미터
        params = ttk.LabelFrame(left, text="시뮬레이션 입력", padding=6)
        params.pack(fill=tk.X, pady=2)

        self.entries = {}
        items = [
            ("N",       "노드 수 N",          "100"),
            ("L_min",   "L 최소 [m]",         "0.1"),
            ("L_max",   "L 최대 [m]",         "80.0"),
            ("tol",     "수렴 tol [K]",       "0.01"),
            ("maxiter", "최대 반복",          "40"),
            ("Ah",      "A_flow_hot [m²]",    "1e-3"),
            ("Ac",      "A_flow_cold [m²]",   "1e-3"),
            ("Pwh",     "P_w_hot [m]",        "0.628"),
            ("Pwc",     "P_w_cold [m]",       "0.628"),
        ]
        for key, label, default in items:
            row = ttk.Frame(params); row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=label, width=18).pack(side=tk.LEFT)
            e = ttk.Entry(row, width=14); e.insert(0, default)
            e.pack(side=tk.LEFT, padx=2)
            self.entries[key] = e

        # 버튼
        btn = ttk.LabelFrame(left, text="실행", padding=6)
        btn.pack(fill=tk.X, pady=2)

        ttk.Button(btn, text="🚀 1. 시뮬레이션 실행",
                   command=self.run_sim).pack(fill=tk.X, pady=2)
        ttk.Button(btn, text="💾 2. CSV 저장",
                   command=self.save_csv).pack(fill=tk.X, pady=2)
        ttk.Button(btn, text="📂 3. CSV 로드 + 시각화",
                   command=self.load_csv).pack(fill=tk.X, pady=2)
        ttk.Button(btn, text="📊 그래프 다시 그리기 (현재 결과)",
                   command=self.replot).pack(fill=tk.X, pady=2)
        ttk.Button(btn, text="💿 그래프 PNG 저장",
                   command=self.save_png).pack(fill=tk.X, pady=2)

        # 진행 로그
        log_frame = ttk.LabelFrame(left, text="진행 로그", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        self.log_text = tk.Text(log_frame, font=("Consolas", 9),
                                height=12, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # ── 우측: 그래프 + 요약 ────────────────────
        # 상단: 결과 요약
        summary = ttk.LabelFrame(right, text="결과 요약", padding=6)
        summary.pack(fill=tk.X, pady=2)
        self.summary_text = tk.Text(summary, font=("Consolas", 10),
                                    height=6, state=tk.DISABLED)
        self.summary_text.pack(fill=tk.X)

        # 하단: 그래프
        plot_frame = ttk.LabelFrame(right, text="시각화", padding=6)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=2)

        self.fig = Figure(figsize=(11, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

    def _read_inputs(self):
        e = self.entries
        return {
            "N":        int(e["N"].get()),
            "L_min":    float(e["L_min"].get()),
            "L_max":    float(e["L_max"].get()),
            "tol":      float(e["tol"].get()),
            "max_iter": int(e["maxiter"].get()),
            "geom":     {
                "A_flow_hot":  float(e["Ah"].get()),
                "A_flow_cold": float(e["Ac"].get()),
                "P_w_hot":     float(e["Pwh"].get()),
                "P_w_cold":    float(e["Pwc"].get()),
            }
        }

    def _log(self, msg, clear=False):
        self.log_text.configure(state=tk.NORMAL)
        if clear:
            self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
        self.root.update()

    def _set_summary(self, text):
        self.summary_text.configure(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, text)
        self.summary_text.configure(state=tk.DISABLED)

    # ── 1) 시뮬레이션 실행 ─────────────────────
    def run_sim(self):
        try:
            inp = self._read_inputs()
            self._log("=" * 50, clear=True)
            self._log(f"  시뮬레이션 시작 (N={inp['N']})")
            self._log("=" * 50)

            def cb(it, L, T, err):
                self._log(f"  iter {it:3d} | L={L:>9.4f} m | "
                          f"T_co={T:>8.3f} K | err={err:>+7.4f} K")

            result = run_simulation(
                N=inp["N"], geom_extra=inp["geom"],
                L_min=inp["L_min"], L_max=inp["L_max"],
                tol=inp["tol"], max_iter=inp["max_iter"],
                progress_cb=cb
            )

            self._last_result = result
            self._log("=" * 50)
            self._log(f"  ✅ 수렴: {result['converged']}")
            self._log(f"  ★ L = {result['L']:.5f} m")

            self._update_summary(result)
            plot_results(result["node_data"], fig=self.fig,
                         title_suffix=f"  (L = {result['L']:.4f} m)")
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("오류", str(e))
            self._log(f"  ❌ 오류: {e}")

    # ── 2) CSV 저장 ───────────────────────────
    def save_csv(self):
        if not self._last_result:
            messagebox.showinfo("정보", "먼저 시뮬레이션을 실행하세요.")
            return
        try:
            path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv")],
                initialfile="simulation_result.csv"
            )
            if not path:
                return
            save_results_csv(self._last_result, path)
            self._log(f"  💾 CSV 저장: {os.path.basename(path)}")
            messagebox.showinfo("저장 완료", f"CSV 저장 완료:\n{path}")
        except Exception as e:
            messagebox.showerror("오류", str(e))

    # ── 3) CSV 로드 + 시각화 ──────────────────
    def load_csv(self):
        try:
            path = filedialog.askopenfilename(
                filetypes=[("CSV", "*.csv"), ("All", "*.*")]
            )
            if not path:
                return
            node_data = load_node_data_csv(path)
            self._loaded_data = node_data
            self._log(f"  📂 CSV 로드: {os.path.basename(path)} "
                      f"({len(node_data)} 노드)")

            # 요약 갱신
            x_max  = node_data[-1]["x"]
            T_co   = node_data[0]["T_cold"]   # x=0 = Cold 출구
            T_ho   = node_data[-1]["T_hot"]   # x=L = Hot 출구
            Q_tot  = sum(n["q_cell"] for n in node_data)

            text = (
                f"  ★ [로드된 데이터]\n"
                f"  ★ 길이 L      = {x_max:.5f} m  (노드 {len(node_data)}개)\n"
                f"  ★ Cold 출구   = {T_co:.4f} K ({T_co-273.15:.2f}°C)\n"
                f"  ★ Hot  출구   = {T_ho:.4f} K ({T_ho-273.15:.2f}°C)\n"
                f"  ★ 총 열전달량 = {Q_tot:.2f} W ({Q_tot/1000:.2f} kW)\n"
            )
            self._set_summary(text)

            plot_results(node_data, fig=self.fig,
                         title_suffix=f"  (L = {x_max:.4f} m, 로드)")
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("오류", str(e))

    # ── 그래프 다시 그리기 ────────────────────
    def replot(self):
        if self._last_result:
            plot_results(self._last_result["node_data"], fig=self.fig,
                         title_suffix=f"  (L = {self._last_result['L']:.4f} m)")
            self.canvas.draw()
        elif self._loaded_data:
            x_max = self._loaded_data[-1]["x"]
            plot_results(self._loaded_data, fig=self.fig,
                         title_suffix=f"  (L = {x_max:.4f} m, 로드)")
            self.canvas.draw()
        else:
            messagebox.showinfo("정보", "먼저 시뮬레이션 실행 또는 CSV 로드.")

    # ── PNG 저장 ──────────────────────────────
    def save_png(self):
        try:
            path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")],
                initialfile="result_plot.png"
            )
            if not path:
                return
            self.fig.savefig(path, dpi=150, bbox_inches="tight")
            self._log(f"  💿 그래프 저장: {os.path.basename(path)}")
            messagebox.showinfo("저장 완료", f"그래프 저장 완료:\n{path}")
        except Exception as e:
            messagebox.showerror("오류", str(e))

    # ── 결과 요약 갱신 ────────────────────────
    def _update_summary(self, result):
        c = get_fixed_conditions()
        L  = result["L"]
        nd = result["node_data"]
        T_co = result["T_cold_out"]
        T_ho = nd[-1]["T_hot"]
        target = c["target"]["T_cold_out"]
        Q_tot = sum(n["q_cell"] for n in nd)

        text = (
            f"  ★ 최종 길이 L     = {L:.5f} m       "
            f"(노드 {result['N']}개, dx = {L/result['N']*1000:.3f} mm)\n"
            f"  ★ Cold 출구 온도  = {T_co:.4f} K ({T_co-273.15:.2f}°C)   "
            f"목표 = {target:.2f} K   오차 = {T_co-target:+.4f} K\n"
            f"  ★ Hot  출구 온도  = {T_ho:.4f} K ({T_ho-273.15:.2f}°C)\n"
            f"  ★ 총 열전달량 Q   = {Q_tot:.2f} W ({Q_tot/1000:.2f} kW)\n"
            f"  ★ 수렴 여부       = "
            f"{'✅ 수렴' if result['converged'] else '⚠️ 미수렴'}\n"
        )
        self._set_summary(text)


# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  [Task 5] Main — VHTR HX 시뮬레이션 (Water/Water)")
    print("=" * 60)

    root = tk.Tk()
    app = MainGUI(root)
    root.mainloop()