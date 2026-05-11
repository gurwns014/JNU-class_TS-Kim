"""
[Solver.py] 비등 가능 대향류 솔버 (Strategy + q-iteration)
=============================================================
설계 원칙:
  - 솔버는 상관식별 분기 없음. Correlation.py의 어댑터(BaseBoilCorrelation)만 호출
  - q가 들어가는 식만 q-iteration. q를 안 쓰는 식은 1-pass.
  - 초기 q는 양측 단상 Dittus-Boelter로 가정 (하드코딩 없음)
  - T_wall, P_v 등 어댑터가 요구하는 값만 ctx에 채워줌

[흐름]
  각 셀에서:
    1) regime 판정 (subcooled / two_phase / superheated)
    2) 양측 단상 h_sp 계산 → U_sp → q_flux_0  ← 초기값
    3) 비등 영역이 있고 어댑터가 requires_q == True 면 q-iteration:
       반복마다 T_wall, P_v 갱신 후 어댑터.compute_h() 호출 → q_new
       under-relaxation으로 수렴 (|Δq/q| < tol)
    4) 엔탈피·압력 업데이트, 다음 셀로
"""

import math
from Data_model import get_fixed_conditions, get_state, T_from_PH
from Physics_engine import evaluate_node, friction_factor
from Correlation import get_model, DittusBoelterModel

try:
    from CoolProp.CoolProp import PropsSI
except ImportError:
    PropsSI = None


# ============================================================
# 수치 파라미터 (모듈 상수 — 외부에서 수정 가능)
# ============================================================
Q_ITER_TOL  = 1e-4    # q 상대 수렴 기준
Q_ITER_MAX  = 50      # q 최대 반복
Q_RELAX     = 0.5     # under-relaxation factor
P_MIN_GUARD = 1.0e5   # 비현실 압력 가드 [Pa]
SHOOT_TOL   = 0.5     # outer shooting 수렴 [K]
SHOOT_MAX   = 30


# ============================================================
# 포화 / regime
# ============================================================
def saturation_props(fluid, P):
    H_l   = PropsSI('H', 'P', P, 'Q', 0, fluid)
    H_g   = PropsSI('H', 'P', P, 'Q', 1, fluid)
    T_sat = PropsSI('T', 'P', P, 'Q', 0, fluid)
    sigma = PropsSI('I', 'T', T_sat, 'Q', 0, fluid)
    M     = PropsSI('M', fluid) * 1000.0
    P_crit= PropsSI('Pcrit', fluid)
    return {"H_l": H_l, "H_g": H_g, "h_fg": H_g - H_l,
            "T_sat": T_sat, "sigma": sigma, "M": M, "P_crit": P_crit}


def classify_regime(H, sat):
    if H < sat["H_l"]:
        return "subcooled", 0.0
    elif H > sat["H_g"]:
        return "superheated", 1.0
    else:
        x = (H - sat["H_l"]) / sat["h_fg"]
        return "two_phase", max(0.0, min(1.0, x))


def get_phase_state(fluid, P, H, sat):
    regime, x = classify_regime(H, sat)
    if regime in ("subcooled", "superheated"):
        T = T_from_PH(fluid, P, H)
        s = get_state(fluid, P, T)
        return dict(regime=regime, x=x, T=T, H=H,
                    rho=s["rho"], mu=s["mu"], k=s["k"], Cp=s["Cp"])
    T = sat["T_sat"]
    rho_l = PropsSI('D','P',P,'Q',0,fluid); mu_l = PropsSI('V','P',P,'Q',0,fluid)
    k_l   = PropsSI('L','P',P,'Q',0,fluid); Cp_l = PropsSI('C','P',P,'Q',0,fluid)
    rho_g = PropsSI('D','P',P,'Q',1,fluid); mu_g = PropsSI('V','P',P,'Q',1,fluid)
    k_g   = PropsSI('L','P',P,'Q',1,fluid); Cp_g = PropsSI('C','P',P,'Q',1,fluid)
    return dict(regime=regime, x=x, T=T, H=H,
                rho_l=rho_l, rho_g=rho_g, mu_l=mu_l, mu_g=mu_g,
                k_l=k_l, k_g=k_g, Cp_l=Cp_l, Cp_g=Cp_g,
                rho=rho_l, mu=mu_l, k=k_l, Cp=Cp_l)


# ============================================================
# 단상 h (Dittus-Boelter — 항상 초기값으로 사용)
# ============================================================
def single_phase_h(side_state, fluid, P, m_dot, A_flow, D_h, mode):
    """어떤 regime이든 단상(액) 가정으로 h 반환 — q-init용"""
    if side_state["regime"] in ("subcooled", "superheated"):
        node = evaluate_node(fluid, P, side_state["T"], m_dot,
                              A_flow, D_h, mode=mode)
        return dict(h=node["h"], Re=node["Re"], V=node["V"],
                    f=node["f"], Nu=node["Nu"], Pr=node["Pr"])
    s = side_state
    G    = m_dot / A_flow
    x    = max(min(s["x"], 1.0 - 1e-6), 1e-6)
    Re_l = G * (1.0 - x) * D_h / s["mu_l"]
    Pr_l = s["mu_l"] * s["Cp_l"] / s["k_l"]
    if Re_l < 2300:
        Nu = 4.36
    else:
        n = 0.4 if mode == "heating" else 0.3
        Nu = 0.023 * Re_l**0.8 * Pr_l**n
    h = Nu * s["k_l"] / D_h
    V = G / s["rho_l"]
    f = friction_factor(Re_l)
    return dict(h=h, Re=Re_l, V=V, f=f, Nu=Nu, Pr=Pr_l)


# ============================================================
# T_wall, P_v 계산 (어댑터가 요구할 때만)
# ============================================================
def estimate_T_wall(T_hot, T_cold, h_hot, h_cold, q_flux):
    """인터페이스 에너지보존: 양측 벽면 + 평균 T_wall"""
    T_wh = T_hot  - q_flux / max(h_hot,  1.0)
    T_wc = T_cold + q_flux / max(h_cold, 1.0)
    return T_wh, T_wc, 0.5 * (T_wh + T_wc)


def saturation_pressure_at(fluid, T):
    try:
        return PropsSI('P', 'T', T, 'Q', 0, fluid)
    except Exception:
        return PropsSI('Pcrit', fluid)


# ============================================================
# ctx 빌더 — 어댑터가 요구하는 것만 채움
# ============================================================
def build_ctx(side_state, fluid, P, m_dot, A_flow, D_h, sat,
               q_flux, T_wall, P_v,
               L_pipe, P_H, P_F, mode, horizontal):
    """
    어댑터에 넘길 컨텍스트 dict 생성.
    어댑터가 필요한 키만 꺼내 쓰면 됨 — 일부 빠져도 무관.
    """
    G = m_dot / A_flow
    return dict(
        G=G, q_flux=q_flux, D_h=D_h, P=P, T_wall=T_wall, P_v=P_v,
        sat=sat, L_pipe=L_pipe, P_H=P_H, P_F=P_F,
        mode=mode, horizontal=horizontal,
    )


def compute_h_for_side(side_state, model, fluid, P, m_dot, A_flow, D_h, sat,
                        ctx_base, mode):
    """
    한 측의 h 계산 — 어댑터 인터페이스만 사용.
    단상 regime이면 항상 단상 h를 직접 반환 (어댑터가 single_phase 아니면 무시).
    """
    if side_state["regime"] in ("subcooled", "superheated"):
        # 단상 영역: 비등 어댑터는 적용 X → 단상 h 그대로
        return single_phase_h(side_state, fluid, P, m_dot, A_flow, D_h, mode)

    # two-phase 영역: 어댑터 호출
    ctx = dict(ctx_base)
    ctx["mode"] = mode
    h = model.compute_h(side_state, ctx)

    # 부수 정보 (Re, V, f) — 압력강하용 단상 액 기준
    G = m_dot / A_flow
    s = side_state
    x = max(min(s["x"], 1.0 - 1e-6), 1e-6)
    Re_l = G * (1.0 - x) * D_h / s["mu_l"]
    V    = G / s["rho_l"]
    f    = friction_factor(Re_l)
    return dict(h=h, Re=Re_l, V=V, f=f,
                Nu=h * D_h / s["k_l"],
                Pr=s["mu_l"] * s["Cp_l"] / s["k_l"])


# ============================================================
# 셀 단위 q-iteration (핵심)
# ============================================================
def cell_solve_q(hp, cp, hot_state, cold_state,
                  hot_sat, cold_sat, geom, dx,
                  model, L_pipe, P_H, P_F):
    """
    한 셀에서 q를 단상으로 초기화 → 비등 영역이면 q-iteration.
    'model'은 BaseBoilCorrelation 인스턴스 (어댑터).
    """
    T_hot, T_cold = hp["T"], cp["T"]
    dT     = T_hot - T_cold
    P_w_avg = 0.5 * (geom["P_w_hot"] + geom["P_w_cold"])
    dA      = P_w_avg * dx
    R_wall  = geom["t_wall"] / geom["k_wall"]

    # ── Step 1: q 초기 가정 ──
    h_hot_sp  = single_phase_h(hp, hot_state["fluid"],
                                hot_state["P"], hot_state["m_dot"],
                                geom["A_flow_hot"], geom["D_h"],
                                mode="cooling")
    h_cold_sp = single_phase_h(cp, cold_state["fluid"],
                                cold_state["P"], cold_state["m_dot"],
                                geom["A_flow_cold"], geom["D_h"],
                                mode="heating")
    U_sp     = 1.0 / (1.0/h_hot_sp["h"] + R_wall + 1.0/h_cold_sp["h"])
    q_flux_0 = U_sp * dT

    hot_2p  = (hp["regime"] == "two_phase")
    cold_2p = (cp["regime"] == "two_phase")

    # 단상-단상이면 iteration 불필요
    if not (hot_2p or cold_2p):
        return dict(
            h_hot=h_hot_sp["h"], h_cold=h_cold_sp["h"],
            U=U_sp, q_flux=q_flux_0, q_cell=q_flux_0*dA, dA=dA,
            Re_hot=h_hot_sp["Re"], Re_cold=h_cold_sp["Re"],
            V_hot=h_hot_sp["V"],  V_cold=h_cold_sp["V"],
            f_hot=h_hot_sp["f"],  f_cold=h_cold_sp["f"],
            n_iter=0, q_converged=True, T_wall=None,
        )

    # ── Step 2: q-iteration ──
    # 어댑터가 q를 요구하지 않으면 1-pass로 끝낼 수 있지만,
    # T_wall이나 P_v가 필요한 경우(Chen, Zhang)는 어쨌든 한 번은 갱신 필요.
    # 안전하게 통일된 루프로 처리.
    needs_loop = (model.requires_q or model.requires_Twall or model.requires_Pv)

    q_flux  = q_flux_0
    last_q  = q_flux
    h_hot_info, h_cold_info = h_hot_sp, h_cold_sp
    converged = False
    n_iter    = 0
    T_wall_avg = None

    max_it = Q_ITER_MAX if needs_loop else 1

    for it in range(1, max_it + 1):
        n_iter = it

        # T_wall (필요할 때만 계산하지만, 한 번 계산해서 ctx에 넣어줘도 무방)
        _, _, T_wall_avg = estimate_T_wall(
            T_hot, T_cold, h_hot_info["h"], h_cold_info["h"], q_flux
        )

        # P_v 양측 fluid별로 (필요한 경우만)
        if model.requires_Pv:
            P_v_hot  = saturation_pressure_at(hot_state["fluid"],  T_wall_avg)
            P_v_cold = saturation_pressure_at(cold_state["fluid"], T_wall_avg)
        else:
            P_v_hot = P_v_cold = None

        # 양측 ctx 빌드
        ctx_hot = build_ctx(hp, hot_state["fluid"], hot_state["P"],
                             hot_state["m_dot"], geom["A_flow_hot"], geom["D_h"],
                             hot_sat, q_flux, T_wall_avg, P_v_hot,
                             L_pipe, P_H, P_F, "cooling", False)
        ctx_cold = build_ctx(cp, cold_state["fluid"], cold_state["P"],
                              cold_state["m_dot"], geom["A_flow_cold"], geom["D_h"],
                              cold_sat, q_flux, T_wall_avg, P_v_cold,
                              L_pipe, P_H, P_F, "heating", False)

        h_hot_info  = compute_h_for_side(hp, model, hot_state["fluid"],
                                          hot_state["P"], hot_state["m_dot"],
                                          geom["A_flow_hot"], geom["D_h"],
                                          hot_sat, ctx_hot, mode="cooling")
        h_cold_info = compute_h_for_side(cp, model, cold_state["fluid"],
                                          cold_state["P"], cold_state["m_dot"],
                                          geom["A_flow_cold"], geom["D_h"],
                                          cold_sat, ctx_cold, mode="heating")

        U_new = 1.0 / (1.0/h_hot_info["h"] + R_wall + 1.0/h_cold_info["h"])
        q_new = U_new * dT

        if needs_loop:
            rel = abs(q_new - last_q) / max(abs(last_q), 1.0)
            if rel < Q_ITER_TOL:
                q_flux = q_new
                converged = True
                break
            # under-relaxation
            q_flux = Q_RELAX * q_new + (1.0 - Q_RELAX) * last_q
            last_q = q_flux
        else:
            q_flux = q_new
            converged = True
            break

    U_final = 1.0 / (1.0/h_hot_info["h"] + R_wall + 1.0/h_cold_info["h"])

    return dict(
        h_hot=h_hot_info["h"], h_cold=h_cold_info["h"],
        U=U_final, q_flux=q_flux, q_cell=q_flux*dA, dA=dA,
        Re_hot=h_hot_info["Re"], Re_cold=h_cold_info["Re"],
        V_hot=h_hot_info["V"],  V_cold=h_cold_info["V"],
        f_hot=h_hot_info["f"],  f_cold=h_cold_info["f"],
        n_iter=n_iter, q_converged=converged, T_wall=T_wall_avg,
    )


# ============================================================
# 셀 진행 (엔탈피 + 압력)
# ============================================================
def advance_cell(hot_state, cold_state, hot_sat, cold_sat, geom, dx,
                  model, L_pipe, P_H, P_F):
    hp = get_phase_state(hot_state["fluid"],  hot_state["P"],
                          hot_state["H"], hot_sat)
    cp = get_phase_state(cold_state["fluid"], cold_state["P"],
                          cold_state["H"], cold_sat)

    res = cell_solve_q(hp, cp, hot_state, cold_state,
                        hot_sat, cold_sat, geom, dx,
                        model, L_pipe, P_H, P_F)
    q_cell = res["q_cell"]

    H_hot_new  = hot_state["H"]  - q_cell / hot_state["m_dot"]
    H_cold_new = cold_state["H"] - q_cell / cold_state["m_dot"]

    dP_hot  = (res["f_hot"]  * (dx/geom["D_h"])
               * hp["rho"]   * res["V_hot"]**2 / 2.0)
    dP_cold = (res["f_cold"] * (dx/geom["D_h"])
               * cp["rho"]   * res["V_cold"]**2 / 2.0)
    P_hot_new  = hot_state["P"]  - dP_hot
    P_cold_new = cold_state["P"] + dP_cold

    if P_hot_new < P_MIN_GUARD or P_cold_new < P_MIN_GUARD:
        raise ValueError(
            f"⚠️ 비현실 압력 (P_hot={P_hot_new/1e6:.3f}, "
            f"P_cold={P_cold_new/1e6:.3f} MPa). A_flow / dx 조정 필요."
        )

    hot_next  = dict(hot_state);  hot_next["P"]  = P_hot_new;  hot_next["H"] = H_hot_new
    cold_next = dict(cold_state); cold_next["P"] = P_cold_new; cold_next["H"] = H_cold_new
    hp_n = get_phase_state(hot_next["fluid"], P_hot_new, H_hot_new, hot_sat)
    cp_n = get_phase_state(cold_next["fluid"], P_cold_new, H_cold_new, cold_sat)
    hot_next["T"] = hp_n["T"]; hot_next["x"] = hp_n["x"]
    cold_next["T"]= cp_n["T"]; cold_next["x"]= cp_n["x"]

    info = {
        "T_hot": hp["T"], "T_cold": cp["T"],
        "x_hot": hp["x"], "x_cold": cp["x"],
        "regime_hot": hp["regime"], "regime_cold": cp["regime"],
        "h_hot": res["h_hot"], "h_cold": res["h_cold"],
        "Re_hot": res["Re_hot"], "Re_cold": res["Re_cold"],
        "U": res["U"], "q_cell": q_cell, "q_flux": res["q_flux"],
        "dA": res["dA"], "dT": hp["T"] - cp["T"],
        "dP_hot": dP_hot, "dP_cold": dP_cold,
        "n_iter_q": res["n_iter"], "q_converged": res["q_converged"],
        "T_wall": res["T_wall"],
    }
    return hot_next, cold_next, info


# ============================================================
# 병류(Co-current) 솔버 — 모두 같은 방향 (참고용)
# ============================================================
def solve_parallel(L, N, geom_extra, boil_corr="chen",
                   P_H=None, P_F=None, verbose=False):
    """
    병류(Co-current) 단순 forward-march.
      x=0: Hot, Cold 모두 입구 (같은 쪽에서 들어옴)
      x=L: Hot, Cold 모두 출구
    """
    return _direct_march(L, N, geom_extra, boil_corr,
                          P_H, P_F, verbose, mode="parallel")


# ============================================================
# 대향류(Counter-current) 직접 march 솔버 ★ 슬라이드 그림
# ============================================================
def solve_counter_direct(L, N, geom_extra, boil_corr="chen",
                          P_H=None, P_F=None, verbose=False):
    """
    슬라이드 그림 그대로:
      Hot:  x=0 (입구) ──→ x=L (출구)
      Cold: x=L (입구) ←── x=0 (출구)

    Forward-march 방식 — outer shooting 없음.
    Cold 입구는 x=L 인덱스 N에서 시작하여 i를 감소시키며 진행.

    셀 i (위치 x_i ~ x_{i+1}) 에서:
      Hot  은 i → i+1 방향 (정방향)
      Cold 는 i+1 → i 방향 (역방향)

    한 셀 안에서 Hot 입구 상태와 Cold 입구 상태가 같이 만나므로
    위치 i 와 i+1 의 사이값(평균)으로 ΔT, q를 계산하는 것이 이상적이지만
    여기서는 단순화: 셀 입구쪽 상태(Hot:i, Cold:i+1)로 평가.
    """
    return _direct_march(L, N, geom_extra, boil_corr,
                          P_H, P_F, verbose, mode="counter")


def _direct_march(L, N, geom_extra, boil_corr, P_H, P_F, verbose, mode):
    """
    parallel : Hot, Cold 둘 다 좌→우  →  i=0..N-1 순방향
    counter  : Hot 좌→우, Cold 우→좌  →  Cold 흐름 따라 i=N-1..0 역방향

    슬라이드 그림 방향:
        x=0 ─────────────────── x=L
        Hot 입구 ──→ ──→ ──→ Hot 출구
        Cold 출구 ←── ←── ←── Cold 입구

    counter 모드에서는:
      Cold 입구 상태가 노드 N에 주어짐
      셀 i 계산 시: Cold 입구 = 노드 i+1, Cold 출구 = 노드 i
      따라서 i를 N-1부터 0으로 거꾸로 진행해야 Cold가 연속됨

    Hot은 i=0이 입구 → 보통이면 i=0부터 정방향이지만,
    counter 모드에서는 Hot도 같이 셀별로 진행해야 하므로
    'Cold 흐름 방향 따라 i를 N-1 → 0'으로 진행하면서
    셀 i 안에서 Hot의 입구는 i, Cold의 입구는 i+1로 정의.
    → Hot 출구 상태는 셀 처리 후 노드 i+1에 저장.

    ※ 하지만 Hot은 입구 = 노드 0이라는 경계조건이 있고,
       counter 모드에서 cold 방향으로 거꾸로 진행하면
       Hot의 노드 0 상태가 미리 알려져 있지 않음.
       → 이 경우엔 'Hot 출구 온도'를 추정해서 outer shooting 해야 함.

    따라서 두 방식 모두 제공:
       counter : Hot 입구 = 노드 0 기준, Cold 입구는 노드 N 기준,
                 i = 0..N-1 정방향 진행 (Hot은 정방향, Cold도 셀별로 i+1 입구→i 출구)
                 → Cold의 노드 N(입구) 상태는 알려져 있으나
                   셀 0 처리 시 Cold 입구 = 노드 1 상태가 필요
                 → Cold도 거꾸로 풀어야 하므로,
                   첫 패스에서는 Cold 입구를 모든 노드에 fc[T_in]으로 초기화하고
                   여러 번 iteration (Picard반복)
    """
    c  = get_fixed_conditions()
    fh = c["hot_inlet"];  fc = c["cold_inlet"]
    hot_sat  = saturation_props(fh["fluid"], fh["P_in"])
    cold_sat = saturation_props(fc["fluid"], fc["P_in"])

    geom = {
        "A_flow_hot":  geom_extra["A_flow_hot"],
        "A_flow_cold": geom_extra["A_flow_cold"],
        "P_w_hot":     geom_extra["P_w_hot"],
        "P_w_cold":    geom_extra["P_w_cold"],
        "D_h":         c["geometry"]["D_h"],
        "t_wall":      c["geometry"]["t_wall"],
        "k_wall":      c["geometry"]["k_wall"],
    }
    if P_H is None: P_H = math.pi * geom["D_h"]
    if P_F is None: P_F = math.pi * geom["D_h"]

    model = get_model(boil_corr)
    if verbose:
        print(f"  [Model] {model.name} ({mode} flow)")

    dx = L / N
    H_hot_in  = PropsSI('H', 'T', fh["T_in"], 'P', fh["P_in"], fh["fluid"])
    H_cold_in = PropsSI('H', 'T', fc["T_in"], 'P', fc["P_in"], fc["fluid"])

    # 노드 배열
    T_hot   = [None]*(N+1)
    T_cold  = [None]*(N+1)
    P_hot   = [None]*(N+1)
    P_cold  = [None]*(N+1)
    H_hot   = [None]*(N+1)
    H_cold  = [None]*(N+1)
    x_hot   = [0.0]*(N+1)
    x_cold  = [0.0]*(N+1)
    rg_hot  = ["init"]*(N+1)
    rg_cold = ["init"]*(N+1)

    # ── 경계 조건 설정 ──
    T_hot[0]  = fh["T_in"];  P_hot[0]  = fh["P_in"];  H_hot[0]  = H_hot_in

    if mode == "parallel":
        # Cold도 노드 0에서 시작
        T_cold[0] = fc["T_in"]; P_cold[0] = fc["P_in"]; H_cold[0] = H_cold_in
    else:  # counter
        # Cold 입구는 노드 N
        T_cold[N] = fc["T_in"]; P_cold[N] = fc["P_in"]; H_cold[N] = H_cold_in
        # ★ Cold 흐름 방향(N → 0)을 따라 거꾸로 셀을 풀기 위해
        #   초기 추정으로 모든 Cold 노드를 입구 상태로 채움 (Picard 1차 추정)
        for k in range(N):
            T_cold[k] = fc["T_in"]
            P_cold[k] = fc["P_in"]
            H_cold[k] = H_cold_in

    cell_info = [None]*N

    # ── Picard 반복 (counter 모드에서만 의미) ──
    # parallel은 1-pass로 충분, counter는 양방향 결합이므로 여러 번 통과
    n_picard = 1 if mode == "parallel" else 5

    for picard_it in range(n_picard):
        # 한 번의 풀이 패스
        if mode == "parallel":
            cell_order = range(N)            # 0 → N-1
        else:
            # Cold 흐름 따라가기 위해 N-1 → 0 (셀 i = 노드 i와 i+1 사이)
            cell_order = range(N-1, -1, -1)

        for i in cell_order:
            # 이 셀에서 Hot 입구 = 노드 i, Cold 입구 = 노드 (parallel: i, counter: i+1)
            hot_in = {"fluid": fh["fluid"], "m_dot": fh["m_dot"],
                      "P": P_hot[i], "H": H_hot[i],
                      "T": T_hot[i], "x": x_hot[i]}
            ci_in = i if mode == "parallel" else (i+1)
            cold_in = {"fluid": fc["fluid"], "m_dot": fc["m_dot"],
                       "P": P_cold[ci_in], "H": H_cold[ci_in],
                       "T": T_cold[ci_in], "x": x_cold[ci_in]}

            # 모든 입력이 채워졌는지 확인 (counter 초기 패스에서 일부 None일 수 있음)
            if (hot_in["T"] is None or cold_in["T"] is None
                or hot_in["P"] is None or cold_in["P"] is None):
                continue

            try:
                hp = get_phase_state(hot_in["fluid"],  hot_in["P"],  hot_in["H"],  hot_sat)
                cp = get_phase_state(cold_in["fluid"], cold_in["P"], cold_in["H"], cold_sat)
                res = cell_solve_q(hp, cp, hot_in, cold_in,
                                    hot_sat, cold_sat, geom, dx,
                                    model, L, P_H, P_F)
            except Exception as e:
                if verbose:
                    print(f"  [picard {picard_it}, cell {i}] {e}")
                continue

            q_cell = res["q_cell"]

            # Hot 업데이트 (정방향, 식음)
            H_hot[i+1] = H_hot[i] - q_cell / fh["m_dot"]
            dP_h       = res["f_hot"] * (dx/geom["D_h"]) * hp["rho"] * res["V_hot"]**2 / 2.0
            P_hot[i+1] = P_hot[i] - dP_h

            # Cold 업데이트 (mode별 방향)
            if mode == "parallel":
                H_cold[i+1] = H_cold[i] + q_cell / fc["m_dot"]
                dP_c        = res["f_cold"] * (dx/geom["D_h"]) * cp["rho"] * res["V_cold"]**2 / 2.0
                P_cold[i+1] = P_cold[i] - dP_c
                ci_out = i+1
            else:  # counter — Cold는 i+1(입구) → i(출구), 가열됨
                H_cold[i] = H_cold[i+1] + q_cell / fc["m_dot"]
                dP_c      = res["f_cold"] * (dx/geom["D_h"]) * cp["rho"] * res["V_cold"]**2 / 2.0
                P_cold[i] = P_cold[i+1] - dP_c
                ci_out = i

            if P_hot[i+1] is not None and P_hot[i+1] < P_MIN_GUARD:
                if verbose: print(f"  [stop] P_hot 비현실 @ cell {i}")
                break

            # 새 상태
            hp_n = get_phase_state(fh["fluid"], P_hot[i+1], H_hot[i+1], hot_sat)
            T_hot[i+1]  = hp_n["T"]
            x_hot[i+1]  = hp_n["x"]
            rg_hot[i+1] = hp_n["regime"]

            cp_n = get_phase_state(fc["fluid"], P_cold[ci_out], H_cold[ci_out], cold_sat)
            T_cold[ci_out]  = cp_n["T"]
            x_cold[ci_out]  = cp_n["x"]
            rg_cold[ci_out] = cp_n["regime"]

            cell_info[i] = {
                "h_hot": res["h_hot"], "h_cold": res["h_cold"],
                "Re_hot": res["Re_hot"], "Re_cold": res["Re_cold"],
                "U": res["U"], "q_cell": q_cell, "q_flux": res["q_flux"],
                "dT": hp["T"] - cp["T"],
                "n_iter_q": res["n_iter"], "q_converged": res["q_converged"],
                "T_wall": res["T_wall"],
            }

        if verbose and mode == "counter":
            Tout_cold = T_cold[0]
            print(f"  [picard {picard_it+1}/{n_picard}] T_cold(out,x=0) = {Tout_cold:.3f} K")

    # 빈 셀 처리
    for i in range(N):
        if cell_info[i] is None:
            cell_info[i] = {"h_hot":0,"h_cold":0,"Re_hot":0,"Re_cold":0,
                            "U":0,"q_cell":0,"q_flux":0,"dT":0,
                            "n_iter_q":0,"q_converged":True,"T_wall":None}

    # 노드 데이터 빌드
    node_data = []
    for i in range(N+1):
        ci = cell_info[i-1] if i > 0 else None
        node_data.append({
            "node": i, "x_pos": i*dx,
            "T_hot":  T_hot[i]  if T_hot[i]  is not None else 0.0,
            "T_cold": T_cold[i] if T_cold[i] is not None else 0.0,
            "P_hot":  P_hot[i]  if P_hot[i]  is not None else 0.0,
            "P_cold": P_cold[i] if P_cold[i] is not None else 0.0,
            "H_hot":  H_hot[i]  if H_hot[i]  is not None else 0.0,
            "H_cold": H_cold[i] if H_cold[i] is not None else 0.0,
            "x_hot":  x_hot[i],  "x_cold": x_cold[i],
            "regime_hot":  rg_hot[i],  "regime_cold": rg_cold[i],
            "U":      ci["U"]      if ci else 0.0,
            "h_hot":  ci["h_hot"]  if ci else 0.0,
            "h_cold": ci["h_cold"] if ci else 0.0,
            "Re_hot": ci["Re_hot"] if ci else 0.0,
            "Re_cold":ci["Re_cold"]if ci else 0.0,
            "q_cell": ci["q_cell"] if ci else 0.0,
            "q_flux": ci["q_flux"] if ci else 0.0,
            "dT":     ci["dT"]     if ci else 0.0,
            "n_iter_q":    ci["n_iter_q"]    if ci else 0,
            "q_converged": ci["q_converged"] if ci else True,
            "T_wall":      ci["T_wall"]      if ci else None,
        })

    return dict(L=L, N=N, model=model.name, mode=mode,
                 node_data=node_data, converged=True,
                 T_cold_out=(node_data[0]["T_cold"] if mode=="counter"
                              else node_data[-1]["T_cold"]),
                 T_hot_out=node_data[-1]["T_hot"])


# ============================================================
# 메인 솔버 (대향류 + outer shooting — 옵션)
# ============================================================
def solve_counter_current(L, N, geom_extra, boil_corr="chen",
                           P_H=None, P_F=None,
                           T_cold_x0_guess=None,
                           shoot_tol=SHOOT_TOL, shoot_max=SHOOT_MAX,
                           verbose=False):
    """
    L         : 채널 길이 [m]
    N         : 노드 수
    geom_extra: dict(A_flow_hot, A_flow_cold, P_w_hot, P_w_cold)
    boil_corr : 비등 상관식 이름 (Correlation.py의 어댑터 이름)
    """
    c  = get_fixed_conditions()
    fh = c["hot_inlet"];  fc = c["cold_inlet"]
    hot_sat  = saturation_props(fh["fluid"], fh["P_in"])
    cold_sat = saturation_props(fc["fluid"], fc["P_in"])

    geom = {
        "A_flow_hot":  geom_extra["A_flow_hot"],
        "A_flow_cold": geom_extra["A_flow_cold"],
        "P_w_hot":     geom_extra["P_w_hot"],
        "P_w_cold":    geom_extra["P_w_cold"],
        "D_h":         c["geometry"]["D_h"],
        "t_wall":      c["geometry"]["t_wall"],
        "k_wall":      c["geometry"]["k_wall"],
    }
    if P_H is None: P_H = math.pi * geom["D_h"]
    if P_F is None: P_F = math.pi * geom["D_h"]

    # 어댑터 인스턴스 한 번만 생성 (모든 셀에서 재사용)
    model = get_model(boil_corr)
    if verbose:
        print(f"  [Model] {model.name}  requires: "
              f"q={model.requires_q}, Twall={model.requires_Twall}, "
              f"Pv={model.requires_Pv}, M={model.requires_M}, "
              f"geom={model.requires_geom}")

    dx = L / N
    H_hot_in  = PropsSI('H', 'T', fh["T_in"], 'P', fh["P_in"], fh["fluid"])
    H_cold_in = PropsSI('H', 'T', fc["T_in"], 'P', fc["P_in"], fc["fluid"])

    # Outer shooting
    T_lo = fc["T_in"] + 0.01
    T_hi = fh["T_in"] - 0.01
    history = []
    best = None

    for sh_it in range(shoot_max):
        T_guess = T_cold_x0_guess if (sh_it == 0 and T_cold_x0_guess) \
                  else 0.5*(T_lo + T_hi)

        hot = {"fluid": fh["fluid"], "m_dot": fh["m_dot"],
               "P": fh["P_in"], "H": H_hot_in,
               "T": fh["T_in"], "x": 0.0}
        H_cold_guess = PropsSI('H', 'T', T_guess, 'P', fc["P_in"], fc["fluid"])
        cold = {"fluid": fc["fluid"], "m_dot": fc["m_dot"],
                "P": fc["P_in"], "H": H_cold_guess,
                "T": T_guess, "x": 0.0}

        node_data = [{
            "node": 0, "x_pos": 0.0,
            "T_hot": hot["T"], "T_cold": cold["T"],
            "P_hot": hot["P"], "P_cold": cold["P"],
            "H_hot": hot["H"], "H_cold": cold["H"],
            "x_hot": 0.0, "x_cold": 0.0,
            "regime_hot": "init", "regime_cold": "init",
            "U": 0.0, "h_hot": 0.0, "h_cold": 0.0,
            "Re_hot": 0.0, "Re_cold": 0.0,
            "q_cell": 0.0, "q_flux": 0.0,
            "dT": hot["T"]-cold["T"],
            "n_iter_q": 0, "q_converged": True, "T_wall": None,
        }]

        diverged = False
        for i in range(N):
            try:
                hot_n, cold_n, info = advance_cell(
                    hot, cold, hot_sat, cold_sat, geom, dx,
                    model, L_pipe=L, P_H=P_H, P_F=P_F
                )
            except ValueError:
                diverged = True
                break

            hot, cold = hot_n, cold_n
            node_data.append({
                "node": i+1, "x_pos": (i+1)*dx,
                "T_hot": hot["T"], "T_cold": cold["T"],
                "P_hot": hot["P"], "P_cold": cold["P"],
                "H_hot": hot["H"], "H_cold": cold["H"],
                "x_hot": hot["x"], "x_cold": cold["x"],
                "regime_hot": info["regime_hot"], "regime_cold": info["regime_cold"],
                "U": info["U"], "h_hot": info["h_hot"], "h_cold": info["h_cold"],
                "Re_hot": info["Re_hot"], "Re_cold": info["Re_cold"],
                "q_cell": info["q_cell"], "q_flux": info["q_flux"],
                "dT": info["dT"],
                "n_iter_q": info["n_iter_q"],
                "q_converged": info["q_converged"],
                "T_wall": info["T_wall"],
            })

        if diverged:
            T_hi = T_guess
            history.append((sh_it+1, T_guess, None, "diverged"))
            continue

        T_cold_xL = cold["T"]
        diff = T_cold_xL - fc["T_in"]
        history.append((sh_it+1, T_guess, T_cold_xL, diff))
        best = (T_guess, node_data, diff)

        if verbose:
            print(f"  shoot {sh_it+1:2d} | T_guess={T_guess:.3f} | "
                  f"T_cold(x=L)={T_cold_xL:.3f} | err={diff:+.3f}")

        if abs(diff) < shoot_tol:
            return dict(L=L, N=N, model=model.name, node_data=node_data,
                         converged=True, shoot_history=history,
                         T_cold_out=node_data[0]["T_cold"],
                         T_hot_out=node_data[-1]["T_hot"])

        if diff > 0:
            T_hi = T_guess
        else:
            T_lo = T_guess

    T_guess, node_data, diff = best
    return dict(L=L, N=N, model=model.name, node_data=node_data,
                 converged=False, shoot_history=history,
                 T_cold_out=node_data[0]["T_cold"],
                 T_hot_out=node_data[-1]["T_hot"])


# ============================================================
# 자가 검증
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  [Solver.py] 어댑터 + q-iteration 자가 검증")
    print("=" * 70)

    # 비등 유도를 위해 Cold 유량을 일시 감소
    from Data_model import FIXED_CONDITIONS
    original_m = FIXED_CONDITIONS["cold_inlet"]["m_dot"]
    FIXED_CONDITIONS["cold_inlet"]["m_dot"] = 0.5

    geom_extra = {"A_flow_hot": 1e-3, "A_flow_cold": 1e-3,
                  "P_w_hot": 0.628, "P_w_cold": 0.628}

    for corr in ["chen", "shah", "gungor_winterton",
                  "bertsch", "kim_mudawar", "zhang",
                  "jens_lottes", "yu"]:
        print(f"\n[{corr}]  L=3.0 m, N=60")
        try:
            result = solve_counter_current(3.0, 60, geom_extra,
                                            boil_corr=corr,
                                            shoot_tol=1.0, shoot_max=15,
                                            verbose=False)
            nd = result["node_data"]
            from collections import Counter
            rc = Counter(n["regime_cold"] for n in nd)
            x_max = max(n["x_cold"] for n in nd)

            iters = [n["n_iter_q"] for n in nd if n["n_iter_q"] > 0]
            avg_it = sum(iters)/len(iters) if iters else 0
            max_it = max(iters) if iters else 0
            all_conv = all(n["q_converged"] for n in nd)

            Q = sum(n["q_cell"] for n in nd)
            print(f"  Cold T_out={result['T_cold_out']:.2f} K   "
                  f"max x={x_max:.4f}   regimes={dict(rc)}")
            print(f"  q-iter(평균/최대)={avg_it:.1f}/{max_it}   "
                  f"q-수렴={all_conv}   Q={Q/1000:.1f} kW   "
                  f"outer={result['converged']}")
        except Exception as e:
            print(f"  ❌ {e}")

    FIXED_CONDITIONS["cold_inlet"]["m_dot"] = original_m
