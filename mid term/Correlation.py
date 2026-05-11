"""
[Boiling Correlations] Correlation.py
=====================================
비등 열전달 상관식 모음
- 각 상관식은 통일된 인터페이스: htp(...) → h_tp [W/m²K] 반환
- Solver에서 쉽게 import해서 사용 가능
- 입력: 액/기 물성치 (CoolProp으로 미리 얻음), 유동조건 (G, x, q'', D_h)

[참고 단위]
    G    : 질량플럭스 [kg/m²s]
    x    : quality [-] (0=포화액, 1=포화증기)
    q''  : 벽 열속 [W/m²]
    h_fg : 증발잠열 [J/kg]
    σ    : 표면장력 [N/m]
    M    : 분자량 [kg/kmol]
    P_r  : reduced pressure = P / P_crit
    Pr   : Prandtl 수 (혼동주의)

[포함 상관식]
    1. Dittus-Boelter (단상, 참고용)
    2. Chen (1966)         — Dengler & Addoms / Bennett 변형
    3. Shah (1982)
    4. Gungor & Winterton (1986)
    5. Bertsch et al. (2009)
    6. Kim & Mudawar (2014)
    7. Zhang (2005)
    8. Jens & Lottes (1951)   — ΔT_sat 형태
    9. Tran et al. (1996) + Yu et al. (2002)
"""

import math


# ============================================================
# 헬퍼 함수
# ============================================================
def _safe_x(x):
    """x를 (1e-6, 1-1e-6) 사이로 clip — 0 또는 1에서 발산 방지"""
    return max(1e-6, min(1.0 - 1e-6, x))


def martinelli_Xtt(x, rho_l, rho_g, mu_l, mu_g):
    """
    Lockhart-Martinelli 파라미터 X_tt
        X_tt = ((1-x)/x)^0.9 · (ρ_g/ρ_l)^0.5 · (μ_l/μ_g)^0.1
    """
    x = _safe_x(x)
    return ((1.0-x)/x)**0.9 * (rho_g/rho_l)**0.5 * (mu_l/mu_g)**0.1


def reynolds_l(G, x, D_h, mu_l):
    """액상 Re: Re_l = G(1-x)·D_h/μ_l"""
    return G * (1.0 - _safe_x(x)) * D_h / mu_l


def reynolds_g(G, x, D_h, mu_g):
    """기상 Re: Re_g = G·x·D_h/μ_g"""
    return G * _safe_x(x) * D_h / mu_g


def reynolds_fo(G, D_h, mu_l):
    """전체 액 flow Re_fo (x=0 가정): Re_fo = G·D_h/μ_l"""
    return G * D_h / mu_l


def reynolds_go(G, D_h, mu_g):
    """전체 기 flow Re_go (x=1 가정)"""
    return G * D_h / mu_g


def prandtl(Cp, mu, k):
    """Pr = μ·Cp/k"""
    return mu * Cp / k


def boiling_number(q_flux, G, h_fg):
    """Bo = q'' / (G·h_fg)"""
    return q_flux / (G * h_fg)


def weber_l(G, D_h, rho_l, sigma):
    """We_l = G²·D / (ρ_l·σ)"""
    return G**2 * D_h / (rho_l * sigma)


def froude_l(G, rho_l, D_h, g=9.81):
    """Fr_l = G² / (ρ_l² · g · D_h)"""
    return G**2 / (rho_l**2 * g * D_h)


def convection_number(x, rho_l, rho_g):
    """Co = ((1-x)/x)^0.8 · (ρ_g/ρ_l)^0.5  (Shah)"""
    x = _safe_x(x)
    return ((1.0-x)/x)**0.8 * (rho_g/rho_l)**0.5


# ============================================================
# 1. Dittus-Boelter (단상, 참고용)
# ============================================================
def dittus_boelter(Re, Pr, k, D_h, mode="heating"):
    """
    Dittus-Boelter (난류, 단상)
      Nu = 0.023 · Re^0.8 · Pr^0.4   (heating)
      Nu = 0.023 · Re^0.8 · Pr^0.3   (cooling)
    """
    n = 0.4 if mode == "heating" else 0.3
    if Re < 2300:
        Nu = 4.36
    else:
        Nu = 0.023 * Re**0.8 * Pr**n
    return Nu * k / D_h


# ============================================================
# 2. Chen correlation (1966)
# ============================================================
def chen_1966(x, G, q_flux, D_h, T_w, T_sat,
              rho_l, rho_g, mu_l, mu_g, k_l, Cp_l,
              h_fg, sigma, P_l, P_v):
    """
    Chen (1966):
        h_tp = h_mic + h_mac
        h_mac = F · h_DB,l  (Dittus-Boelter on liquid, F는 Reynolds factor)
        h_mic = Forster-Zuber pool boiling × S (suppression factor)

    입력:
        T_w, T_sat : 벽/포화 온도 [K]
        P_l, P_v   : 액상/기상 포화압력 [Pa] (Forster-Zuber에서 ΔP 필요)
    반환: h_tp [W/m²K]
    """
    x = _safe_x(x)

    # ── h_mac: Dittus-Boelter (liquid only) × F ──
    Re_l = reynolds_l(G, x, D_h, mu_l)
    Pr_l = prandtl(Cp_l, mu_l, k_l)
    h_DB = 0.023 * Re_l**0.8 * Pr_l**0.4 * k_l / D_h  # liquid only

    # Reynolds factor F (X_tt 기반)
    Xtt = martinelli_Xtt(x, rho_l, rho_g, mu_l, mu_g)
    if 1.0/Xtt <= 0.1:
        F = 1.0
    else:
        F = 2.35 * (1.0/Xtt + 0.213)**0.736

    h_mac = F * h_DB

    # ── h_mic: Forster-Zuber × S ──
    Re_tp = Re_l * F**1.25
    S = 1.0 / (1.0 + 2.53e-6 * Re_tp**1.17)

    dT_sat = max(T_w - T_sat, 1e-6)
    dP_sat = max(P_v - P_l, 1.0)  # Pa

    # Forster-Zuber pool boiling
    # h_pb = 0.00122 · (k_l^0.79 · Cp_l^0.45 · ρ_l^0.49 / (σ^0.5 · μ_l^0.29 · h_fg^0.24 · ρ_v^0.24))
    #        × ΔT_sat^0.24 · ΔP_sat^0.75
    num = k_l**0.79 * Cp_l**0.45 * rho_l**0.49
    den = sigma**0.5 * mu_l**0.29 * h_fg**0.24 * rho_g**0.24
    h_FZ = 0.00122 * (num/den) * dT_sat**0.24 * dP_sat**0.75

    h_mic = S * h_FZ
    return h_mac + h_mic


# ============================================================
# 3. Shah correlation (1982)
# ============================================================
def shah_1982(x, G, q_flux, D_h,
              rho_l, rho_g, mu_l, k_l, Cp_l,
              h_fg, horizontal=False):
    """
    Shah (1982):
        h_tp = ψ · h_sp     (h_sp: Dittus-Boelter, liquid only, x=0 가정)

    ψ는 N(Co or Fr 보정), Bo에 따라 ψ_nb, ψ_cb, ψ_bs 중 max
    """
    x = _safe_x(x)
    Re_l = reynolds_l(G, x, D_h, mu_l)
    Pr_l = prandtl(Cp_l, mu_l, k_l)
    h_sp = 0.023 * Re_l**0.8 * Pr_l**0.4 * k_l / D_h

    Co = convection_number(x, rho_l, rho_g)
    Bo = boiling_number(q_flux, G, h_fg)

    if horizontal:
        Fr = froude_l(G, rho_l, D_h)
        N = Co if Fr >= 0.04 else 0.38 * Fr**(-0.3) * Co
    else:
        N = Co

    # ψ_cb: convective boiling
    psi_cb = 1.8 / N**0.8

    # ψ_nb: nucleate boiling
    if N > 1.0:
        if Bo > 0.3e-4:
            psi_nb = 230.0 * Bo**0.5
        else:
            psi_nb = 1.0 + 46.0 * Bo**0.5
    else:
        psi_nb = 0.0

    # ψ_bs: bubble suppression regime
    if 0.1 < N <= 1.0:
        F = 14.7 if Bo >= 11e-4 else 15.43
        psi_bs = F * Bo**0.5 * math.exp(2.74 * N**(-0.1))
    elif N <= 0.1:
        F = 14.7 if Bo >= 11e-4 else 15.43
        psi_bs = F * Bo**0.5 * math.exp(2.74 * N**(-0.15))
    else:
        psi_bs = 0.0

    psi = max(psi_nb, psi_cb, psi_bs)
    return psi * h_sp


# ============================================================
# 4. Gungor & Winterton (1986)
# ============================================================
def gungor_winterton_1986(x, G, q_flux, D_h,
                           rho_l, rho_g, mu_l, mu_g, k_l, Cp_l,
                           h_fg, P, P_crit, M,
                           horizontal=False):
    """
    Gungor & Winterton (1986):
        h_tp = E · h_sp + S · h_pool

    h_pool : Cooper pool boiling correlation
    E      : enhancement factor (Bo, X_tt 기반)
    S      : suppression factor (E, Re_l 기반)
    수평관 + Fr<0.05 → E_2, S_2 보정 추가
    """
    x = _safe_x(x)
    Re_l = reynolds_l(G, x, D_h, mu_l)
    Pr_l = prandtl(Cp_l, mu_l, k_l)
    h_sp = 0.023 * Re_l**0.8 * Pr_l**0.4 * k_l / D_h

    Xtt = martinelli_Xtt(x, rho_l, rho_g, mu_l, mu_g)
    Bo  = boiling_number(q_flux, G, h_fg)

    # Cooper pool boiling
    P_r = P / P_crit
    h_pool = (55.0 * P_r**0.12
              * (-math.log10(P_r))**(-0.55)
              * M**(-0.5)
              * q_flux**0.67)

    E = 1.0 + 24000.0 * Bo**1.16 + 1.37 * (1.0/Xtt)**0.86
    S = 1.0 / (1.0 + 1.15e-6 * E**2 * Re_l**1.17)

    # 수평관 보정
    if horizontal:
        Fr = froude_l(G, rho_l, D_h)
        if Fr < 0.05:
            E2 = Fr**(0.1 - 2.0*Fr)
            S2 = math.sqrt(Fr)
            E *= E2
            S *= S2

    return E * h_sp + S * h_pool


# ============================================================
# 5. Bertsch et al. (2009)
# ============================================================
def bertsch_2009(x, G, q_flux, D_h, L,
                  rho_l, rho_g, mu_l, mu_g, k_l, k_g, Cp_l, Cp_g,
                  sigma, h_fg, P, P_crit, M, g=9.81):
    """
    Bertsch (2009):
        h_tp = E · h_cb + S · h_nb

    h_cb : convective boiling (Hausen 상관식 활용, two-phase 가중)
    h_nb : Cooper pool boiling
    E    : x²-x⁶ + exp(-0.6·Co) 기반
    S    : 1 - x
    """
    x = _safe_x(x)

    # Hausen correlation for single-phase (liquid only, gas only)
    Re_fo = reynolds_fo(G, D_h, mu_l)
    Re_go = reynolds_go(G, D_h, mu_g)
    Pr_l  = prandtl(Cp_l, mu_l, k_l)
    Pr_g  = prandtl(Cp_g, mu_g, k_g)

    DhL = D_h / max(L, D_h)  # L 매우 작을 때 보호

    Nu_fo = 3.66 + (0.0668 * DhL * Re_fo * Pr_l) / (1.0 + 0.04*(DhL*Re_fo*Pr_l)**(2.0/3.0))
    Nu_go = 3.66 + (0.0668 * DhL * Re_go * Pr_g) / (1.0 + 0.04*(DhL*Re_go*Pr_g)**(2.0/3.0))

    h_sp_lo = Nu_fo * k_l / D_h
    h_sp_go = Nu_go * k_g / D_h

    h_cb = h_sp_lo * (1.0 - x) + h_sp_go * x

    # Cooper pool boiling
    P_r = P / P_crit
    h_nb = (55.0 * P_r**0.12
            * (-math.log10(P_r))**(-0.55)
            * M**(-0.5)
            * q_flux**0.67)

    # Confinement number
    Co_conf = math.sqrt(sigma / (g * (rho_l - rho_g) * D_h**2))

    E = 1.0 + 80.0 * (x**2 - x**6) * math.exp(-0.6 * Co_conf)
    S = 1.0 - x

    return E * h_cb + S * h_nb


# ============================================================
# 6. Kim & Mudawar (2014)
# ============================================================
def kim_mudawar_2014(x, G, q_flux, D_h,
                      rho_l, rho_g, mu_l, mu_g, k_l, Cp_l,
                      sigma, h_fg, P, P_crit,
                      P_H, P_F):
    """
    Kim & Mudawar (2014):
        h_tp = (h_nb² + h_cb²)^0.5

    h_nb : nucleate boiling 항
    h_cb : convective boiling 항
    P_H, P_F : heated / wetted perimeter [m]
    """
    x = _safe_x(x)
    P_R = P / P_crit
    Bo  = boiling_number(q_flux, G, h_fg)
    We_fo = weber_l(G, D_h, rho_l, sigma)
    Xtt   = martinelli_Xtt(x, rho_l, rho_g, mu_l, mu_g)

    # Dittus-Boelter base (liquid only, x=0 가정)
    Re_fo = reynolds_fo(G, D_h, mu_l)
    Pr_l  = prandtl(Cp_l, mu_l, k_l)
    h_DB  = 0.023 * Re_fo**0.8 * Pr_l**0.4 * k_l / D_h

    PHF = P_H / P_F  # 가열면적/젖음둘레 비

    h_cb = (2345.0 * (Bo * PHF)**0.7 * P_R**0.38 * (1.0 - x)**(-0.51)) * h_DB

    h_nb = (5.2 * (Bo * PHF)**0.08 * We_fo**(-0.54)
            + 3.5 * (1.0/Xtt)**0.94 * (rho_g/rho_l)**0.25) * h_DB

    return math.sqrt(h_nb**2 + h_cb**2)


# ============================================================
# 7. Zhang correlation (2005)
# ============================================================
def zhang_2005(x, G, q_flux, D_h, T_w, T_sat,
                rho_l, rho_g, mu_l, mu_g, k_l, Cp_l,
                h_fg, sigma, P_l, P_v):
    """
    Zhang (2005):
        h = h_pb + ξ · φ_f · h_sp,v

    h_pb       : Forster-Zuber pool boiling
    φ_f        : √(1 + C/X + 1/X²)  (X: Martinelli)
    h_sp,v     : 4.36·k_l/D_h (laminar) or Dittus-Boelter (turbulent) — vapor or liquid 기반
    ξ = 0.64
    """
    x = _safe_x(x)
    Xtt = martinelli_Xtt(x, rho_l, rho_g, mu_l, mu_g)

    # Chisholm C 파라미터 (Lockhart-Martinelli 일반값: turbulent-turbulent = 20)
    C = 20.0
    phi_f = math.sqrt(1.0 + C/Xtt + 1.0/Xtt**2)

    # 단상 (liquid-only) h
    Re_l = reynolds_l(G, x, D_h, mu_l)
    if Re_l < 2300:
        h_sp = 4.36 * k_l / D_h
    else:
        Pr_l = prandtl(Cp_l, mu_l, k_l)
        h_sp = 0.023 * Re_l**0.8 * Pr_l**0.4 * k_l / D_h

    # Forster-Zuber
    dT_sat = max(T_w - T_sat, 1e-6)
    dP_sat = max(P_v - P_l, 1.0)
    num = k_l**0.79 * Cp_l**0.45 * rho_l**0.49
    den = sigma**0.5 * mu_l**0.29 * h_fg**0.24 * rho_g**0.24
    h_pb = 0.00122 * (num/den) * dT_sat**0.24 * dP_sat**0.75

    xi = 0.64
    return h_pb + xi * phi_f * h_sp


# ============================================================
# 8. Jens & Lottes (1951)
# ============================================================
def jens_lottes_1951(q_flux, P):
    """
    Jens & Lottes (1951) — ANL:
        ΔT_sat = 0.7925 · (q'')^(1/4) · exp(-P / 6.2e6)
        (SI 단위: ΔT [K], q'' [W/m²], P [Pa])

    반환: ΔT_sat [K]

    ※ 다른 상관식과 달리 h가 아닌 벽 과열도 ΔT를 직접 줌
       → h_nb = q'' / ΔT_sat 로 변환 가능
    ※ 슬라이드의 "P/28728" 은 lb/ft² 단위로 본 P/600 (psi에서 환산된 값)인데,
      SI 변환 시 6.2 MPa (= 6.2e6 Pa)로 쓰는 것이 일반적이고 정확함.
    """
    return 0.7925 * q_flux**0.25 * math.exp(-P / 6.2e6)


def jens_lottes_h(q_flux, P):
    """Jens-Lottes에서 h_nb로 변환: h = q'' / ΔT_sat"""
    dT = jens_lottes_1951(q_flux, P)
    return q_flux / max(dT, 1e-6)


# ============================================================
# 9. Tran et al. (1996) / Yu et al. (2002)
# ============================================================
def tran_1996(G, q_flux, D_h, rho_l, rho_g, sigma, h_fg):
    """
    Tran et al. (1996) — ANL:
        h_tp = 8.4e-5 · (Bo²·We_l)^0.3 · (ρ_l/ρ_v)^(-0.4)

    ※ 슬라이드의 계수 8.4e-5는 원논문 단위계 그대로 따른 값.
      SI(W/m²K)로 환산 시 계수가 매우 작게 나옴 → 원논문(Int. J. Multiphase
      Flow, 1996) 직접 참조 권장. 결과 값의 크기 확인 필수.
    """
    Bo  = boiling_number(q_flux, G, h_fg)
    We  = weber_l(G, D_h, rho_l, sigma)
    return 8.4e-5 * (Bo**2 * We)**0.3 * (rho_l/rho_g)**(-0.4)


def yu_2002(G, q_flux, D_h, rho_l, rho_g, sigma, h_fg):
    """
    Yu et al. (2002) — Tran et al. 따름:
        h_tp = 6.4e6 · (Bo²·We_l)^0.27 · (ρ_l/ρ_v)^(-0.2)
    """
    Bo  = boiling_number(q_flux, G, h_fg)
    We  = weber_l(G, D_h, rho_l, sigma)
    return 6.4e6 * (Bo**2 * We)**0.27 * (rho_l/rho_g)**(-0.2)


# ============================================================
# Solver용 통합 디스패처
# ============================================================
CORRELATION_LIST = [
    "dittus_boelter", "chen", "shah", "gungor_winterton",
    "bertsch", "kim_mudawar", "zhang",
    "jens_lottes", "tran", "yu",
]


def get_h_tp(name, **kwargs):
    """
    Solver에서 사용할 통합 인터페이스.
    예: get_h_tp("chen", x=0.2, G=..., q_flux=..., ...)
    """
    name = name.lower().replace("-", "_").replace(" ", "_")
    if name in ("dittus_boelter", "db"):
        return dittus_boelter(**kwargs)
    if name in ("chen", "chen_1966"):
        return chen_1966(**kwargs)
    if name in ("shah", "shah_1982"):
        return shah_1982(**kwargs)
    if name in ("gungor_winterton", "gw", "gungor"):
        return gungor_winterton_1986(**kwargs)
    if name in ("bertsch", "bertsch_2009"):
        return bertsch_2009(**kwargs)
    if name in ("kim_mudawar", "km", "kim"):
        return kim_mudawar_2014(**kwargs)
    if name in ("zhang", "zhang_2005"):
        return zhang_2005(**kwargs)
    if name in ("jens_lottes", "jl"):
        return jens_lottes_h(**kwargs)
    if name in ("tran", "tran_1996"):
        return tran_1996(**kwargs)
    if name in ("yu", "yu_2002"):
        return yu_2002(**kwargs)
    raise ValueError(f"Unknown correlation: {name}. "
                     f"Available: {CORRELATION_LIST}")


# ════════════════════════════════════════════════════════════
#                    ADAPTER CLASSES
# ════════════════════════════════════════════════════════════
# 솔버가 통일된 인터페이스로 호출할 수 있도록 각 상관식을
# 어댑터 클래스로 감쌈.
#
# [공통 규약]
#   클래스 속성으로 자기가 필요로 하는 것 선언:
#       requires_q     : q_flux가 식에 들어가는가? (→ q-iteration 필요)
#       requires_Twall : T_wall이 필요한가?
#       requires_Pv    : P_v = P_sat(T_wall)이 필요한가?
#       requires_M     : 분자량 필요?
#       requires_geom  : 추가 기하 정보 (P_H, P_F, L_pipe 등 키)
#       single_phase   : 단상도 처리할 수 있는가?
#
#   인스턴스 메소드:
#       compute_h(self, state, ctx) -> float
#           state: 유체 물성 dict
#                  (regime, x, T, rho_l, rho_g, mu_l, mu_g, k_l, Cp_l, ...)
#           ctx  : 솔버가 채워주는 부가 정보 dict
#                  (G, q_flux, T_wall, P_v, D_h, P, sat, P_H, P_F, L_pipe, ...)
#
# [새 식 추가 절차]
#   1) 위쪽에 순수 수식 함수 추가
#   2) 여기 어댑터 클래스 추가
#   3) REGISTRY에 한 줄 등록
#   → 솔버는 절대 건드리지 않음
# ════════════════════════════════════════════════════════════

class BaseBoilCorrelation:
    """모든 어댑터의 부모 — 공통 인터페이스 정의"""
    name           = "base"
    requires_q     = False
    requires_Twall = False
    requires_Pv    = False
    requires_M     = False
    requires_geom  = ()       # ("P_H","P_F","L_pipe",...) 등
    single_phase   = False    # 단상 영역도 직접 처리?

    def compute_h(self, state, ctx):
        raise NotImplementedError(f"{self.name}.compute_h 미구현")

    def __repr__(self):
        return f"<{self.name}>"


# ──────────────────────────────────────────────────────────────
# 0. 단상 Dittus-Boelter (참고/fallback)
# ──────────────────────────────────────────────────────────────
class DittusBoelterModel(BaseBoilCorrelation):
    name         = "dittus_boelter"
    single_phase = True   # 이 어댑터는 단상 전용

    def compute_h(self, state, ctx):
        # 단상이면 액 또는 기 물성 단일 사용
        rho = state.get("rho", state.get("rho_l"))
        mu  = state.get("mu",  state.get("mu_l"))
        k   = state.get("k",   state.get("k_l"))
        Cp  = state.get("Cp",  state.get("Cp_l"))
        G   = ctx["G"]
        D_h = ctx["D_h"]
        Re  = G * D_h / mu
        Pr  = prandtl(Cp, mu, k)
        mode = ctx.get("mode", "heating")
        return dittus_boelter(Re, Pr, k, D_h, mode=mode)


# ──────────────────────────────────────────────────────────────
# 1. Chen (1966)
# ──────────────────────────────────────────────────────────────
class ChenModel(BaseBoilCorrelation):
    name           = "chen"
    requires_q     = True
    requires_Twall = True
    requires_Pv    = True

    def compute_h(self, state, ctx):
        sat = ctx["sat"]
        return chen_1966(
            x=state["x"], G=ctx["G"], q_flux=ctx["q_flux"], D_h=ctx["D_h"],
            T_w=ctx["T_wall"], T_sat=sat["T_sat"],
            rho_l=state["rho_l"], rho_g=state["rho_g"],
            mu_l=state["mu_l"],   mu_g=state["mu_g"],
            k_l=state["k_l"],     Cp_l=state["Cp_l"],
            h_fg=sat["h_fg"], sigma=sat["sigma"],
            P_l=ctx["P"], P_v=ctx["P_v"],
        )


# ──────────────────────────────────────────────────────────────
# 2. Shah (1982)
# ──────────────────────────────────────────────────────────────
class ShahModel(BaseBoilCorrelation):
    name       = "shah"
    requires_q = True   # Bo = q''/(G·h_fg)

    def compute_h(self, state, ctx):
        sat = ctx["sat"]
        return shah_1982(
            x=state["x"], G=ctx["G"], q_flux=ctx["q_flux"], D_h=ctx["D_h"],
            rho_l=state["rho_l"], rho_g=state["rho_g"],
            mu_l=state["mu_l"], k_l=state["k_l"], Cp_l=state["Cp_l"],
            h_fg=sat["h_fg"],
            horizontal=ctx.get("horizontal", False),
        )


# ──────────────────────────────────────────────────────────────
# 3. Gungor-Winterton (1986)
# ──────────────────────────────────────────────────────────────
class GungorWintertonModel(BaseBoilCorrelation):
    name       = "gungor_winterton"
    requires_q = True
    requires_M = True

    def compute_h(self, state, ctx):
        sat = ctx["sat"]
        return gungor_winterton_1986(
            x=state["x"], G=ctx["G"], q_flux=ctx["q_flux"], D_h=ctx["D_h"],
            rho_l=state["rho_l"], rho_g=state["rho_g"],
            mu_l=state["mu_l"], mu_g=state["mu_g"],
            k_l=state["k_l"], Cp_l=state["Cp_l"],
            h_fg=sat["h_fg"],
            P=ctx["P"], P_crit=sat["P_crit"], M=sat["M"],
            horizontal=ctx.get("horizontal", False),
        )


# ──────────────────────────────────────────────────────────────
# 4. Bertsch (2009)
# ──────────────────────────────────────────────────────────────
class BertschModel(BaseBoilCorrelation):
    name          = "bertsch"
    requires_q    = True
    requires_M    = True
    requires_geom = ("L_pipe",)

    def compute_h(self, state, ctx):
        sat = ctx["sat"]
        return bertsch_2009(
            x=state["x"], G=ctx["G"], q_flux=ctx["q_flux"],
            D_h=ctx["D_h"], L=ctx["L_pipe"],
            rho_l=state["rho_l"], rho_g=state["rho_g"],
            mu_l=state["mu_l"], mu_g=state["mu_g"],
            k_l=state["k_l"], k_g=state["k_g"],
            Cp_l=state["Cp_l"], Cp_g=state["Cp_g"],
            sigma=sat["sigma"], h_fg=sat["h_fg"],
            P=ctx["P"], P_crit=sat["P_crit"], M=sat["M"],
        )


# ──────────────────────────────────────────────────────────────
# 5. Kim-Mudawar (2014)
# ──────────────────────────────────────────────────────────────
class KimMudawarModel(BaseBoilCorrelation):
    name          = "kim_mudawar"
    requires_q    = True
    requires_geom = ("P_H", "P_F")   # heated/wetted perimeter

    def compute_h(self, state, ctx):
        sat = ctx["sat"]
        return kim_mudawar_2014(
            x=state["x"], G=ctx["G"], q_flux=ctx["q_flux"], D_h=ctx["D_h"],
            rho_l=state["rho_l"], rho_g=state["rho_g"],
            mu_l=state["mu_l"], mu_g=state["mu_g"],
            k_l=state["k_l"], Cp_l=state["Cp_l"],
            sigma=sat["sigma"], h_fg=sat["h_fg"],
            P=ctx["P"], P_crit=sat["P_crit"],
            P_H=ctx["P_H"], P_F=ctx["P_F"],
        )


# ──────────────────────────────────────────────────────────────
# 6. Zhang (2005)
# ──────────────────────────────────────────────────────────────
class ZhangModel(BaseBoilCorrelation):
    name           = "zhang"
    requires_q     = True
    requires_Twall = True
    requires_Pv    = True

    def compute_h(self, state, ctx):
        sat = ctx["sat"]
        return zhang_2005(
            x=state["x"], G=ctx["G"], q_flux=ctx["q_flux"], D_h=ctx["D_h"],
            T_w=ctx["T_wall"], T_sat=sat["T_sat"],
            rho_l=state["rho_l"], rho_g=state["rho_g"],
            mu_l=state["mu_l"], mu_g=state["mu_g"],
            k_l=state["k_l"], Cp_l=state["Cp_l"],
            h_fg=sat["h_fg"], sigma=sat["sigma"],
            P_l=ctx["P"], P_v=ctx["P_v"],
        )


# ──────────────────────────────────────────────────────────────
# 7. Jens-Lottes (1951)
# ──────────────────────────────────────────────────────────────
class JensLottesModel(BaseBoilCorrelation):
    name       = "jens_lottes"
    requires_q = True

    def compute_h(self, state, ctx):
        return jens_lottes_h(q_flux=ctx["q_flux"], P=ctx["P"])


# ──────────────────────────────────────────────────────────────
# 8. Tran (1996)
# ──────────────────────────────────────────────────────────────
class TranModel(BaseBoilCorrelation):
    name       = "tran"
    requires_q = True

    def compute_h(self, state, ctx):
        sat = ctx["sat"]
        return tran_1996(
            G=ctx["G"], q_flux=ctx["q_flux"], D_h=ctx["D_h"],
            rho_l=state["rho_l"], rho_g=state["rho_g"],
            sigma=sat["sigma"], h_fg=sat["h_fg"],
        )


# ──────────────────────────────────────────────────────────────
# 9. Yu (2002)
# ──────────────────────────────────────────────────────────────
class YuModel(BaseBoilCorrelation):
    name       = "yu"
    requires_q = True

    def compute_h(self, state, ctx):
        sat = ctx["sat"]
        return yu_2002(
            G=ctx["G"], q_flux=ctx["q_flux"], D_h=ctx["D_h"],
            rho_l=state["rho_l"], rho_g=state["rho_g"],
            sigma=sat["sigma"], h_fg=sat["h_fg"],
        )


# ============================================================
# 레지스트리 — 솔버가 이름으로 찾을 때 사용
# ============================================================
_REGISTRY = {
    "dittus_boelter":   DittusBoelterModel,
    "db":               DittusBoelterModel,
    "chen":             ChenModel,
    "chen_1966":        ChenModel,
    "shah":             ShahModel,
    "shah_1982":        ShahModel,
    "gungor_winterton": GungorWintertonModel,
    "gw":               GungorWintertonModel,
    "gungor":           GungorWintertonModel,
    "bertsch":          BertschModel,
    "bertsch_2009":     BertschModel,
    "kim_mudawar":      KimMudawarModel,
    "kim":              KimMudawarModel,
    "km":               KimMudawarModel,
    "zhang":            ZhangModel,
    "zhang_2005":       ZhangModel,
    "jens_lottes":      JensLottesModel,
    "jl":               JensLottesModel,
    "tran":             TranModel,
    "tran_1996":        TranModel,
    "yu":               YuModel,
    "yu_2002":          YuModel,
}


def get_model(name):
    """
    이름으로 어댑터 인스턴스 반환.
    예: get_model("chen")  → ChenModel()
    """
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown correlation '{name}'. "
            f"Available: {sorted(set(m.name for m in _REGISTRY.values()))}"
        )
    return _REGISTRY[key]()


def list_models():
    """사용 가능한 (이름, 클래스) 리스트 반환"""
    seen = set()
    out = []
    for k, cls in _REGISTRY.items():
        if cls.name in seen: continue
        seen.add(cls.name)
        out.append((cls.name, cls))
    return out


# ============================================================
# 자가 검증
# ============================================================
if __name__ == "__main__":
    # 가상의 R-134a / 물 비등 조건으로 모든 상관식 호출 테스트
    print("=" * 60)
    print("  [Correlation.py] 자가 검증")
    print("=" * 60)

    # 예시 조건 (Water @ 6 MPa, 포화)
    x      = 0.3
    G      = 500.0          # kg/m²s
    q_flux = 100e3          # W/m²
    D_h    = 2e-3           # m
    L      = 1.0            # m

    rho_l, rho_g = 758.0, 30.83
    mu_l,  mu_g  = 9.7e-5,  1.97e-5
    k_l,   k_g   = 0.566,   0.0651
    Cp_l,  Cp_g  = 5320.0,  5390.0
    h_fg   = 1571e3         # J/kg
    sigma  = 0.0205         # N/m
    P      = 6e6
    P_crit = 22.064e6
    M      = 18.015         # kg/kmol
    T_w    = 550.0
    T_sat  = 548.73
    P_l, P_v = P, P + 50e3  # 단순 가정
    P_H, P_F = math.pi*D_h, math.pi*D_h

    # 단상 (참고)
    Re_l = reynolds_l(G, x, D_h, mu_l)
    Pr_l = prandtl(Cp_l, mu_l, k_l)
    h_db = dittus_boelter(Re_l, Pr_l, k_l, D_h)
    print(f"  Dittus-Boelter (단상 액)  h = {h_db:>10.1f} W/m²K")

    # Chen
    h = chen_1966(x, G, q_flux, D_h, T_w, T_sat,
                  rho_l, rho_g, mu_l, mu_g, k_l, Cp_l,
                  h_fg, sigma, P_l, P_v)
    print(f"  Chen (1966)              h = {h:>10.1f} W/m²K")

    # Shah
    h = shah_1982(x, G, q_flux, D_h, rho_l, rho_g, mu_l, k_l, Cp_l, h_fg)
    print(f"  Shah (1982)              h = {h:>10.1f} W/m²K")

    # Gungor-Winterton
    h = gungor_winterton_1986(x, G, q_flux, D_h,
                               rho_l, rho_g, mu_l, mu_g, k_l, Cp_l,
                               h_fg, P, P_crit, M)
    print(f"  Gungor-Winterton (1986)  h = {h:>10.1f} W/m²K")

    # Bertsch
    h = bertsch_2009(x, G, q_flux, D_h, L,
                      rho_l, rho_g, mu_l, mu_g, k_l, k_g, Cp_l, Cp_g,
                      sigma, h_fg, P, P_crit, M)
    print(f"  Bertsch (2009)           h = {h:>10.1f} W/m²K")

    # Kim-Mudawar
    h = kim_mudawar_2014(x, G, q_flux, D_h,
                          rho_l, rho_g, mu_l, mu_g, k_l, Cp_l,
                          sigma, h_fg, P, P_crit, P_H, P_F)
    print(f"  Kim-Mudawar (2014)       h = {h:>10.1f} W/m²K")

    # Zhang
    h = zhang_2005(x, G, q_flux, D_h, T_w, T_sat,
                    rho_l, rho_g, mu_l, mu_g, k_l, Cp_l,
                    h_fg, sigma, P_l, P_v)
    print(f"  Zhang (2005)             h = {h:>10.1f} W/m²K")

    # Jens-Lottes
    dT = jens_lottes_1951(q_flux, P)
    h_jl = jens_lottes_h(q_flux, P)
    print(f"  Jens-Lottes (1951)       ΔT_sat = {dT:.3f} K   "
          f"h(=q''/ΔT) = {h_jl:.1f} W/m²K")

    # Tran / Yu
    h = tran_1996(G, q_flux, D_h, rho_l, rho_g, sigma, h_fg)
    print(f"  Tran et al. (1996)       h = {h:>12.4e} W/m²K  (단위계 주의)")
    h = yu_2002(G, q_flux, D_h, rho_l, rho_g, sigma, h_fg)
    print(f"  Yu et al. (2002)         h = {h:>10.1f} W/m²K")

    # 디스패처 테스트
    print()
    print("  [디스패처 테스트] get_h_tp('shah', ...)")
    h = get_h_tp("shah", x=x, G=G, q_flux=q_flux, D_h=D_h,
                 rho_l=rho_l, rho_g=rho_g, mu_l=mu_l, k_l=k_l, Cp_l=Cp_l,
                 h_fg=h_fg)
    print(f"  → h = {h:.1f} W/m²K  ✅")

    # ── 어댑터 클래스 테스트 ──
    print()
    print("=" * 60)
    print("  [어댑터 클래스 테스트]")
    print("=" * 60)

    sat_test = {
        "h_fg": h_fg, "sigma": sigma, "T_sat": T_sat,
        "P_crit": P_crit, "M": M,
        "H_l": 0.0, "H_g": h_fg,
    }
    state_test = {
        "regime": "two_phase", "x": x, "T": T_sat,
        "rho_l": rho_l, "rho_g": rho_g,
        "mu_l": mu_l, "mu_g": mu_g,
        "k_l": k_l, "k_g": k_g,
        "Cp_l": Cp_l, "Cp_g": Cp_g,
    }
    ctx_test = {
        "G": G, "q_flux": q_flux, "D_h": D_h,
        "T_wall": T_w, "P_v": P_v, "P": P,
        "sat": sat_test,
        "P_H": P_H, "P_F": P_F, "L_pipe": L,
        "horizontal": False, "mode": "heating",
    }

    for name, cls in list_models():
        if cls.single_phase:
            # 단상 어댑터는 단상 state로 테스트
            sp_state = {"regime":"subcooled","x":0.0,"T":T_sat-20,
                        "rho":rho_l,"mu":mu_l,"k":k_l,"Cp":Cp_l}
            h_val = cls().compute_h(sp_state, ctx_test)
        else:
            h_val = cls().compute_h(state_test, ctx_test)
        flags = []
        if cls.requires_q:     flags.append("q")
        if cls.requires_Twall: flags.append("Twall")
        if cls.requires_Pv:    flags.append("Pv")
        if cls.requires_M:     flags.append("M")
        if cls.requires_geom:  flags.append(f"geom={cls.requires_geom}")
        flag_str = ",".join(flags) if flags else "—"
        print(f"  {name:<20s}  h = {h_val:>12.4e} W/m²K   needs:{flag_str}")

    print()
    print("  ✅ 모든 상관식 + 어댑터 정상 호출")
    print(f"  사용 가능한 이름: {[n for n,_ in list_models()]}")
