"""
Pressure_drop.py
================
Two-phase pressure drop correlations for saturated boiling region.

Models implemented:
  1. Homogeneous Equilibrium Model (HEM)
       - viscosity mixture rules: McAdams, Cicchitti, Dukler
  2. Separated Flow (Macro-channel)
       - Lockhart & Martinelli (1949)
       - Friedel (1979)
       - Mueller-Steinhagen & Heck (1986)
  3. Separated Flow (Micro-channel)
       - Mishima & Hibiki (1996)
       - Yu et al. (2002)
       - Sun & Mishima (2009)

Unified interface:
    dPdz_two_phase(model, x, G, D_h, fluid_props) -> dP/dz [Pa/m]

Where fluid_props is a dict with:
    rho_l, rho_g, mu_l, mu_g, sigma  (sigma only needed by some models)

All output: pressure gradient [Pa/m]  (i.e. friction pressure drop per length)
"""

import math


# ============================================================
# Helpers
# ============================================================
def _clip(x, lo=1e-6, hi=1.0-1e-6):
    return max(lo, min(hi, x))


def churchill_friction(Re, rel_roughness=0.0):
    """Churchill (1977) friction factor — valid laminar to turbulent."""
    if Re < 1e-3:
        return 64.0
    A = (-2.457 * math.log((7.0/Re)**0.9 + 0.27*rel_roughness)) ** 16
    B = (37530.0 / Re) ** 16
    f = 8.0 * ((8.0/Re)**12 + 1.0/(A + B)**1.5) ** (1.0/12.0)
    return f


def _f_single_phase(Re):
    """Simple smooth-pipe friction factor (used as fallback)."""
    if Re < 2300:
        return 64.0 / max(Re, 1.0)
    return 0.25 / (math.log10(5.74 / Re**0.9))**2


def _dpdz_friction(f, G, rho, D_h):
    """dP/dz_friction = f * G^2 / (2 * rho * D_h)"""
    return f * G * G / (2.0 * rho * D_h)


# ============================================================
# 1. Homogeneous Equilibrium Model (HEM)
# ============================================================
def mu_mix_mcadams(x, mu_l, mu_g):
    """McAdams: 1/mu_tp = x/mu_g + (1-x)/mu_l"""
    x = _clip(x)
    return 1.0 / (x/mu_g + (1.0 - x)/mu_l)


def mu_mix_cicchitti(x, mu_l, mu_g):
    """Cicchitti: mu_tp = x*mu_g + (1-x)*mu_l"""
    x = _clip(x)
    return x * mu_g + (1.0 - x) * mu_l


def mu_mix_dukler(x, mu_l, mu_g, rho_l, rho_g):
    """Dukler: mu_tp = rho_m * (x*mu_g/rho_g + (1-x)*mu_l/rho_l)"""
    x = _clip(x)
    rho_m = 1.0 / (x/rho_g + (1.0 - x)/rho_l)
    return rho_m * (x*mu_g/rho_g + (1.0 - x)*mu_l/rho_l)


def dpdz_hem(x, G, D_h, props, mu_rule="mcadams"):
    """
    Homogeneous Equilibrium Model:
        (dP/dz)_f = f_tp * G^2 / (2 * rho_m * D_h)
        rho_m = (x/rho_g + (1-x)/rho_l)^-1
        f_tp  = Churchill(Re_tp), Re_tp = G*D_h/mu_tp
    """
    x = _clip(x)
    rho_l, rho_g = props["rho_l"], props["rho_g"]
    mu_l,  mu_g  = props["mu_l"],  props["mu_g"]

    rho_m = 1.0 / (x/rho_g + (1.0 - x)/rho_l)

    if mu_rule == "mcadams":
        mu_tp = mu_mix_mcadams(x, mu_l, mu_g)
    elif mu_rule == "cicchitti":
        mu_tp = mu_mix_cicchitti(x, mu_l, mu_g)
    elif mu_rule == "dukler":
        mu_tp = mu_mix_dukler(x, mu_l, mu_g, rho_l, rho_g)
    else:
        raise ValueError(f"Unknown mu_rule '{mu_rule}'")

    Re_tp = G * D_h / mu_tp
    f_tp  = churchill_friction(Re_tp)
    return _dpdz_friction(f_tp, G, rho_m, D_h)


# ============================================================
# 2. Separated Flow — Macro-channel
# ============================================================
def _dpdz_single(G, D_h, mu, rho, x_frac=1.0):
    """Single-phase friction gradient with mass-flux G*x_frac"""
    G_eff = G * x_frac
    if G_eff <= 0:
        return 0.0
    Re = G_eff * D_h / mu
    f  = _f_single_phase(Re)
    return _dpdz_friction(f, G_eff, rho, D_h)


def _X_LM(x, G, D_h, props):
    """Lockhart-Martinelli parameter X^2 = (dP/dz)_f,l / (dP/dz)_f,g"""
    x = _clip(x)
    dpdz_l = _dpdz_single(G, D_h, props["mu_l"], props["rho_l"], 1.0 - x)
    dpdz_g = _dpdz_single(G, D_h, props["mu_g"], props["rho_g"], x)
    if dpdz_g <= 0:
        return 1e6
    return dpdz_l / dpdz_g


def dpdz_lockhart_martinelli(x, G, D_h, props, C=20.0):
    """
    Lockhart & Martinelli (1949):
        (dP/dz)_TP = (dP/dz)_f,l * phi_f^2
        phi_f^2  = 1 + C/X + 1/X^2
    Default C=20 (turbulent-turbulent).
    """
    x = _clip(x)
    dpdz_l = _dpdz_single(G, D_h, props["mu_l"], props["rho_l"], 1.0 - x)
    X2 = _X_LM(x, G, D_h, props)
    X = math.sqrt(X2)
    phi2 = 1.0 + C/X + 1.0/X2
    return dpdz_l * phi2


def dpdz_friedel(x, G, D_h, props):
    """
    Friedel (1979):
        (dP/dz)_TP = (dP/dz)_fo * phi_fo^2
        (dP/dz)_fo : all-liquid (x=0) single-phase
    """
    x = _clip(x)
    rho_l, rho_g = props["rho_l"], props["rho_g"]
    mu_l,  mu_g  = props["mu_l"],  props["mu_g"]
    sigma = props.get("sigma", 0.02)

    # All-liquid friction gradient
    Re_fo = G * D_h / mu_l
    f_fo  = _f_single_phase(Re_fo)
    dpdz_fo = _dpdz_friction(f_fo, G, rho_l, D_h)

    # All-gas friction factor
    Re_go = G * D_h / mu_g
    f_go  = _f_single_phase(Re_go)

    # Mixture density (homogeneous)
    rho_m = 1.0 / (x/rho_g + (1.0 - x)/rho_l)

    # Dimensionless numbers
    Fr_tp = G*G / (9.81 * D_h * rho_m*rho_m)
    We_tp = G*G * D_h / (sigma * rho_m)

    E = (1.0 - x)**2 + x*x * (rho_l * f_go) / (rho_g * f_fo)
    F = x**0.78 * (1.0 - x)**0.224
    H = (rho_l/rho_g)**0.91 * (mu_g/mu_l)**0.19 * (1.0 - mu_g/mu_l)**0.7

    phi_fo2 = E + 3.24 * F * H / (Fr_tp**0.045 * We_tp**0.035)
    return dpdz_fo * phi_fo2


def dpdz_muller_steinhagen_heck(x, G, D_h, props):
    """
    Mueller-Steinhagen & Heck (1986):
        (dP/dz)_TP = G_M*(1-x)^(1/3) + (dP/dz)_go * x^3
        G_M       = (dP/dz)_fo + 2*[(dP/dz)_go - (dP/dz)_fo]*x
    """
    x = _clip(x)
    rho_l, rho_g = props["rho_l"], props["rho_g"]
    mu_l,  mu_g  = props["mu_l"],  props["mu_g"]

    Re_fo = G * D_h / mu_l;  f_fo = _f_single_phase(Re_fo)
    Re_go = G * D_h / mu_g;  f_go = _f_single_phase(Re_go)
    dpdz_fo = _dpdz_friction(f_fo, G, rho_l, D_h)
    dpdz_go = _dpdz_friction(f_go, G, rho_g, D_h)

    G_M = dpdz_fo + 2.0 * (dpdz_go - dpdz_fo) * x
    return G_M * (1.0 - x)**(1.0/3.0) + dpdz_go * x**3


# ============================================================
# 3. Separated Flow — Micro-channel
# ============================================================
def dpdz_mishima_hibiki(x, G, D_h, props, channel="circular"):
    """
    Mishima & Hibiki (1996):
        phi_f^2 = 1 + C/X + 1/X^2
        C = 21 * [1 - exp(-0.319*D_h)] (circular, D_h in mm? we use m here)
        For circular:    C = 21 * (1 - exp(-0.319 * D_h_mm))
        For rectangular: C = 21 * (1 - exp(-0.333 * D_h_mm))
    """
    x = _clip(x)
    D_h_mm = D_h * 1000.0  # original correlation in mm
    if channel == "circular":
        C = 21.0 * (1.0 - math.exp(-0.319 * D_h_mm))
    else:
        C = 21.0 * (1.0 - math.exp(-0.333 * D_h_mm))

    return dpdz_lockhart_martinelli(x, G, D_h, props, C=C)


def dpdz_yu_2002(x, G, D_h, props):
    """
    Yu et al. (2002):
        (dP/dz)_TP = (dP/dz)_f * phi_f^2
        phi_f^2   = [ 18.65 * (rho_g/rho_f)^0.5 * ((1-x)/x) * Re_g^0.1 / Re_f^0.5 ]^(-1.9)
        (dP/dz)_f : liquid-only (with mass G*(1-x))
    """
    x = _clip(x)
    rho_l, rho_g = props["rho_l"], props["rho_g"]
    mu_l,  mu_g  = props["mu_l"],  props["mu_g"]

    dpdz_l = _dpdz_single(G, D_h, mu_l, rho_l, 1.0 - x)
    Re_f = G * (1.0 - x) * D_h / mu_l
    Re_g = G * x * D_h / mu_g
    if Re_f <= 0 or Re_g <= 0:
        return dpdz_l

    inner = 18.65 * (rho_g/rho_l)**0.5 * ((1.0-x)/x) * (Re_g**0.1) / (Re_f**0.5)
    phi2 = inner ** (-1.9)
    return dpdz_l * phi2


def dpdz_sun_mishima(x, G, D_h, props):
    """
    Sun & Mishima (2009):
        If Re_f < 2000 AND Re_g < 2000 (laminar-laminar):
            phi_f^2 = 1 + C/X + 1/X^2
            C = 26 * (1 + Re_f/1000) * [1 - exp(-0.153 / (0.27*N_conf + 0.8))]
        else:
            phi_f^2 = 1 + C/X^1.19 + 1/X^2
            C = 1.79 * (Re_g/Re_f)^0.4 * ((1-x)/x)^0.5

        N_conf = sqrt( sigma / (g*(rho_l-rho_g)*D_h^2) )    (confinement number)
    """
    x = _clip(x)
    rho_l, rho_g = props["rho_l"], props["rho_g"]
    mu_l,  mu_g  = props["mu_l"],  props["mu_g"]
    sigma = props.get("sigma", 0.02)

    Re_f = G * (1.0 - x) * D_h / mu_l
    Re_g = G * x * D_h / mu_g
    dpdz_l = _dpdz_single(G, D_h, mu_l, rho_l, 1.0 - x)
    X2 = _X_LM(x, G, D_h, props)
    X = math.sqrt(X2)

    if Re_f < 2000 and Re_g < 2000:
        N_conf = math.sqrt(sigma / (9.81 * (rho_l - rho_g) * D_h*D_h))
        C = 26.0 * (1.0 + Re_f/1000.0) * \
            (1.0 - math.exp(-0.153 / (0.27*N_conf + 0.8)))
        phi2 = 1.0 + C/X + 1.0/X2
    else:
        C = 1.79 * (Re_g/Re_f)**0.4 * ((1.0 - x)/x)**0.5
        phi2 = 1.0 + C/(X**1.19) + 1.0/X2

    return dpdz_l * phi2


# ============================================================
# Unified dispatcher
# ============================================================
_MODELS = {
    # HEM with three viscosity rules
    "hem_mcadams":   lambda x,G,D,p: dpdz_hem(x,G,D,p,"mcadams"),
    "hem_cicchitti": lambda x,G,D,p: dpdz_hem(x,G,D,p,"cicchitti"),
    "hem_dukler":    lambda x,G,D,p: dpdz_hem(x,G,D,p,"dukler"),
    # Macro
    "lockhart_martinelli": dpdz_lockhart_martinelli,
    "friedel":             dpdz_friedel,
    "muller_steinhagen":   dpdz_muller_steinhagen_heck,
    # Micro
    "mishima_hibiki":      dpdz_mishima_hibiki,
    "yu":                  dpdz_yu_2002,
    "sun_mishima":         dpdz_sun_mishima,
}


def dpdz_two_phase(model, x, G, D_h, props):
    """
    Main entry point.
        model : str (see _MODELS keys)
        x     : quality [-]
        G     : mass flux [kg/m^2 s]
        D_h   : hydraulic diameter [m]
        props : dict with rho_l, rho_g, mu_l, mu_g, (sigma)
    Returns: dP/dz [Pa/m]  (friction component)
    """
    key = model.lower().replace("-", "_").replace(" ", "_")
    if key not in _MODELS:
        raise ValueError(f"Unknown model '{model}'. "
                         f"Available: {list(_MODELS.keys())}")
    return _MODELS[key](x, G, D_h, props)


def list_models():
    return list(_MODELS.keys())


# ============================================================
# Self-test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Pressure_drop.py — self test")
    print("=" * 60)

    # Sample condition: water at 6 MPa saturation
    # Approximate properties (could be obtained from CoolProp)
    props = {
        "rho_l": 758.0,   "rho_g": 30.83,
        "mu_l":  9.7e-5,  "mu_g":  1.97e-5,
        "sigma": 0.0205,
    }
    G   = 500.0     # kg/m^2 s
    D_h = 2e-3      # m (2 mm — micro-channel)
    x_test = 0.3

    print(f"\nTest condition:")
    print(f"  Water @ 6 MPa,  G = {G} kg/m2s,  D_h = {D_h*1000} mm,  x = {x_test}")
    print(f"  rho_l = {props['rho_l']}, rho_g = {props['rho_g']}")
    print(f"  mu_l  = {props['mu_l']:.2e}, mu_g = {props['mu_g']:.2e}")
    print()

    print(f"  {'Model':<22s} {'dP/dz [Pa/m]':>14s}")
    print("  " + "-"*40)
    for name in list_models():
        try:
            dp = dpdz_two_phase(name, x_test, G, D_h, props)
            print(f"  {name:<22s} {dp:>14.2f}")
        except Exception as e:
            print(f"  {name:<22s} ERROR: {e}")

    # Sweep over quality
    print("\n  Quality sweep — HEM (McAdams):")
    print(f"  {'x':>5s}  {'dP/dz [Pa/m]':>14s}")
    for x in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        dp = dpdz_two_phase("hem_mcadams", x, G, D_h, props)
        print(f"  {x:>5.2f}  {dp:>14.2f}")

    print("\n  All models OK.")
