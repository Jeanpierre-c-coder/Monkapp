# Fix the quoting issue by using triple single quotes inside the embedded code.
from textwrap import dedent
from pathlib import Path

app_code_fixed = dedent(r"""
# -*- coding: utf-8 -*-
import os
import sys
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# -------------------------------------------------------------
# Imports facultatifs des modules de l'utilisateur (si présents)
# -------------------------------------------------------------
# On supporte plusieurs variantes de noms de fichiers pour éviter les soucis.
BTM = None
PIDController = None
OnOffController = None

def _try_import_user_modules():
    global BTM, PIDController, OnOffController
    # 1) Modèle "modèle thermique amélioré..." (RC 2-noeuds)
    #    On expose BuildingThermalModel compatible via un wrapper léger.
    candidates = [
        "building_thermal_model",  # si l'utilisateur a ce module
        "modèle thermique amélioré avec des techniques de modélisation avancées",  # nom FR
    ]
    for name in candidates:
        try:
            mod = __import__(name.replace(" ", "_"))
            if hasattr(mod, "BuildingThermalModel"):
                BTM = mod.BuildingThermalModel
                PIDController = getattr(mod, "PIDController", None)
                OnOffController = getattr(mod, "OnOffController", None)
                return True
        except Exception:
            pass
    return False

_import_ok = _try_import_user_modules()

# -----------------------------------------------------------------
# Fallback minimal : RC 2-nœuds (air + enveloppe) + contrôleurs
# -----------------------------------------------------------------
# Si les imports utilisateurs n'existent pas, on fournit un modèle équivalent.
if not _import_ok:
    from enum import Enum

    class HeatingSystemType(Enum):
        ELECTRIC_RADIATOR = "electric_radiator"
        HEAT_PUMP = "heat_pump"

    @dataclass
    class HeatingSystem:
        type: HeatingSystemType = HeatingSystemType.HEAT_PUMP
        max_power: float = 6000.0
        cop_nominal: float = 3.2

        def deliver_power(self, demand_W: float, Te: float, Ti: float) -> Tuple[float, float]:
            demand_W = max(demand_W, 0.0)
            P_in = min(demand_W, self.max_power)
            if self.type == HeatingSystemType.ELECTRIC_RADIATOR:
                return P_in, P_in
            # COP qui baisse avec l'écart de température (simple mais efficace)
            dT = max(Ti - Te, 1.0)
            cop = max(1.6, self.cop_nominal - 0.03 * (dT - 20.0))
            return P_in, P_in / cop

    @dataclass
    class FallbackConfig:
        floor_area: float = 90.0
        external_wall_area: float = 100.0
        roof_area: float = 90.0
        window_area: float = 18.0
        air_volume: float = 90.0 * 2.5
        Ca: float = 1.2 * 1005 * (90.0 * 2.5)  # J/K
        Ce: float = 180_000 * (100 + 90) / 2   # J/K
        U_wall: float = 0.45
        U_window: float = 1.4
        U_roof: float = 0.25
        h_in: float = 7.5
        h_out: float = 15.0
        infiltration_ach: float = 0.5
        hvac: HeatingSystem = field(default_factory=HeatingSystem)

    class _OnOffController:
        def __init__(self, deadband: float = 0.3, demand_power: float = 5000.0):
            self.deadband = deadband
            self.power = demand_power
        def control(self, t, Ti, sp_h, sp_c=None):
            if Ti < sp_h - self.deadband:
                return self.power
            if sp_c is not None and Ti > sp_c + self.deadband:
                return -self.power
            return 0.0

    class _PIDController:
        def __init__(self, kp=1200.0, ki=0.02, kd=0.0, limit=6000.0):
            self.kp, self.ki, self.kd = kp, ki, kd
            self.limit = limit
            self.ei = 0.0
            self.last_t = None
            self.last_e = None
        def control(self, t, Ti, sp_h, sp_c=None):
            e = sp_h - Ti
            if self.last_t is None:
                self.last_t, self.last_e = t, e
            dt = max(t - self.last_t, 1.0)
            de = e - (self.last_e if self.last_e is not None else e)
            self.ei = np.clip(self.ei + e * dt, -5e3, 5e3)
            u = self.kp*e + self.ki*self.ei + self.kd*(de/dt)
            self.last_t, self.last_e = t, e
            return float(np.clip(u, -self.limit, self.limit))

    OnOffController = OnOffController or _OnOffController
    PIDController = PIDController or _PIDController

    class FallbackModel:
        def __init__(self, cfg: FallbackConfig, controller):
            self.cfg = cfg
            self.ctrl = controller
        def _conductances(self):
            UA = (self.cfg.U_wall * (self.cfg.external_wall_area) +
                  self.cfg.U_window * self.cfg.window_area +
                  self.cfg.U_roof * self.cfg.roof_area)
            qv = self.cfg.infiltration_ach * self.cfg.air_volume / 3600.0
            G_inf = 1.225 * 1005 * qv
            return UA, G_inf
        def step(self, t, Ti, Tev, Te, Gsol, Qi, sp):
            UA, G_inf = self._conductances()
            A_int = self.cfg.external_wall_area + self.cfg.roof_area
            G_in = self.cfg.h_in * A_int
            G_out = self.cfg.h_out * A_int

            # Contrôle puissance
            P_dem = self.ctrl.control(t, Ti, sp, None)
            if P_dem >= 0:
                P_in, Ein = self.cfg.hvac.deliver_power(P_dem, Te, Ti)
                Qhvac = P_in
            else:
                P_in, Ein = self.cfg.hvac.deliver_power(-P_dem, Te, Ti)
                Qhvac = -P_in

            Qsol_air = 0.6 * self.cfg.window_area * max(Gsol, 0.0)
            Qsol_env = 0.4 * self.cfg.window_area * max(Gsol, 0.0)
            Qint_air = 0.7 * Qi
            Qint_env = 0.3 * Qi

            dTi = ((G_inf)*(Te - Ti) + G_in*(Tev - Ti) + Qsol_air + Qint_air + Qhvac) / self.cfg.Ca
            dTev = (G_in*(Ti - Tev) + G_out*(Te - Tev) + Qsol_env + Qint_env) / self.cfg.Ce

            return dTi, dTev, P_dem, Qhvac, Ein

    # Wrapper pour uniformiser l'API avec le modèle utilisateur
    class BuildingThermalModel:
        def __init__(self, controller=None):
            self.cfg = FallbackConfig()
            self.controller = controller or PIDController()
            self.model = FallbackModel(self.cfg, self.controller)
        def simulate(self, t0, tf, dt, Ti0=None, Tev0=None, weather=None, occupancy=None):
            if Ti0 is None:
                Ti0 = weather["Tout"][0] if weather is not None else 18.0
            if Tev0 is None:
                Tev0 = Ti0
            t = np.arange(t0, tf+dt, dt)
            Ti = np.zeros_like(t, dtype=float)
            Tev = np.zeros_like(t, dtype=float)
            Pdem = np.zeros_like(t, dtype=float)
            Phvac = np.zeros_like(t, dtype=float)
            Pin = np.zeros_like(t, dtype=float)

            Ti[0], Tev[0] = Ti0, Tev0
            for k in range(1, len(t)):
                Te = weather["Tout"][k]
                Gs = weather["Gsol"][k]
                Qi = occupancy["Qint"][k]
                sp = occupancy["Setpoint"][k]
                dTi, dTev, P_dem, Qhvac, Ein = self.model.step(t[k], Ti[k-1], Tev[k-1], Te, Gs, Qi, sp)
                Ti[k] = Ti[k-1] + dTi*dt
                Tev[k] = Tev[k-1] + dTev*dt
                Pdem[k] = P_dem
                Phvac[k] = Qhvac
                Pin[k] = Ein

            df = pd.DataFrame({
                "t_s": t,
                "T_indoor_C": Ti,
                "T_envelope_C": Tev,
                "T_outdoor_C": weather["Tout"],
                "G_solar_Wm2": weather["Gsol"],
                "Q_internal_W": occupancy["Qint"],
                "Setpoint_heat_C": occupancy["Setpoint"],
                "P_demand_W": Pdem,
                "P_system_to_air_W": Phvac,
                "Power_input_elec_or_fuel_W": Pin
            })
            df["dt_s"] = np.gradient(df["t_s"])
            df["E_input_Wh"] = (df["Power_input_elec_or_fuel_W"] * df["dt_s"] / 3600.0).cumsum()
            return df

# --------------------------------------
# Outils : météo & occupation synthétiques
# --------------------------------------
def synthetic_weather(hours: int, Tout_day=6.0, Tout_night=1.0, solar_peak=350.0, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(0, hours*3600+3600, 3600)
    hh = (t/3600) % 24
    is_day = (hh >= 7) & (hh <= 18)
    Tout = np.where(is_day,
                    Tout_day - 2*np.cos((hh-7)/(11)*np.pi),
                    Tout_night + 1*np.cos((hh-18)/(13)*np.pi))
    Tout += rng.normal(0, 0.4, size=Tout.shape)  # petite variabilité
    Gsol = np.zeros_like(Tout)
    mask = (hh >= 8) & (hh <= 16)
    Gsol[mask] = solar_peak * np.sin((hh[mask]-8)/8*np.pi)
    return {"t": t, "Tout": Tout, "Gsol": Gsol}

def synthetic_occupancy(hours: int,
                        internal_day_W=800.0,
                        internal_night_W=200.0,
                        setpoint_day_C=20.5,
                        setpoint_night_C=18.5,
                        day_start=7, night_start=22):
    t = np.arange(0, hours*3600+3600, 3600)
    hh = (t/3600) % 24
    is_day = (hh >= day_start) & (hh < night_start)
    Qint = np.where(is_day, internal_day_W, internal_night_W)
    Setpoint = np.where(is_day, setpoint_day_C, setpoint_night_C)
    return {"t": t, "Qint": Qint, "Setpoint": Setpoint}

# --------------------------------------
# Streamlit UI
# --------------------------------------
st.set_page_config(page_title="Régulation thermique intelligente", layout="wide")

st.title("🏠 Régulation thermique intelligente — Simulation & RL (démo)")

with st.sidebar:
    st.header("⚙️ Paramètres généraux")
    sim_hours = st.slider("Durée de simulation (heures)", 24, 336, 168, step=24)
    dt_h = st.selectbox("Pas de temps (h)", [0.25, 0.5, 1.0], index=2)
    dt_s = int(dt_h*3600)

    st.markdown("---")
    st.subheader("Météo synthétique")
    Tout_day = st.number_input("T° ext jour (°C)", value=6.0, step=0.5)
    Tout_night = st.number_input("T° ext nuit (°C)", value=1.0, step=0.5)
    solar_peak = st.number_input("Irradiance max (W/m²)", value=350.0, step=10.0)

    st.markdown("---")
    st.subheader("Occupation & consigne")
    int_day = st.number_input("Gains internes jour (W)", value=800.0, step=50.0)
    int_night = st.number_input("Gains internes nuit (W)", value=200.0, step=50.0)
    sp_day = st.number_input("Consigne jour (°C)", value=20.5, step=0.1)
    sp_night = st.number_input("Consigne nuit (°C)", value=18.5, step=0.1)

    st.markdown("---")
    st.subheader("Contrôleur")
    controller_type = st.selectbox("Type", ["PID", "On/Off"])
    if controller_type == "PID":
        kp = st.number_input("Kp", value=1200.0, step=50.0)
        ki = st.number_input("Ki", value=0.02, step=0.01, format="%.2f")
        kd = st.number_input("Kd", value=0.0, step=0.01)
    else:
        deadband = st.number_input("Deadband (°C)", value=0.3, step=0.1)
        demand_power = st.number_input("Puissance demande (W)", value=5000.0, step=100.0)

col1, col2 = st.columns([1.2, 1])

# --------------------------------------
# Simulation
# --------------------------------------
with col1:
    st.subheader("📈 Simulation physique (RC 2-nœuds)")

    # Crée les profils
    weather = synthetic_weather(sim_hours, Tout_day, Tout_night, solar_peak)
    occ = synthetic_occupancy(sim_hours, int_day, int_night, sp_day, sp_night)
    # Discrétise aux pas choisis
    # (nos synthétiques sont déjà à 1h; si dt<1h on interpole)
    if dt_h < 1.0:
        t_grid = np.arange(0, sim_hours*3600+dt_s, dt_s)
        weather["Tout"] = np.interp(t_grid, weather["t"], weather["Tout"])
        weather["Gsol"] = np.interp(t_grid, weather["t"], weather["Gsol"])
        occ["Qint"] = np.interp(t_grid, occ["t"], occ["Qint"])
        occ["Setpoint"] = np.interp(t_grid, occ["t"], occ["Setpoint"])
        weather["t"] = t_grid
        occ["t"] = t_grid

    # Construire le contrôleur
    if controller_type == "PID":
        controller = PIDController(kp=kp, ki=ki, kd=kd, limit=6000.0) if hasattr(PIDController, "__call__") else PIDController(kp, ki, kd, 6000.0)
    else:
        controller = OnOffController(deadband=deadband, demand_power=demand_power)

    # Modèle
    model = BuildingThermalModel(controller=controller)

    # Simulation
    df = model.simulate(0.0, weather["t"][-1], dt_s, weather=weather, occupancy=occ)

    # Métriques
    comfort_band = 0.5  # ±0.5°C
    err = df["T_indoor_C"] - df["Setpoint_heat_C"]
    comfort_viol = np.mean(np.abs(err) > comfort_band) * 100.0
    energy_kwh = df["E_input_Wh"].iloc[-1] / 1000.0
    peak_power = df["P_system_to_air_W"].abs().max()

    st.metric("Taux de violations de confort (±0,5°C)", f"{comfort_viol:.1f}%")
    st.metric("Énergie consommée", f"{energy_kwh:.2f} kWh")
    st.metric("Puissance crête délivrée", f"{peak_power:.0f} W")

    # Graphiques
    t_h = (df["t_s"].values - df["t_s"].values[0]) / 3600.0
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(t_h, df["T_indoor_C"], label="T_int (°C)")
    ax1.plot(t_h, df["T_outdoor_C"], label="T_ext (°C)", alpha=0.7)
    ax1.plot(t_h, df["Setpoint_heat_C"], "--", label="Consigne (°C)")
    ax1.set_xlabel("Heures")
    ax1.set_ylabel("Température (°C)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(t_h, df["P_system_to_air_W"], label="P -> air (W)")
    ax2.set_xlabel("Heures")
    ax2.set_ylabel("Puissance (W)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")
    st.pyplot(fig2)

with col2:
    st.subheader("🔧 Astuces pour de meilleurs résultats")
    st.markdown('''
- **PID recommandé** pour une meilleure tenue de consigne. Ajustez `Kp` puis ajoutez un petit `Ki`.
- **Réduisez le pas** (0.5h ou 0.25h) pour plus de stabilité si ça oscille.
- **Consignes jour/nuit**: 20–21°C jour, 18–19°C nuit sont souvent un bon compromis.
- **Gains internes réalistes**: 600–900W jour pour 100 m² (personnes+équipements/éclairage).
- **PAC**: un COP qui dépend de l'écart `Ti-Te` améliore l'estimation de l'énergie.
- **Limiter la puissance crête** aide à réduire les surchauffes (anti-windup du PID déjà inclus).
''')

# -------------------------------------------------------
# (Optionnel) RL démo : si l'agent de l'utilisateur existe
# -------------------------------------------------------
st.markdown("---")
st.header("🤖 Entraînement RL (démo accélérée) — optionnel")

with st.expander("Afficher / masquer"):
    try:
        from thermal_environment import ThermalBuildingEnv, RewardConfig
        from lstm_dqn_agent import LSTMDQNAgent, TrainingConfig
        rl_available = True
    except Exception as e:
        rl_available = False
        st.info("Modules RL non détectés (`thermal_environment.py` et `lstm_dqn_agent.py`). Seule la simulation physique est disponible.")

    if rl_available:
        episodes = st.slider("Épisodes (démo)", 5, 100, 20, step=5)
        max_steps = st.slider("Pas/épisode (heures)", 24, 168, 48, step=24)
        lr = st.number_input("Learning rate", value=5e-4, format="%.6f")
        batch = st.number_input("Batch size", value=32, step=1)
        seq_len = st.number_input("Longueur des séquences (h)", value=12, step=1)
        epsilon_decay = st.number_input("Epsilon decay", value=400, step=50)

        if st.button("Lancer l'entraînement (démo)"):
            env = ThermalBuildingEnv(episode_length_hours=max_steps, time_step_hours=1.0)
            obs = env.reset()
            agent = LSTMDQNAgent(len(obs), env.action_space.n,
                                 TrainingConfig(learning_rate=lr,
                                                batch_size=int(batch),
                                                sequence_length=int(seq_len),
                                                epsilon_decay=int(epsilon_decay)))
            rewards = []
            for ep in range(int(episodes)):
                ob = env.reset()
                done = False
                ep_reward = 0.0
                steps = 0
                while not done and steps < max_steps:
                    action = agent.select_action(ob, training=True)
                    nxt, r, done, info = env.step(action)
                    agent.step(ob, action, r, nxt, done)
                    ob = nxt
                    ep_reward += r
                    steps += 1
                rewards.append(ep_reward)
                st.write(f"Episode {ep+1}/{episodes} — Reward: {ep_reward:.2f}")
            fig, ax = plt.subplots(figsize=(8,3))
            ax.plot(rewards, marker="o")
            ax.set_title("Courbe de récompense (démo)")
            ax.set_xlabel("Épisode")
            ax.set_ylabel("Reward")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

# -----------------
# Export des données
# -----------------
st.subheader("💾 Export")
fname = st.text_input("Nom de fichier (CSV)", "simulation_results.csv")
if st.button("Exporter"):
    try:
        df.to_csv(fname, index=False)
        st.success(f"Fichier sauvegardé : {fname}")
    except Exception as e:
        st.error(str(e))
""")

app_path = Path("/mnt/data/streamlit_app.py")
app_path.write_text(app_code_fixed, encoding="utf-8")
print("Created:", app_path)
