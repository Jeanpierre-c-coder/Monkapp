import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class BuildingParameters:
    """Paramètres physiques du bâtiment"""
    # Capacités thermiques (J/K)
    C_building: float = 3.6e6      # Capacité thermique globale
    C_wall: float = 1.8e6          # Capacité thermique des murs
    C_air: float = 1.2e6           # Capacité thermique de l'air
    
    # Coefficients de transmission thermique (W/K)
    U_wall_ext: float = 150.0      # Mur-extérieur
    U_wall_int: float = 300.0      # Mur-intérieur
    U_global: float = 200.0        # Global bâtiment
    U_window: float = 50.0         # Fenêtres
    
    # Caractéristiques fenêtres
    eta_window: float = 0.8        # Efficacité fenêtres
    S_window: float = 20.0         # Surface fenêtres (m²)
    
    # Orientation et géométrie
    building_orientation: float = 0.0  # 0=Sud, 90=Ouest, etc.
    surface_total: float = 150.0   # Surface habitable (m²)
    height: float = 2.7            # Hauteur sous plafond (m)
    
    # Infiltrations et ventilation
    air_changes_per_hour: float = 0.5  # Renouvellement d'air (h⁻¹)

class BuildingThermalModel:
    """Modèle thermique dynamique du bâtiment"""
    
    def __init__(self, params: BuildingParameters):
        self.params = params
        self.state = np.array([20.0, 18.0, 16.0])  # [T_int, T_wall, T_air]
        self.time = 0.0
        
    def thermal_dynamics(self, state: np.ndarray, t: float, 
                        T_ext: float, I_solar: float, Q_heating: float, 
                        Q_occupancy: float, wind_speed: float = 0.0) -> List[float]:
        """
        Système d'équations différentielles couplées pour la thermique
        
        Args:
            state: [T_int, T_wall, T_air] températures actuelles (°C)
            t: temps (s)
            T_ext: température extérieure (°C)
            I_solar: irradiation solaire (W/m²)
            Q_heating: puissance chauffage (W)
            Q_occupancy: apports internes occupation (W)
            wind_speed: vitesse du vent (m/s)
        
        Returns:
            Dérivées temporelles des températures
        """
        T_int, T_wall, T_air = state
        
        # 1. Apports solaires avec angle d'incidence
        sun_angle = self._calculate_sun_angle(t)
        solar_factor = max(0, np.cos(np.radians(sun_angle + self.params.building_orientation)))
        Q_solar = self.params.eta_window * self.params.S_window * I_solar * solar_factor
        
        # 2. Pertes par transmission avec effet du vent
        wind_factor = 1.0 + 0.1 * wind_speed  # Effet convectif du vent
        Q_wall_ext = self.params.U_wall_ext * wind_factor * (T_wall - T_ext)
        Q_wall_int = self.params.U_wall_int * (T_int - T_wall)
        Q_window = self.params.U_window * wind_factor * (T_int - T_ext)
        
        # 3. Pertes par infiltration/ventilation
        rho_air = 1.2  # kg/m³
        Cp_air = 1000  # J/(kg·K)
        V_building = self.params.surface_total * self.params.height
        Q_infiltration = (self.params.air_changes_per_hour * V_building * rho_air * 
                         Cp_air * (T_int - T_ext)) / 3600
        
        # 4. Couplage thermique air-bâtiment
        h_conv_int = 8.0  # W/(m²·K) coefficient convection interne
        A_internal = self.params.surface_total * 3  # Surface d'échange approximative
        Q_conv_internal = h_conv_int * A_internal * (T_air - T_int)
        
        # 5. Équations différentielles
        # Température intérieure (masse thermique principale)
        dT_int_dt = (Q_heating + Q_solar + Q_occupancy + Q_conv_internal - 
                     Q_wall_int - Q_window - Q_infiltration) / self.params.C_building
        
        # Température des murs (inertie thermique)
        dT_wall_dt = (Q_wall_ext - Q_wall_int) / self.params.C_wall
        
        # Température de l'air (réponse rapide)
        dT_air_dt = -Q_conv_internal / self.params.C_air
        
        return [dT_int_dt, dT_wall_dt, dT_air_dt]
    
    def _calculate_sun_angle(self, t: float) -> float:
        """Calcul simplifié de l'angle solaire (degrés)"""
        hours = (t / 3600) % 24  # Conversion en heures
        # Approximation simple: angle solaire varie de -90° (lever) à +90° (coucher)
        if 6 <= hours <= 18:
            return -90 + 15 * (hours - 6)  # 15°/heure
        else:
            return -90  # Nuit
    
    def update(self, dt: float, T_ext: float, I_solar: float, 
               Q_heating: float, Q_occupancy: float = 0.0, 
               wind_speed: float = 0.0) -> Dict[str, float]:
        """
        Mise à jour du modèle thermique
        
        Args:
            dt: pas de temps (s)
            T_ext: température extérieure (°C)
            I_solar: irradiation solaire (W/m²)
            Q_heating: puissance chauffage (W)
            Q_occupancy: apports internes (W)
            wind_speed: vitesse du vent (m/s)
        
        Returns:
            État thermique mis à jour
        """
        # Résolution numérique des EDO
        t_span = [self.time, self.time + dt]
        
        # Fonction wrapper pour odeint
        def dynamics_wrapper(state, t):
            return self.thermal_dynamics(state, t, T_ext, I_solar, 
                                       Q_heating, Q_occupancy, wind_speed)
        
        # Intégration numérique
        solution = odeint(dynamics_wrapper, self.state, t_span)
        self.state = solution[-1]
        self.time += dt
        
        # Limitations physiques
        self.state = np.clip(self.state, -50, 60)  # Températures réalistes
        
        return {
            'T_int': self.state[0],
            'T_wall': self.state[1], 
            'T_air': self.state[2],
            'time': self.time
        }
    
    def get_energy_consumption(self, Q_heating: float, dt: float) -> float:
        """Calcul de la consommation énergétique"""
        # Conversion W·s -> kWh
        return (Q_heating * dt) / 3.6e6
    
    def get_thermal_inertia_indicator(self) -> float:
        """Indicateur d'inertie thermique (différence T_wall - T_int)"""
        return abs(self.state[1] - self.state[0])
    
    def reset(self, initial_temp: float = 20.0):
        """Réinitialisation du modèle"""
        self.state = np.array([initial_temp, initial_temp-2, initial_temp-1])
        self.time = 0.0

# Exemple d'utilisation et test
if __name__ == "__main__":
    # Création du modèle
    params = BuildingParameters(
        C_building=3.6e6,
        U_global=200.0,
        S_window=25.0,
        building_orientation=0  # Orientation Sud
    )
    
    building = BuildingThermalModel(params)
    
    # Simulation sur 24h avec conditions variables
    import matplotlib.pyplot as plt
    
    dt = 300  # 5 minutes
    times = []
    temperatures = []
    heating_power = []
    
    for hour in range(24):
        # Conditions météo simplifiées
        T_ext = 10 + 5 * np.sin(2 * np.pi * hour / 24)  # Variation journalière
        I_solar = max(0, 800 * np.sin(np.pi * hour / 12)) if 6 <= hour <= 18 else 0
        
        # Chauffage simple (thermostat)
        Q_heat = 3000 if building.state[0] < 20 else 0
        
        # Occupation (9h-17h)
        Q_occ = 200 if 9 <= hour <= 17 else 0
        
        # Mise à jour
        state = building.update(dt, T_ext, I_solar, Q_heat, Q_occ)
        
        times.append(hour)
        temperatures.append([state['T_int'], T_ext, state['T_wall']])
        heating_power.append(Q_heat)
    
    # Visualisation
    temperatures = np.array(temperatures)
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(times, temperatures[:, 0], 'r-', label='T_intérieure', linewidth=2)
    plt.plot(times, temperatures[:, 1], 'b--', label='T_extérieure', linewidth=1)
    plt.plot(times, temperatures[:, 2], 'g:', label='T_murs', linewidth=1.5)
    plt.ylabel('Température (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Évolution des températures sur 24h')
    
    plt.subplot(2, 1, 2)
    plt.plot(times, heating_power, 'orange', linewidth=2)
    plt.ylabel('Puissance chauffage (W)')
    plt.xlabel('Heure')
    plt.grid(True, alpha=0.3)
    plt.title('Puissance de chauffage')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Consommation totale: {sum(building.get_energy_consumption(p, dt*3600) for p in heating_power):.2f} kWh")