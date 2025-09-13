import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
import matplotlib.pyplot as plt
from enum import Enum
import warnings
from abc import ABC, abstractmethod
import json
from pathlib import Path

# Constantes physiques
class PhysicalConstants:
    """Constantes physiques pour la modélisation thermique"""
    STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴)
    AIR_SPECIFIC_HEAT = 1005  # J/(kg·K)
    AIR_DENSITY = 1.225  # kg/m³ à 15°C
    WATER_SPECIFIC_HEAT = 4186  # J/(kg·K)
    CONCRETE_SPECIFIC_HEAT = 880  # J/(kg·K)
    CONCRETE_DENSITY = 2400  # kg/m³
    INSULATION_CONDUCTIVITY = 0.04  # W/(m·K)
    CONCRETE_CONDUCTIVITY = 1.7  # W/(m·K)

# Énumérations pour les types de bâtiments
class BuildingType(Enum):
    RESIDENTIAL = "residential"
    OFFICE = "office"
    INDUSTRIAL = "industrial"
    SCHOOL = "school"

class HeatingSystemType(Enum):
    ELECTRIC_RADIATOR = "electric_radiator"
    HEAT_PUMP = "heat_pump"
    GAS_BOILER = "gas_boiler"
    UNDERFLOOR = "underfloor"

class InsulationType(Enum):
    POOR = "poor"          # Bâtiment ancien non rénové
    AVERAGE = "average"    # Isolation standard
    GOOD = "good"          # RT2012
    EXCELLENT = "excellent" # Passif/Bâtiment BBC

@dataclass
class MaterialProperties:
    """Propriétés thermiques des matériaux"""
    density: float  # kg/m³
    specific_heat: float  # J/(kg·K)
    conductivity: float  # W/(m·K)
    thickness: float  # m
    
    @property
    def thermal_mass(self) -> float:
        """Capacité thermique massique (J/(m²·K))"""
        return self.density * self.specific_heat * self.thickness
    
    @property
    def thermal_resistance(self) -> float:
        """Résistance thermique (m²·K/W)"""
        return self.thickness / self.conductivity

@dataclass
class AdvancedBuildingConfig:
    """Configuration avancée du bâtiment avec propriétés physiques détaillées"""
    
    # Géométrie du bâtiment
    floor_area: float = 150.0  # m²
    ceiling_height: float = 2.7  # m
    external_wall_area: float = 120.0  # m²
    window_area: float = 25.0  # m²
    roof_area: float = 150.0  # m²
    
    # Type et caractéristiques
    building_type: BuildingType = BuildingType.RESIDENTIAL
    construction_year: int = 2010
    insulation_type: InsulationType = InsulationType.AVERAGE
    
    # Propriétés des matériaux
    wall_materials: List[MaterialProperties] = field(default_factory=lambda: [
        MaterialProperties(density=2400, specific_heat=880, conductivity=1.7, thickness=0.2),  # Béton
        MaterialProperties(density=20, specific_heat=1200, conductivity=0.04, thickness=0.1),   # Isolation
        MaterialProperties(density=700, specific_heat=1600, conductivity=0.15, thickness=0.01)  # Plâtre
    ])
    
    roof_materials: List[MaterialProperties] = field(default_factory=lambda: [
        MaterialProperties(density=600, specific_heat=1200, conductivity=0.15, thickness=0.02),  # Tuiles
        MaterialProperties(density=20, specific_heat=1200, conductivity=0.04, thickness=0.2),    # Isolation
        MaterialProperties(density=500, specific_heat=1600, conductivity=0.12, thickness=0.02)   # OSB
    ])
    
    # Fenêtres et vitrages
    window_u_value: float = 1.4  # W/(m²·K) - Coefficient de transmission thermique
    window_g_value: float = 0.6  # Facteur solaire
    window_frame_fraction: float = 0.25  # Fraction du cadre
    
    # Système de chauffage
    heating_system: HeatingSystemType = HeatingSystemType.ELECTRIC_RADIATOR
    heating_efficiency: float = 0.95
    max_heating_power: float = 12000  # W
    
    # Ventilation et infiltration
    mechanical_ventilation_rate: float = 0.5  # Vol/h
    infiltration_rate: float = 0.3  # Vol/h (à 50 Pa)
    heat_recovery_efficiency: float = 0.0  # VMC double flux
    
    # Occupation et gains internes
    max_occupancy: int = 4  # Nombre max d'occupants
    occupant_sensible_heat: float = 75  # W par occupant
    occupant_latent_heat: float = 55  # W par occupant
    equipment_heat_gain: float = 500  # W gains équipements
    lighting_heat_gain: float = 200  # W gains éclairage
    
    # Paramètres de confort
    comfort_temp_min: float = 19.0  # °C
    comfort_temp_max: float = 26.0  # °C
    humidity_comfort_min: float = 30.0  # %
    humidity_comfort_max: float = 70.0  # %
    
    def get_global_u_value(self) -> float:
        """Calcule le coefficient U global du bâtiment"""
        # Murs
        wall_resistance = sum(mat.thermal_resistance for mat in self.wall_materials)
        wall_u = 1 / wall_resistance
        
        # Toiture
        roof_resistance = sum(mat.thermal_resistance for mat in self.roof_materials)
        roof_u = 1 / roof_resistance
        
        # Moyenne pondérée
        total_area = self.external_wall_area + self.roof_area + self.window_area
        u_global = (
            (wall_u * self.external_wall_area + 
             roof_u * self.roof_area + 
             self.window_u_value * self.window_area) / total_area
        )
        
        return u_global
    
    def get_thermal_mass(self) -> float:
        """Calcule la masse thermique totale du bâtiment"""
        wall_mass = sum(mat.thermal_mass * self.external_wall_area for mat in self.wall_materials)
        roof_mass = sum(mat.thermal_mass * self.roof_area for mat in self.roof_materials)
        
        # Masse thermique de l'air intérieur
        air_volume = self.floor_area * self.ceiling_height
        air_mass = air_volume * PhysicalConstants.AIR_DENSITY * PhysicalConstants.AIR_SPECIFIC_HEAT
        
        return wall_mass + roof_mass + air_mass

class SolarRadiationModel:
    """Modèle avancé de rayonnement solaire"""
    
    def __init__(self, latitude: float = 48.8566, longitude: float = 2.3522):  # Paris par défaut
        self.latitude = np.radians(latitude)
        self.longitude = np.radians(longitude)
    
    def solar_position(self, day_of_year: int, hour: float) -> Tuple[float, float]:
        """Calcule la position du soleil (élévation, azimuth)"""
        # Déclinaison solaire
        declination = np.radians(23.45) * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        
        # Angle horaire
        hour_angle = np.radians(15 * (hour - 12))
        
        # Élévation solaire
        elevation = np.arcsin(
            np.sin(declination) * np.sin(self.latitude) +
            np.cos(declination) * np.cos(self.latitude) * np.cos(hour_angle)
        )
        
        # Azimuth solaire
        azimuth = np.arctan2(
            np.sin(hour_angle),
            np.cos(hour_angle) * np.sin(self.latitude) - np.tan(declination) * np.cos(self.latitude)
        )
        
        return max(0, elevation), azimuth
    
    def direct_normal_irradiance(self, day_of_year: int, hour: float, 
                               cloud_cover: float = 0.3) -> float:
        """Irradiance directe normale (W/m²)"""
        elevation, _ = self.solar_position(day_of_year, hour)
        
        if elevation <= 0:
            return 0
        
        # Modèle simplifié d'irradiance
        extraterrestrial_irradiance = 1353 * (1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365))
        
        # Atténuation atmosphérique
        air_mass = 1 / np.sin(elevation) if elevation > 0.1 else 10
        atmospheric_transmission = 0.7 ** (air_mass ** 0.678)
        
        # Effet des nuages
        cloud_factor = 1 - 0.8 * cloud_cover
        
        return extraterrestrial_irradiance * atmospheric_transmission * cloud_factor * np.sin(elevation)
    
    def solar_gains(self, building_config: AdvancedBuildingConfig, day_of_year: int, 
                   hour: float, cloud_cover: float = 0.3) -> float:
        """Calcule les gains solaires du bâtiment (W)"""
        dni = self.direct_normal_irradiance(day_of_year, hour, cloud_cover)
        elevation, azimuth = self.solar_position(day_of_year, hour)
        
        if elevation <= 0:
            return 0
        
        # Gains par les fenêtres (supposées orientées sud)
        incident_angle = abs(azimuth)  # Angle d'incidence simplifié
        cosine_factor = max(0, np.cos(incident_angle))
        
        window_gains = (dni * building_config.window_area * 
                       building_config.window_g_value * cosine_factor)
        
        # Gains diffus
        diffuse_irradiance = dni * 0.15  # ~15% de l'irradiance directe
        diffuse_gains = diffuse_irradiance * building_config.window_area * building_config.window_g_value * 0.5
        
        return window_gains + diffuse_gains

class HVACSystem:
    """Modèle avancé de système HVAC"""
    
    def __init__(self, heating_system: HeatingSystemType, config: AdvancedBuildingConfig):
        self.heating_system = heating_system
        self.config = config
        self._power_curve = self._get_power_curve()
    
    def _get_power_curve(self) -> Callable[[float, float], float]:
        """Courbe de puissance du système selon le type"""
        if self.heating_system == HeatingSystemType.HEAT_PUMP:
            return self._heat_pump_power
        elif self.heating_system == HeatingSystemType.GAS_BOILER:
            return self._gas_boiler_power
        else:  # Electric radiator
            return self._electric_radiator_power
    
    def _heat_pump_power(self, control_signal: float, outdoor_temp: float) -> Tuple[float, float]:
        """Modèle de pompe à chaleur avec COP variable"""
        # COP dépendant de la température extérieure
        cop = max(2.0, 4.5 - 0.1 * (20 - outdoor_temp))
        
        # Puissance électrique consommée
        electrical_power = control_signal * self.config.max_heating_power / cop
        thermal_power = electrical_power * cop
        
        return thermal_power, electrical_power
    
    def _gas_boiler_power(self, control_signal: float, outdoor_temp: float) -> Tuple[float, float]:
        """Modèle de chaudière gaz avec rendement variable"""
        efficiency = self.config.heating_efficiency * (0.85 + 0.1 * control_signal)
        gas_power = control_signal * self.config.max_heating_power / efficiency
        thermal_power = gas_power * efficiency
        
        return thermal_power, gas_power
    
    def _electric_radiator_power(self, control_signal: float, outdoor_temp: float) -> Tuple[float, float]:
        """Modèle de radiateur électrique"""
        electrical_power = control_signal * self.config.max_heating_power
        thermal_power = electrical_power * self.config.heating_efficiency
        
        return thermal_power, electrical_power
    
    def get_heating_power(self, control_signal: float, outdoor_temp: float) -> Dict[str, float]:
        """Retourne les puissances thermique et consommée"""
        control_signal = np.clip(control_signal, 0, 1)
        thermal_power, consumed_power = self._power_curve(control_signal, outdoor_temp)
        
        return {
            'thermal_power': thermal_power,
            'consumed_power': consumed_power,
            'efficiency': thermal_power / max(consumed_power, 1e-6)
        }

class AdvancedThermalModel:
    """Modèle thermique avancé multi-zones avec physique réaliste"""
    
    def __init__(self, config: AdvancedBuildingConfig, time_step: float = 60.0):
        self.config = config
        self.time_step = time_step  # secondes
        
        # Modèles intégrés
        self.solar_model = SolarRadiationModel()
        self.hvac_system = HVACSystem(config.heating_system, config)
        
        # Historique pour l'inertie (initialisation AVANT reset)
        self.temperature_history = []
        self.power_history = []
        
        # Variables d'état
        self.reset()
        
        # Paramètres calculés
        self.thermal_mass = config.get_thermal_mass()
        self.u_global = config.get_global_u_value()
        
        # Constantes de temps
        self.time_constant = self.thermal_mass / (self.u_global * (
            config.external_wall_area + config.roof_area + config.window_area))
        
    def reset(self):
        """Réinitialise le modèle"""
        self.indoor_temperature = 20.0  # °C
        self.wall_temperature = 18.0    # °C
        self.indoor_humidity = 50.0     # %
        self.last_heating_power = 0.0   # W
        self.temperature_history.clear()
        self.power_history.clear()
    
    def thermal_dynamics(self, t: float, y: np.ndarray, 
                        outdoor_temp: float, solar_gains: float, 
                        internal_gains: float, heating_power: float,
                        ventilation_losses: float) -> np.ndarray:
        """Équations différentielles du système thermique"""
        T_indoor, T_wall, humidity = y
        
        # Capacités thermiques
        C_indoor = self.config.floor_area * self.config.ceiling_height * \
                  PhysicalConstants.AIR_DENSITY * PhysicalConstants.AIR_SPECIFIC_HEAT
        
        C_wall = sum(mat.thermal_mass * self.config.external_wall_area 
                    for mat in self.config.wall_materials)
        
        # Coefficients de transfert
        U_wall_ext = self.u_global * self.config.external_wall_area
        U_wall_int = 8.0 * self.config.external_wall_area  # Convection interne
        U_window = self.config.window_u_value * self.config.window_area
        
        # Équations différentielles
        # dT_indoor/dt
        dT_indoor_dt = (
            heating_power + solar_gains + internal_gains  # Apports
            - U_wall_int * (T_indoor - T_wall)            # Échange avec murs
            - U_window * (T_indoor - outdoor_temp)        # Pertes fenêtres
            - ventilation_losses                          # Pertes ventilation
        ) / C_indoor
        
        # dT_wall/dt
        dT_wall_dt = (
            U_wall_ext * (outdoor_temp - T_wall) +       # Échange extérieur
            U_wall_int * (T_indoor - T_wall)             # Échange intérieur
        ) / C_wall
        
        # Évolution de l'humidité (simplifiée)
        humidity_production = internal_gains * 0.001  # Production d'humidité
        humidity_extraction = (humidity - 40) * 0.1   # Ventilation
        dhumidity_dt = humidity_production - humidity_extraction
        
        return np.array([dT_indoor_dt, dT_wall_dt, dhumidity_dt])
    
    def calculate_ventilation_losses(self, indoor_temp: float, outdoor_temp: float) -> float:
        """Calcule les pertes par ventilation et infiltration"""
        air_volume = self.config.floor_area * self.config.ceiling_height
        
        # Débit d'air total (m³/s)
        ventilation_flow = (self.config.mechanical_ventilation_rate * air_volume / 3600)
        infiltration_flow = (self.config.infiltration_rate * air_volume / 3600)
        
        total_flow = ventilation_flow + infiltration_flow
        
        # Récupération de chaleur
        if self.config.heat_recovery_efficiency > 0:
            effective_temp_diff = (indoor_temp - outdoor_temp) * \
                                (1 - self.config.heat_recovery_efficiency)
        else:
            effective_temp_diff = indoor_temp - outdoor_temp
        
        # Pertes thermiques (W)
        losses = total_flow * PhysicalConstants.AIR_DENSITY * \
                PhysicalConstants.AIR_SPECIFIC_HEAT * effective_temp_diff
        
        return losses
    
    def calculate_internal_gains(self, occupancy_level: float, 
                               hour: float, day_of_week: int) -> float:
        """Calcule les gains internes variables"""
        # Gains d'occupation
        occupancy = occupancy_level * self.config.max_occupancy
        occupant_gains = occupancy * (self.config.occupant_sensible_heat + 
                                    self.config.occupant_latent_heat)
        
        # Gains d'équipements (variables selon l'heure)
        equipment_factor = 0.3 + 0.7 * (
            0.5 * (1 + np.cos(2 * np.pi * (hour - 18) / 24))
        )  # Plus élevés en soirée
        equipment_gains = self.config.equipment_heat_gain * equipment_factor
        
        # Gains d'éclairage (jour/nuit)
        lighting_factor = 1.0 if 6 <= hour <= 22 else 0.1
        lighting_gains = self.config.lighting_heat_gain * lighting_factor
        
        return occupant_gains + equipment_gains + lighting_gains
    
    def step(self, control_signal: float, outdoor_temp: float, 
             occupancy_level: float, day_of_year: int, hour: float,
             day_of_week: int = 1, cloud_cover: float = 0.3) -> Dict[str, float]:
        """Simulation d'un pas de temps"""
        
        # Calcul des gains solaires
        solar_gains = self.solar_model.solar_gains(
            self.config, day_of_year, hour, cloud_cover
        )
        
        # Calcul des gains internes
        internal_gains = self.calculate_internal_gains(
            occupancy_level, hour, day_of_week
        )
        
        # Calcul des pertes de ventilation
        ventilation_losses = self.calculate_ventilation_losses(
            self.indoor_temperature, outdoor_temp
        )
        
        # Puissance de chauffage
        heating_info = self.hvac_system.get_heating_power(control_signal, outdoor_temp)
        heating_power = heating_info['thermal_power']
        
        # Résolution des équations différentielles
        t_span = [0, self.time_step]
        y0 = [self.indoor_temperature, self.wall_temperature, self.indoor_humidity]
        
        def dynamics(t, y):
            return self.thermal_dynamics(
                t, y, outdoor_temp, solar_gains, internal_gains,
                heating_power, ventilation_losses
            )
        
        solution = solve_ivp(
            dynamics, t_span, y0, method='RK45', rtol=1e-6, atol=1e-8
        )
        
        # Mise à jour des états
        final_state = solution.y[:, -1]
        self.indoor_temperature = final_state[0]
        self.wall_temperature = final_state[1]
        self.indoor_humidity = np.clip(final_state[2], 10, 90)
        
        # Historique
        self.temperature_history.append(self.indoor_temperature)
        self.power_history.append(heating_power)
        
        # Calcul du confort thermique (PMV/PPD)
        comfort_metrics = self.calculate_comfort_metrics(
            self.indoor_temperature, self.indoor_humidity
        )
        
        return {
            'indoor_temperature': self.indoor_temperature,
            'wall_temperature': self.wall_temperature,
            'indoor_humidity': self.indoor_humidity,
            'outdoor_temperature': outdoor_temp,
            'heating_power': heating_power,
            'consumed_power': heating_info['consumed_power'],
            'solar_gains': solar_gains,
            'internal_gains': internal_gains,
            'ventilation_losses': ventilation_losses,
            'system_efficiency': heating_info['efficiency'],
            **comfort_metrics
        }
    
    def calculate_comfort_metrics(self, temperature: float, humidity: float) -> Dict[str, float]:
        """Calcule les métriques de confort (PMV, PPD, violations)"""
        # Confort en température
        temp_violation = 0
        if temperature < self.config.comfort_temp_min:
            temp_violation = self.config.comfort_temp_min - temperature
        elif temperature > self.config.comfort_temp_max:
            temp_violation = temperature - self.config.comfort_temp_max
        
        # Confort en humidité
        humidity_violation = 0
        if humidity < self.config.humidity_comfort_min:
            humidity_violation = self.config.humidity_comfort_min - humidity
        elif humidity > self.config.humidity_comfort_max:
            humidity_violation = humidity - self.config.humidity_comfort_max
        
        # PMV simplifié (Predicted Mean Vote)
        # Modèle simplifié basé sur la température operative
        pmv = 0.28 * (temperature - 22)  # Approximation linéaire
        
        # PPD (Predicted Percentage Dissatisfied)
        ppd = 100 - 95 * np.exp(-0.03353 * pmv**4 - 0.2179 * pmv**2)
        
        return {
            'temperature_violation': temp_violation,
            'humidity_violation': humidity_violation,
            'pmv': pmv,
            'ppd': ppd,
            'comfort_score': max(0, 100 - temp_violation * 10 - humidity_violation * 2)
        }
    
    def get_thermal_characteristics(self) -> Dict[str, float]:
        """Retourne les caractéristiques thermiques du bâtiment"""
        return {
            'thermal_mass': self.thermal_mass,
            'u_global': self.u_global,
            'time_constant': self.time_constant,
            'floor_area': self.config.floor_area,
            'heated_volume': self.config.floor_area * self.config.ceiling_height,
            'max_heating_power': self.config.max_heating_power,
            'specific_power': self.config.max_heating_power / self.config.floor_area
        }
    
    def optimize_control_strategy(self, weather_forecast: List[Dict], 
                                occupancy_forecast: List[float],
                                horizon_hours: int = 24) -> List[float]:
        """Optimisation prédictive de la stratégie de contrôle"""
        def objective(control_sequence):
            total_cost = 0
            temp_model = self.indoor_temperature
            
            for i, control in enumerate(control_sequence):
                if i >= len(weather_forecast):
                    break
                    
                weather = weather_forecast[i]
                occupancy = occupancy_forecast[i % len(occupancy_forecast)]
                
                # Simulation du pas de temps
                result = self.step(
                    control, weather['temperature'], occupancy,
                    weather.get('day_of_year', 1),
                    weather.get('hour', 12),
                    cloud_cover=weather.get('cloud_cover', 0.3)
                )
                
                # Coût = consommation + pénalité confort
                energy_cost = result['consumed_power'] * 0.15 / 1000  # €/kWh
                comfort_penalty = result['temperature_violation'] * 2
                
                total_cost += energy_cost + comfort_penalty
                temp_model = result['indoor_temperature']
            
            return total_cost
        
        # Optimisation avec contraintes
        x0 = [0.5] * horizon_hours  # Point de départ
        bounds = [(0, 1) for _ in range(horizon_hours)]  # Contraintes de contrôle
        
        result = minimize(
            objective, x0, bounds=bounds, method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        
        return result.x.tolist()

# Exemple d'utilisation et de test
if __name__ == "__main__":
    # Configuration d'un bâtiment résidentiel bien isolé
    config = AdvancedBuildingConfig(
        floor_area=120.0,
        building_type=BuildingType.RESIDENTIAL,
        insulation_type=InsulationType.GOOD,
        heating_system=HeatingSystemType.HEAT_PUMP,
        max_heating_power=8000,
        heat_recovery_efficiency=0.85
    )
    
    # Création du modèle
    thermal_model = AdvancedThermalModel(config, time_step=300)  # 5 minutes
    
    # Simulation sur 24h
    results = []
    for hour in range(24):
        for minute in [0, 20, 40]:  # Pas de 20 minutes
            # Conditions variables
            outdoor_temp = 5 + 10 * np.sin(2 * np.pi * (hour + minute/60) / 24)
            occupancy = 0.8 if 18 <= hour <= 23 else 0.2
            control = 0.6  # 60% de puissance
            
            result = thermal_model.step(
                control_signal=control,
                outdoor_temp=outdoor_temp,
                occupancy_level=occupancy,
                day_of_year=15,  # Janvier
                hour=hour + minute/60,
                cloud_cover=0.4
            )
            
            results.append({
                'time': hour + minute/60,
                **result
            })
    
    # Analyse des résultats
    df = pd.DataFrame(results)
    print(f"Température moyenne: {df['indoor_temperature'].mean():.2f}°C")
    print(f"Consommation totale: {df['consumed_power'].sum()/1000:.2f} kWh")
    print(f"Score de confort moyen: {df['comfort_score'].mean():.1f}/100")
    print(f"Violations de confort: {(df['temperature_violation'] > 0).sum()} sur {len(df)}")
    
    # Caractéristiques thermiques
    characteristics = thermal_model.get_thermal_characteristics()
    print("\nCaractéristiques thermiques du bâtiment:")
    for key, value in characteristics.items():
        print(f"  {key}: {value:.2f}")