#!/usr/bin/env python3
"""
Script de démonstration principal pour le système d'apprentissage par renforcement
appliqué à la régulation thermique de bâtiment.

Ce script permet de :
1. Tester individuellement chaque composant
2. Lancer un entraînement rapide
3. Comparer les différents contrôleurs
4. Analyser les performances

Usage:
    python main_demo.py --mode [test|train|compare|analyze]
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Imports des modules du projet
try:
    from building_thermal_model import BuildingThermalModel, BuildingParameters
    from user_models import RegularUserModel, StochasticUserModel, MultiUserManager
    from weather_interface import SyntheticWeatherProvider, WeatherProcessor
    from lstm_dqn_agent import LSTMDQNAgent, TrainingConfig
    from thermal_environment import ThermalBuildingEnv, ThermalEnvManager, RewardConfig
    from training_evaluation import ThermalTrainer, BenchmarkEvaluator, BaselineController
except ImportError as e:
    print(f"Erreur d'import: {e}")
    print("Assurez-vous que tous les modules sont dans le même dossier.")
    sys.exit(1)

class ThermalSystemDemo:
    """Classe principale pour la démonstration du système thermique"""
    
    def __init__(self):
        self.setup_directories()
        print("=== SYSTÈME DE RÉGULATION THERMIQUE INTELLIGENT ===")
        print("Basé sur l'apprentissage par renforcement avec mémoire LSTM\n")
    
    def setup_directories(self):
        """Crée les dossiers nécessaires"""
        for folder in ['models', 'logs', 'results', 'plots']:
            os.makedirs(folder, exist_ok=True)
    
    def test_components(self):
        """Test rapide de tous les composants"""
        print("🔧 TEST DES COMPOSANTS\n")
        
        # 1. Test modèle thermique
        print("1. Modèle thermique du bâtiment...")
        building = BuildingThermalModel(BuildingParameters())
        
        # Simulation courte
        for hour in range(24):
            state = building.update(
                dt=3600,  # 1 heure
                T_ext=10 + 5 * np.sin(2 * np.pi * hour / 24),
                I_solar=max(0, 800 * np.sin(np.pi * hour / 12)),
                Q_heating=2000,
                Q_occupancy=100
            )
            if hour % 6 == 0:
                print(f"   H{hour:2d}: T_int = {state['T_int']:.1f}°C")
        
        print("   ✅ Modèle thermique fonctionnel\n")
        
        # 2. Test modèles utilisateurs
        print("2. Modèles d'utilisateurs...")
        regular_user = RegularUserModel()
        stochastic_user = StochasticUserModel(seed=123)
        
        presence_regular = sum(regular_user.is_present(h) for h in range(24))
        presence_stochastic = sum(stochastic_user.is_present(h) for h in range(24))
        
        print(f"   Utilisateur régulier: {presence_regular}/24h présent")
        print(f"   Utilisateur stochastique: {presence_stochastic}/24h présent")
        print("   ✅ Modèles utilisateurs fonctionnels\n")
        
        # 3. Test interface météo
        print("3. Interface météorologique...")
        weather = SyntheticWeatherProvider(seed=42)
        processor = WeatherProcessor(weather)
        
        current_weather = weather.get_current_weather()
        thermal_params = processor.get_thermal_parameters(48.8566, 2.3522)
        
        print(f"   Température: {current_weather.temperature:.1f}°C")
        print(f"   Irradiance: {current_weather.solar_irradiance:.0f} W/m²")
        print(f"   Gains solaires bâtiment: {thermal_params['solar_gains']:.0f}W")
        print("   ✅ Interface météo fonctionnelle\n")
        
        # 4. Test environnement
        print("4. Environnement de simulation...")
        env = ThermalBuildingEnv(episode_length_hours=48)
        obs = env.reset()
        
        total_reward = 0
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
        print(f"   Espace observation: {len(obs)} dimensions")
        print(f"   Espace actions: {env.action_space.n} actions")
        print(f"   Récompense test: {total_reward:.2f}")
        print("   ✅ Environnement fonctionnel\n")
        
        # 5. Test agent (création seulement)
        print("5. Agent LSTM-DQN...")
        config = TrainingConfig(
            learning_rate=1e-3,
            batch_size=8,
            sequence_length=6,
            memory_size=1000
        )
        
        agent = LSTMDQNAgent(len(obs), env.action_space.n, config)
        
        # Test sélection action
        test_obs = {
            'T_int': 20.0, 'T_ext': 15.0, 'T_target': 21.0,
            'T_ext_forecast_6h': 14.0, 'T_ext_forecast_12h': 13.0,
            'solar_gains': 200.0, 'occupancy': 1.0, 'occupancy_forecast_2h': 1.0,
            'hour': 14, 'day_of_week': 2,
            'heating_power_prev': 0.4, 'energy_consumed_24h': 80.0
        }
        
        action = agent.select_action(test_obs, training=False)
        print(f"   Taille réseau: {sum(p.numel() for p in agent.q_network.parameters())} paramètres")
        print(f"   Action test: {action}")
        print(f"   Device: {agent.device}")
        print("   ✅ Agent LSTM-DQN fonctionnel\n")
        
        print("🎉 TOUS LES COMPOSANTS SONT OPÉRATIONNELS !\n")
    
    def quick_training_demo(self):
        """Démonstration d'entraînement rapide (quelques épisodes)"""
        print("🚀 DÉMONSTRATION D'ENTRAÎNEMENT RAPIDE\n")
        
        # Configuration légère pour démo
        config = {
            'training': {
                'episodes': 20,
                'max_steps_per_episode': 48,  # 2 jours
                'save_frequency': 10,
                'eval_frequency': 10,
                'eval_episodes': 2
            },
            'agent': {
                'learning_rate': 5e-3,  # Apprentissage plus rapide
                'batch_size': 8,
                'sequence_length': 6,
                'memory_size': 2000,
                'epsilon_decay': 100  # Exploration plus rapide
            },
            'environment': {
                'episode_length_hours': 48,
                'time_step_hours': 1.0
            }
        }
        
        # Entraînement
        trainer = ThermalTrainer(config)
        trainer.setup_training()
        
        print("Début entraînement (20 épisodes)...")
        start_time = datetime.now()
        
        for episode in range(config['training']['episodes']):
            observation = trainer.train_env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < config['training']['max_steps_per_episode']:
                action = trainer.agent.select_action(observation, training=True)
                next_obs, reward, done, info = trainer.train_env.step(action)
                trainer.agent.step(observation, action, reward, next_obs, done)
                
                observation = next_obs
                episode_reward += reward
                steps += 1
            
            trainer.training_metrics['episode_rewards'].append(episode_reward)
            
            if episode % 5 == 0:
                stats = trainer.agent.get_training_stats()
                print(f"  Episode {episode:2d}: Reward={episode_reward:6.1f}, "
                      f"ε={stats['epsilon']:.3f}, Steps={steps}")
        
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\nEntraînement terminé en {duration:.1f}s")
        
        # Analyse rapide des résultats
        rewards = trainer.training_metrics['episode_rewards']
        print(f"Récompense initiale: {rewards[0]:.1f}")
        print(f"Récompense finale: {rewards[-1]:.1f}")
        print(f"Amélioration: {((rewards[-1] - rewards[0]) / abs(rewards[0]) * 100):+.1f}%")
        
        # Plot simple
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(rewards)
        plt.title('Progression des récompenses')
        plt.xlabel('Episode')
        plt.ylabel('Récompense')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        epsilons = [trainer.agent.config.epsilon_start * 
                   (trainer.agent.config.epsilon_decay / (trainer.agent.config.epsilon_decay + i))
                   for i in range(len(rewards))]
        plt.plot(epsilons)
        plt.title('Décroissance de l\'exploration')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/quick_training_demo.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return trainer.agent
    
    def controller_comparison_demo(self):
        """Démonstration comparative des contrôleurs"""
        print("⚖️  COMPARAISON DES CONTRÔLEURS\n")
        
        # Création des contrôleurs
        controllers = {
            'Thermostat Simple': BaselineController.SimpleThermostat(hysteresis=0.5),
            'PID': BaselineController.PIDController(kp=2.0, ki=0.1, kd=0.5),
            'Programmé': BaselineController.ScheduledController()
        }
        
        # Environnement de test
        env = ThermalBuildingEnv(
            episode_length_hours=168,  # 1 semaine
            time_step_hours=1.0
        )
        
        results = {}
        
        print("Test sur 1 semaine (168h)...\n")
        
        for name, controller in controllers.items():
            print(f"🔹 Test {name}...")
            
            # Reset contrôleur si nécessaire
            if hasattr(controller, 'heating_on'):
                controller.heating_on = False
            if hasattr(controller, 'integral'):
                controller.integral = 0.0
                controller.previous_error = 0.0
            
            observation = env.reset()
            total_reward = 0
            total_energy = 0
            comfort_violations = 0
            temperatures = []
            heating_powers = []
            done = False
            step = 0
            
            while not done and step < 168:
                action = controller.select_action(observation)
                next_obs, reward, done, info = env.step(action)
                
                total_reward += reward
                step += 1
                
                # Métriques
                T_int = info['building_state']['T_int']
                T_target = observation['T_target']
                heating_power = action * 5000 / 5  # Conversion approximative
                
                temperatures.append(abs(T_int - T_target))
                heating_powers.append(heating_power)
                
                if abs(T_int - T_target) > 1.0:
                    comfort_violations += 1
                
                observation = next_obs
            
            # Calcul métriques finales
            avg_temp_error = np.mean(temperatures)
            total_energy = sum(heating_powers) * 1e-3  # kWh approximatif
            comfort_rate = comfort_violations / step
            
            results[name] = {
                'reward': total_reward,
                'energy_kwh': total_energy,
                'comfort_violations': comfort_rate,
                'avg_temp_error': avg_temp_error,
                'stability': np.std(heating_powers)
            }
            
            print(f"   Récompense totale: {total_reward:.1f}")
            print(f"   Énergie consommée: {total_energy:.1f} kWh")
            print(f"   Violations confort: {comfort_rate:.1%}")
            print(f"   Erreur temp. moy.: {avg_temp_error:.2f}°C")
            print()
        
        # Visualisation comparative
        self._plot_controller_comparison(results)
        
        return results
    
    def _plot_controller_comparison(self, results: dict):
        """Visualise la comparaison des contrôleurs"""
        controllers = list(results.keys())
        metrics = ['reward', 'energy_kwh', 'comfort_violations', 'avg_temp_error']
        metric_labels = ['Récompense totale', 'Énergie (kWh)', 
                        'Taux violation confort', 'Erreur température (°C)']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [results[ctrl][metric] for ctrl in controllers]
            
            bars = axes[i].bar(controllers, values, color=colors[i], alpha=0.7)
            axes[i].set_title(label)
            axes[i].set_ylabel(label)
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Ajout des valeurs sur les barres
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.2f}', ha='center', va='bottom')
            
            # Rotation labels si nécessaire
            if len(max(controllers, key=len)) > 8:
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('plots/controller_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Tableau récapitulatif
        print("📊 TABLEAU RÉCAPITULATIF")
        print("-" * 80)
        print(f"{'Contrôleur':<15} {'Récompense':<12} {'Énergie(kWh)':<12} {'Confort(%)':<12} {'Temp.Err':<10}")
        print("-" * 80)
        
        for ctrl in controllers:
            r = results[ctrl]
            print(f"{ctrl:<15} {r['reward']:<12.1f} {r['energy_kwh']:<12.1f} "
                  f"{r['comfort_violations']:<12.1%} {r['avg_temp_error']:<10.2f}")
        
        print("-" * 80)
    
    def analyze_thermal_behavior(self):
        """Analyse du comportement thermique détaillé"""
        print("🔬 ANALYSE DU COMPORTEMENT THERMIQUE\n")
        
        # Configuration pour analyse détaillée
        building_params = BuildingParameters(
            C_building=3.6e6,
            U_global=200.0,
            S_window=25.0
        )
        
        building = BuildingThermalModel(building_params)
        weather = SyntheticWeatherProvider(seed=42)
        user = RegularUserModel()
        
        # Simulation sur 48h avec différents scénarios de chauffage
        scenarios = {
            'Constant 50%': lambda h: 2500,  # Puissance constante
            'ON/OFF simple': lambda h: 5000 if h % 4 < 2 else 0,  # Cyclage 2h
            'Adaptatif météo': lambda h: max(0, 5000 - weather._generate_weather_point(
                datetime.now(), 48.8566, 2.3522).temperature * 100)
        }
        
        results = {}
        
        for scenario_name, heating_func in scenarios.items():
            print(f"🔹 Scénario: {scenario_name}")
            
            building.reset(20.0)
            
            times = []
            temperatures = []
            heating_powers = []
            external_temps = []
            energy_consumption = 0
            
            for hour in range(48):
                current_time = datetime.now().replace(hour=hour % 24)
                
                # Conditions externes
                weather_point = weather._generate_weather_point(current_time, 48.8566, 2.3522)
                T_ext = weather_point.temperature
                I_solar = weather_point.solar_irradiance
                
                # Occupation
                is_present = user.is_present(hour % 24)
                Q_occupancy = 150 if is_present else 0
                
                # Chauffage selon scénario
                Q_heating = heating_func(hour)
                
                # Simulation
                state = building.update(
                    dt=3600,
                    T_ext=T_ext,
                    I_solar=I_solar,
                    Q_heating=Q_heating,
                    Q_occupancy=Q_occupancy,
                    wind_speed=weather_point.wind_speed
                )
                
                # Collecte données
                times.append(hour)
                temperatures.append(state['T_int'])
                external_temps.append(T_ext)
                heating_powers.append(Q_heating)
                energy_consumption += building.get_energy_consumption(Q_heating, 3600)
            
            results[scenario_name] = {
                'times': times,
                'temperatures': temperatures,
                'external_temps': external_temps,
                'heating_powers': heating_powers,
                'total_energy': energy_consumption,
                'temp_stability': np.std(temperatures),
                'avg_temp': np.mean(temperatures)
            }
            
            print(f"   Température moyenne: {np.mean(temperatures):.1f}°C")
            print(f"   Stabilité (écart-type): {np.std(temperatures):.2f}°C")
            print(f"   Consommation totale: {energy_consumption:.2f} kWh")
            print(f"   Inertie thermique max: {building.get_thermal_inertia_indicator():.2f}°C")
            print()
        
        # Visualisation
        self._plot_thermal_analysis(results)
        
        return results
    
    def _plot_thermal_analysis(self, results: dict):
        """Visualise l'analyse thermique"""
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, (scenario, data) in enumerate(results.items()):
            color = colors[i % len(colors)]
            
            # Températures
            axes[0].plot(data['times'], data['temperatures'], 
                        label=f'{scenario} (int.)', color=color, linewidth=2)
            if i == 0:  # Température extérieure une seule fois
                axes[0].plot(data['times'], data['external_temps'], 
                           'k--', label='Extérieure', alpha=0.7)
            
            # Puissances de chauffage
            axes[1].plot(data['times'], np.array(data['heating_powers'])/1000, 
                        label=scenario, color=color, linewidth=2)
            
            # Consommation cumulative
            cumulative_energy = np.cumsum([p/1000 for p in data['heating_powers']]) * 1e-3
            axes[2].plot(data['times'], cumulative_energy, 
                        label=scenario, color=color, linewidth=2)
        
        # Configuration des axes
        axes[0].set_ylabel('Température (°C)')
        axes[0].set_title('Évolution des températures')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_ylabel('Puissance (kW)')
        axes[1].set_title('Puissance de chauffage')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_ylabel('Énergie cumulée (kWh)')
        axes[2].set_xlabel('Temps (heures)')
        axes[2].set_title('Consommation énergétique cumulative')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/thermal_behavior_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

def main():
    """Fonction principale avec menu interactif"""
    
    parser = argparse.ArgumentParser(description='Démonstration système thermique intelligent')
    parser.add_argument('--mode', choices=['test', 'train', 'compare', 'analyze', 'all'], 
                       default='all', help='Mode de démonstration')
    args = parser.parse_args()
    
    demo = ThermalSystemDemo()
    
    try:
        if args.mode == 'test' or args.mode == 'all':
            demo.test_components()
        
        if args.mode == 'train' or args.mode == 'all':
            trained_agent = demo.quick_training_demo()
            print()
        
        if args.mode == 'compare' or args.mode == 'all':
            comparison_results = demo.controller_comparison_demo()
            print()
        
        if args.mode == 'analyze' or args.mode == 'all':
            thermal_analysis = demo.analyze_thermal_behavior()
            print()
        
        print("🎉 DÉMONSTRATION TERMINÉE AVEC SUCCÈS !")
        print("\nFichiers générés:")
        print("  📁 plots/ - Graphiques de performance")
        print("  📁 models/ - Modèles sauvegardés")
        print("  📁 results/ - Résultats d'évaluation")
        
        print("\n💡 Pour un entraînement complet:")
        print("  python training_evaluation.py")
        
    except Exception as e:
        print(f"❌ Erreur pendant la démonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())