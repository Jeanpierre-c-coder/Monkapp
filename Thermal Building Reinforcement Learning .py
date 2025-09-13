import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
import json
from datetime import datetime
import os
from collections import deque
import torch

# Import des modules précédents
from thermal_environment import ThermalBuildingEnv, ThermalEnvManager, RewardConfig
from lstm_dqn_agent import LSTMDQNAgent, TrainingConfig
from building_thermal_model import BuildingParameters
from user_models import RegularUserModel, StochasticUserModel
from weather_interface import SyntheticWeatherProvider

class BaselineController:
    """Contrôleurs de référence pour comparaison"""
    
    class SimpleThermostat:
        """Thermostat simple ON/OFF avec hystérésis"""
        
        def __init__(self, hysteresis: float = 0.5):
            self.hysteresis = hysteresis
            self.heating_on = False
        
        def select_action(self, observation: Dict[str, float]) -> int:
            T_int = observation['T_int']
            T_target = observation['T_target']
            
            if not self.heating_on and T_int < T_target - self.hysteresis:
                self.heating_on = True
            elif self.heating_on and T_int > T_target + self.hysteresis:
                self.heating_on = False
            
            return 5 if self.heating_on else 0  # Action max ou min
    
    class PIDController:
        """Contrôleur PID pour régulation thermique"""
        
        def __init__(self, kp: float = 2.0, ki: float = 0.1, kd: float = 0.5):
            self.kp = kp
            self.ki = ki
            self.kd = kd
            
            self.integral = 0.0
            self.previous_error = 0.0
            self.action_history = deque(maxlen=6)  # Pour mapping vers actions discrètes
        
        def select_action(self, observation: Dict[str, float]) -> int:
            T_int = observation['T_int']
            T_target = observation['T_target']
            
            # Calcul de l'erreur
            error = T_target - T_int
            
            # Terme intégral
            self.integral += error
            self.integral = np.clip(self.integral, -10, 10)  # Anti-windup
            
            # Terme dérivé
            derivative = error - self.previous_error
            self.previous_error = error
            
            # Sortie PID
            output = self.kp * error + self.ki * self.integral + self.kd * derivative
            
            # Conversion en action discrète (0-5)
            output_normalized = np.clip(output / 5.0, 0, 1)  # Normalisation
            action = int(output_normalized * 5)
            
            self.action_history.append(action)
            return action
    
    class ScheduledController:
        """Contrôleur avec programmation horaire"""
        
        def __init__(self):
            # Programme type : température selon heure et occupation
            self.schedule = {
                'occupied_day': 21.0,
                'occupied_night': 19.0,
                'absent': 18.0
            }
        
        def select_action(self, observation: Dict[str, float]) -> int:
            T_int = observation['T_int']
            T_target = observation['T_target']
            occupancy = observation['occupancy']
            hour = observation['hour']
            
            # Ajustement cible selon programmation
            if occupancy > 0.5:
                if 6 <= hour <= 22:
                    target_adjusted = self.schedule['occupied_day']
                else:
                    target_adjusted = self.schedule['occupied_night']
            else:
                target_adjusted = self.schedule['absent']
            
            # Contrôle simple basé sur écart
            error = target_adjusted - T_int
            
            if error > 2.0:
                return 5  # Max
            elif error > 1.0:
                return 4
            elif error > 0.5:
                return 3
            elif error > -0.5:
                return 2
            elif error > -1.0:
                return 1
            else:
                return 0  # Off

class ThermalTrainer:
    """Classe principale pour l'entraînement de l'agent thermique"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
        # Environnements
        self.env_manager = ThermalEnvManager()
        self.train_env = None
        self.eval_env = None
        
        # Agent
        self.agent = None
        
        # Métriques d'entraînement
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'energy_consumption': [],
            'comfort_violations': [],
            'losses': [],
            'epsilon_values': [],
            'learning_rates': []
        }
        
        # Résultats d'évaluation
        self.evaluation_results = {}
        
    def _default_config(self) -> Dict:
        """Configuration par défaut"""
        return {
            'training': {
                'episodes': 500,
                'max_steps_per_episode': 168,  # 1 semaine
                'save_frequency': 50,
                'eval_frequency': 25,
                'eval_episodes': 5
            },
            'agent': {
                'learning_rate': 1e-3,
                'batch_size': 32,
                'gamma': 0.95,
                'epsilon_start': 1.0,
                'epsilon_end': 0.01,
                'epsilon_decay': 1000,
                'memory_size': 10000,
                'sequence_length': 24,
                'hidden_size': 128
            },
            'environment': {
                'episode_length_hours': 168,
                'time_step_hours': 1.0,
                'action_type': 'discrete',
                'max_heating_power': 5000.0
            },
            'paths': {
                'models': './models',
                'logs': './logs',
                'results': './results'
            }
        }
    
    def setup_training(self):
        """Initialise l'entraînement"""
        
        # Création des dossiers
        for path in self.config['paths'].values():
            os.makedirs(path, exist_ok=True)
        
        # Environnements
        self.train_env = self.env_manager.create_training_env()
        self.eval_env = self.env_manager.create_evaluation_env()
        
        # Configuration agent
        agent_config = TrainingConfig(**self.config['agent'])
        
        # Calcul de la taille d'état
        dummy_obs = self.train_env.reset()
        state_size = len(dummy_obs)
        action_size = self.train_env.action_space.n
        
        # Création de l'agent
        self.agent = LSTMDQNAgent(state_size, action_size, agent_config)
        
        print(f"=== CONFIGURATION ENTRAÎNEMENT ===")
        print(f"Taille état: {state_size}")
        print(f"Nombre actions: {action_size}")
        print(f"Episodes: {self.config['training']['episodes']}")
        print(f"Device: {self.agent.device}")
    
    def train(self):
        """Boucle principale d'entraînement"""
        
        if self.agent is None:
            self.setup_training()
        
        print(f"=== DÉBUT ENTRAÎNEMENT ===")
        start_time = time.time()
        
        for episode in range(self.config['training']['episodes']):
            episode_start = time.time()
            episode_reward = 0
            episode_steps = 0
            
            # Réinitialisation
            observation = self.train_env.reset()
            done = False
            
            # Épisode
            while not done and episode_steps < self.config['training']['max_steps_per_episode']:
                # Sélection action
                action = self.agent.select_action(observation, training=True)
                
                # Pas dans l'environnement
                next_observation, reward, done, info = self.train_env.step(action)
                
                # Apprentissage
                self.agent.step(observation, action, reward, next_observation, done)
                
                observation = next_observation
                episode_reward += reward
                episode_steps += 1
            
            # Métriques d'épisode
            episode_summary = self.train_env.get_episode_summary()
            
            self.training_metrics['episode_rewards'].append(episode_reward)
            self.training_metrics['episode_lengths'].append(episode_steps)
            self.training_metrics['energy_consumption'].append(
                episode_summary.get('total_energy_consumption_kwh', 0)
            )
            self.training_metrics['comfort_violations'].append(
                episode_summary.get('comfort_violation_rate', 0)
            )
            
            # Métriques agent
            agent_stats = self.agent.get_training_stats()
            self.training_metrics['epsilon_values'].append(agent_stats['epsilon'])
            self.training_metrics['losses'].append(agent_stats['avg_loss_100'])
            
            # Affichage périodique
            if episode % 10 == 0:
                episode_time = time.time() - episode_start
                avg_reward = np.mean(self.training_metrics['episode_rewards'][-10:])
                
                print(f"Episode {episode:4d} | "
                      f"Reward: {episode_reward:6.2f} | "
                      f"Avg10: {avg_reward:6.2f} | "
                      f"Steps: {episode_steps:3d} | "
                      f"ε: {agent_stats['epsilon']:.3f} | "
                      f"Time: {episode_time:.1f}s")
            
            # Sauvegarde périodique
            if episode % self.config['training']['save_frequency'] == 0 and episode > 0:
                self._save_checkpoint(episode)
            
            # Évaluation périodique
            if episode % self.config['training']['eval_frequency'] == 0 and episode > 0:
                self._evaluate_agent(episode)
        
        # Entraînement terminé
        training_time = time.time() - start_time
        print(f"=== ENTRAÎNEMENT TERMINÉ ===")
        print(f"Temps total: {training_time:.1f}s")
        print(f"Temps par épisode: {training_time/self.config['training']['episodes']:.1f}s")
        
        # Sauvegarde finale
        self._save_final_model()
        self._save_training_metrics()
    
    def _save_checkpoint(self, episode: int):
        """Sauvegarde un checkpoint"""
        checkpoint_path = os.path.join(
            self.config['paths']['models'], 
            f'checkpoint_episode_{episode}.pth'
        )
        self.agent.save(checkpoint_path)
        print(f"Checkpoint sauvegardé: {checkpoint_path}")
    
    def _evaluate_agent(self, episode: int):
        """Évalue l'agent sur plusieurs épisodes"""
        print(f"--- Évaluation épisode {episode} ---")
        
        eval_rewards = []
        eval_metrics = {
            'energy_consumption': [],
            'comfort_violations': [],
            'efficiency_scores': []
        }
        
        for eval_ep in range(self.config['training']['eval_episodes']):
            observation = self.eval_env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 720:  # Max 1 mois
                action = self.agent.select_action(observation, training=False)
                observation, reward, done, info = self.eval_env.step(action)
                total_reward += reward
                steps += 1
            
            eval_rewards.append(total_reward)
            
            # Métriques détaillées
            summary = self.eval_env.get_episode_summary()
            eval_metrics['energy_consumption'].append(summary.get('total_energy_consumption_kwh', 0))
            eval_metrics['comfort_violations'].append(summary.get('comfort_violation_rate', 0))
            eval_metrics['efficiency_scores'].append(summary.get('overall_efficiency', 0))
        
        # Résultats moyens
        avg_reward = np.mean(eval_rewards)
        avg_energy = np.mean(eval_metrics['energy_consumption'])
        avg_violations = np.mean(eval_metrics['comfort_violations'])
        avg_efficiency = np.mean(eval_metrics['efficiency_scores'])
        
        print(f"  Récompense moyenne: {avg_reward:.2f}")
        print(f"  Consommation moyenne: {avg_energy:.2f} kWh")
        print(f"  Taux violation confort: {avg_violations:.3f}")
        print(f"  Score d'efficacité: {avg_efficiency:.3f}")
        
        # Stockage pour analyse
        self.evaluation_results[episode] = {
            'avg_reward': avg_reward,
            'avg_energy': avg_energy,
            'avg_violations': avg_violations,
            'avg_efficiency': avg_efficiency,
            'all_rewards': eval_rewards
        }
    
    def _save_final_model(self):
        """Sauvegarde le modèle final"""
        final_path = os.path.join(self.config['paths']['models'], 'final_model.pth')
        self.agent.save(final_path)
        print(f"Modèle final sauvegardé: {final_path}")
    
    def _save_training_metrics(self):
        """Sauvegarde les métriques d'entraînement"""
        metrics_path = os.path.join(self.config['paths']['logs'], 'training_metrics.json')
        
        # Conversion des numpy arrays en listes pour JSON
        metrics_json = {}
        for key, values in self.training_metrics.items():
            if isinstance(values, list) and len(values) > 0:
                if isinstance(values[0], np.ndarray):
                    metrics_json[key] = [v.tolist() for v in values]
                else:
                    metrics_json[key] = values
            else:
                metrics_json[key] = values
        
        with open(metrics_path, 'w') as f:
            json.dump({
                'training_metrics': metrics_json,
                'evaluation_results': self.evaluation_results,
                'config': self.config
            }, f, indent=2)
        
        print(f"Métriques sauvegardées: {metrics_path}")
    
    def plot_training_progress(self):
        """Visualise la progression de l'entraînement"""
        
        if not self.training_metrics['episode_rewards']:
            print("Pas de données d'entraînement à afficher")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        episodes = range(len(self.training_metrics['episode_rewards']))
        
        # Récompenses
        axes[0, 0].plot(episodes, self.training_metrics['episode_rewards'], alpha=0.6)
        if len(self.training_metrics['episode_rewards']) > 20:
            # Moyenne mobile
            window = min(20, len(self.training_metrics['episode_rewards']) // 4)
            moving_avg = pd.Series(self.training_metrics['episode_rewards']).rolling(window).mean()
            axes[0, 0].plot(episodes, moving_avg, 'r-', linewidth=2, label=f'Moyenne mobile ({window})')
            axes[0, 0].legend()
        axes[0, 0].set_title('Récompenses par épisode')
        axes[0, 0].set_xlabel('Épisode')
        axes[0, 0].set_ylabel('Récompense totale')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Consommation énergétique
        axes[0, 1].plot(episodes, self.training_metrics['energy_consumption'])
        axes[0, 1].set_title('Consommation énergétique')
        axes[0, 1].set_xlabel('Épisode')
        axes[0, 1].set_ylabel('Énergie (kWh)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Violations de confort
        axes[0, 2].plot(episodes, self.training_metrics['comfort_violations'])
        axes[0, 2].set_title('Violations de confort')
        axes[0, 2].set_xlabel('Épisode')
        axes[0, 2].set_ylabel('Taux de violation')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Epsilon (exploration)
        axes[1, 0].plot(episodes, self.training_metrics['epsilon_values'])
        axes[1, 0].set_title('Taux d\'exploration (ε)')
        axes[1, 0].set_xlabel('Épisode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Pertes
        if self.training_metrics['losses']:
            valid_losses = [l for l in self.training_metrics['losses'] if l > 0]
            if valid_losses:
                axes[1, 1].plot(valid_losses)
                axes[1, 1].set_title('Perte d\'entraînement')
                axes[1, 1].set_xlabel('Épisode')
                axes[1, 1].set_ylabel('Perte')
                axes[1, 1].set_yscale('log')
                axes[1, 1].grid(True, alpha=0.3)
        
        # Résultats d'évaluation
        if self.evaluation_results:
            eval_episodes = list(self.evaluation_results.keys())
            eval_rewards = [self.evaluation_results[ep]['avg_reward'] for ep in eval_episodes]
            eval_efficiency = [self.evaluation_results[ep]['avg_efficiency'] for ep in eval_episodes]
            
            axes[1, 2].plot(eval_episodes, eval_rewards, 'o-', label='Récompense')
            ax_twin = axes[1, 2].twinx()
            ax_twin.plot(eval_episodes, eval_efficiency, 'r^-', label='Efficacité')
            
            axes[1, 2].set_title('Résultats d\'évaluation')
            axes[1, 2].set_xlabel('Épisode')
            axes[1, 2].set_ylabel('Récompense moyenne')
            ax_twin.set_ylabel('Efficacité')
            axes[1, 2].legend(loc='upper left')
            ax_twin.legend(loc='upper right')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['paths']['results'], 'training_progress.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()

class BenchmarkEvaluator:
    """Évaluateur pour comparer l'agent RL avec les méthodes de référence"""
    
    def __init__(self, trained_agent: LSTMDQNAgent = None):
        self.trained_agent = trained_agent
        self.env_manager = ThermalEnvManager()
        
        # Contrôleurs de référence
        self.baselines = {
            'thermostat': BaselineController.SimpleThermostat(),
            'pid': BaselineController.PIDController(),
            'scheduled': BaselineController.ScheduledController()
        }
        
        if trained_agent:
            self.controllers = {'rl_agent': trained_agent, **self.baselines}
        else:
            self.controllers = self.baselines
    
    def evaluate_all_controllers(self, scenarios: List[str] = None, 
                                episodes_per_scenario: int = 5) -> Dict:
        """Évalue tous les contrôleurs sur différents scénarios"""
        
        scenarios = scenarios or ['normal', 'winter', 'irregular_user', 'poor_insulation']
        results = {}
        
        print("=== ÉVALUATION COMPARATIVE ===")
        
        for scenario in scenarios:
            print(f"\n--- Scénario: {scenario} ---")
            
            # Création environnement spécifique
            env = self._create_scenario_env(scenario)
            scenario_results = {}
            
            for controller_name, controller in self.controllers.items():
                print(f"  Évaluation {controller_name}...")
                
                episode_metrics = []
                
                for episode in range(episodes_per_scenario):
                    metrics = self._run_single_evaluation(env, controller)
                    episode_metrics.append(metrics)
                
                # Moyennes et écarts-types
                avg_metrics = self._aggregate_metrics(episode_metrics)
                scenario_results[controller_name] = avg_metrics
                
                print(f"    Récompense: {avg_metrics['total_reward']:.2f} ± {avg_metrics['reward_std']:.2f}")
                print(f"    Énergie: {avg_metrics['energy_consumption']:.2f} kWh")
                print(f"    Confort: {avg_metrics['comfort_violations']:.3f}")
            
            results[scenario] = scenario_results
        
        return results
    
    def _create_scenario_env(self, scenario: str) -> ThermalBuildingEnv:
        """Crée un environnement selon le scénario"""
        
        if scenario == 'normal':
            return self.env_manager.create_evaluation_env()
        
        elif scenario == 'winter':
            winter_weather = SyntheticWeatherProvider(seed=1)  # Hiver rigoureux
            return self.env_manager.create_env('winter_eval',
                                             weather_provider=winter_weather,
                                             episode_length_hours=720)
        
        elif scenario == 'irregular_user':
            irregular_user = StochasticUserModel(seed=456)
            return self.env_manager.create_env('irregular_eval',
                                             user_model=irregular_user,
                                             episode_length_hours=720)
        
        elif scenario == 'poor_insulation':
            poor_params = BuildingParameters(U_global=400.0, C_building=2.0e6)
            return self.env_manager.create_env('poor_insulation_eval',
                                             building_params=poor_params,
                                             episode_length_hours=720)
        
        else:
            return self.env_manager.create_evaluation_env()
    
    def _run_single_evaluation(self, env: ThermalBuildingEnv, controller) -> Dict:
        """Exécute une évaluation sur un épisode"""
        
        observation = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        # Reset état des contrôleurs si nécessaire
        if hasattr(controller, 'reset'):
            controller.reset()
        
        while not done and steps < 720:  # Max 1 mois
            
            if hasattr(controller, 'select_action'):
                # Contrôleurs baseline
                action = controller.select_action(observation)
            else:
                # Agent RL
                action = controller.select_action(observation, training=False)
            
            observation, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        
        # Métriques finales
        summary = env.get_episode_summary()
        
        return {
            'total_reward': total_reward,
            'energy_consumption': summary.get('total_energy_consumption_kwh', 0),
            'comfort_violations': summary.get('comfort_violation_rate', 0),
            'avg_temperature_error': summary.get('avg_temperature_error_degC', 0),
            'efficiency_score': summary.get('overall_efficiency', 0),
            'peak_power': summary.get('peak_power_kw', 0),
            'steps': steps
        }
    
    def _aggregate_metrics(self, episode_metrics: List[Dict]) -> Dict:
        """Agrège les métriques de plusieurs épisodes"""
        
        if not episode_metrics:
            return {}
        
        keys = episode_metrics[0].keys()
        aggregated = {}
        
        for key in keys:
            values = [metrics[key] for metrics in episode_metrics]
            aggregated[key] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
        
        return aggregated
    
    def plot_comparison(self, results: Dict, save_path: str = None):
        """Visualise la comparaison des contrôleurs"""
        
        if not results:
            print("Pas de résultats à afficher")
            return
        
        scenarios = list(results.keys())
        controllers = list(results[scenarios[0]].keys())
        
        metrics_to_plot = ['total_reward', 'energy_consumption', 'comfort_violations', 'efficiency_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            
            # Données pour le graphique
            scenario_data = []
            for scenario in scenarios:
                controller_values = []
                controller_errors = []
                
                for controller in controllers:
                    if metric in results[scenario][controller]:
                        controller_values.append(results[scenario][controller][metric])
                        std_key = f'{metric}_std'
                        controller_errors.append(results[scenario][controller].get(std_key, 0))
                    else:
                        controller_values.append(0)
                        controller_errors.append(0)
                
                scenario_data.append(controller_values)
            
            # Plot
            x = np.arange(len(scenarios))
            width = 0.8 / len(controllers)
            
            for j, controller in enumerate(controllers):
                values = [scenario_data[s][j] for s in range(len(scenarios))]
                errors = [results[scenarios[s]][controller].get(f'{metric}_std', 0) for s in range(len(scenarios))]
                
                axes[i].bar(x + j * width, values, width, 
                           yerr=errors, capsize=3, label=controller, alpha=0.8)
            
            axes[i].set_xlabel('Scénarios')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].set_title(f'Comparaison - {metric.replace("_", " ").title()}')
            axes[i].set_xticks(x + width * (len(controllers) - 1) / 2)
            axes[i].set_xticklabels(scenarios)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# Script principal d'entraînement et d'évaluation
def main():
    """Script principal pour entraîner et évaluer l'agent"""
    
    print("=== SYSTÈME D'APPRENTISSAGE THERMIQUE INTELLIGENT ===\n")
    
    # Configuration
    config = {
        'training': {
            'episodes': 300,
            'max_steps_per_episode': 168,
            'save_frequency': 50,
            'eval_frequency': 25,
            'eval_episodes': 3
        },
        'agent': {
            'learning_rate': 1e-3,
            'batch_size': 16,
            'sequence_length': 12,
            'memory_size': 5000
        }
    }
    
    # 1. Entraînement
    print("1. PHASE D'ENTRAÎNEMENT")
    trainer = ThermalTrainer(config)
    trainer.train()
    
    # 2. Visualisation progression
    print("\n2. ANALYSE DE LA PROGRESSION")
    trainer.plot_training_progress()
    
    # 3. Évaluation comparative
    print("\n3. ÉVALUATION COMPARATIVE")
    evaluator = BenchmarkEvaluator(trainer.agent)
    
    # Évaluation sur différents scénarios
    benchmark_results = evaluator.evaluate_all_controllers(
        scenarios=['normal', 'winter', 'irregular_user'],
        episodes_per_scenario=3
    )
    
    # Visualisation comparative
    evaluator.plot_comparison(benchmark_results, 
                             save_path='./results/controller_comparison.png')
    
    # 4. Sauvegarde résultats
    print("\n4. SAUVEGARDE DES RÉSULTATS")
    results_path = './results/benchmark_results.json'
    with open(results_path, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"Résultats comparatifs sauvegardés: {results_path}")
    
    # 5. Résumé final
    print("\n=== RÉSUMÉ FINAL ===")
    for scenario, controllers in benchmark_results.items():
        print(f"\nScénario {scenario}:")
        best_controller = max(controllers.keys(), 
                            key=lambda c: controllers[c]['efficiency_score'])
        print(f"  Meilleur contrôleur: {best_controller}")
        print(f"  Score d'efficacité: {controllers[best_controller]['efficiency_score']:.3f}")
        
        if 'rl_agent' in controllers:
            rl_score = controllers['rl_agent']['efficiency_score']
            best_baseline = max([c for c in controllers.keys() if c != 'rl_agent'], 
                              key=lambda c: controllers[c]['efficiency_score'])
            baseline_score = controllers[best_baseline]['efficiency_score']
            improvement = ((rl_score - baseline_score) / baseline_score) * 100
            print(f"  Amélioration RL vs {best_baseline}: {improvement:+.1f}%")

if __name__ == "__main__":
    main()