import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import List, Tuple, Dict, Optional
import pickle
from dataclasses import dataclass

# Structure pour stocker les expériences
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class TrainingConfig:
    """Configuration d'entraînement de l'agent"""
    learning_rate: float = 1e-3
    batch_size: int = 32
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 1000
    target_update: int = 100
    memory_size: int = 10000
    sequence_length: int = 24  # Séquences de 24h pour LSTM
    hidden_size: int = 128
    num_lstm_layers: int = 2

class LSTMThermalController(nn.Module):
    """Réseau LSTM-DQN pour le contrôle thermique intelligent"""
    
    def __init__(self, input_size: int, action_size: int, config: TrainingConfig):
        super(LSTMThermalController, self).__init__()
        
        self.input_size = input_size
        self.action_size = action_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_lstm_layers
        self.sequence_length = config.sequence_length
        
        # Couche d'embedding pour les features d'entrée
        self.input_embedding = nn.Linear(input_size, 64)
        
        # LSTM pour la mémoire temporelle
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.2 if self.num_layers > 1 else 0
        )
        
        # Couches de traitement post-LSTM
        self.feature_processor = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Têtes de sortie pour différents aspects
        self.value_head = nn.Linear(128, 1)  # Value function
        self.advantage_head = nn.Linear(128, action_size)  # Advantage function
        
        # Tête auxiliaire pour prédiction de température (apprentissage multi-tâche)
        self.temp_prediction_head = nn.Linear(128, 1)
        
    def forward(self, x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass du réseau
        
        Args:
            x: tensor de shape (batch_size, sequence_length, input_size)
            hidden: état caché LSTM optionnel
            
        Returns:
            Dict contenant q_values, temp_prediction, hidden_state
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Embedding des features
        x_embedded = F.relu(self.input_embedding(x))
        
        # LSTM
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        lstm_out, hidden = self.lstm(x_embedded, hidden)
        
        # Utiliser la dernière sortie de la séquence
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Traitement des features
        features = self.feature_processor(last_output)
        
        # Dueling DQN: V(s) + A(s,a) - mean(A(s,a))
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Prédiction auxiliaire de température
        temp_prediction = self.temp_prediction_head(features)
        
        return {
            'q_values': q_values,
            'temp_prediction': temp_prediction,
            'hidden_state': hidden,
            'features': features
        }
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialise l'état caché LSTM"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)

class PrioritizedReplayBuffer:
    """Buffer de replay avec priorité pour améliorer l'apprentissage"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priorité
        self.beta = beta    # Importance sampling
        self.beta_increment = 0.001
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
        
    def add(self, experience: Experience, td_error: float = None):
        """Ajoute une expérience avec sa priorité"""
        if td_error is None:
            priority = self.max_priority
        else:
            priority = (abs(td_error) + 1e-6) ** self.alpha
        
        self.buffer.append(experience)
        self.priorities.append(priority)
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int, sequence_length: int) -> Tuple[List[Experience], np.ndarray, List[int]]:
        """Échantillonne des séquences selon les priorités"""
        if len(self.buffer) < sequence_length:
            return [], np.array([]), []
        
        # Calcul des probabilités de sélection
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Échantillonnage
        indices = []
        experiences = []
        weights = []
        
        for _ in range(batch_size):
            # Sélection d'un index de fin de séquence
            end_idx = np.random.choice(len(self.buffer), p=probs)
            start_idx = max(0, end_idx - sequence_length + 1)
            
            # Extraction de la séquence
            sequence = []
            for i in range(start_idx, end_idx + 1):
                if i < len(self.buffer):
                    sequence.append(self.buffer[i])
            
            # Padding si nécessaire
            while len(sequence) < sequence_length:
                sequence.insert(0, sequence[0])  # Répète le premier élément
            
            indices.append(end_idx)
            experiences.append(sequence)
            
            # Calcul du poids d'importance
            prob = probs[end_idx]
            weight = (len(self.buffer) * prob) ** (-self.beta)
            weights.append(weight)
        
        # Normalisation des poids
        weights = np.array(weights)
        weights = weights / weights.max()
        
        # Mise à jour de beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, weights, indices
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Met à jour les priorités basées sur les erreurs TD"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

class ThermalStateProcessor:
    """Processeur pour normaliser et enrichir l'état du système thermique"""
    
    def __init__(self):
        # Statistiques pour normalisation (à mettre à jour avec données réelles)
        self.state_stats = {
            'T_int': {'mean': 20.0, 'std': 3.0, 'min': 10.0, 'max': 30.0},
            'T_ext': {'mean': 15.0, 'std': 10.0, 'min': -20.0, 'max': 45.0},
            'T_target': {'mean': 20.0, 'std': 2.0, 'min': 16.0, 'max': 24.0},
            'solar_gains': {'mean': 200.0, 'std': 300.0, 'min': 0.0, 'max': 1000.0},
            'occupancy': {'mean': 0.3, 'std': 0.46, 'min': 0.0, 'max': 1.0},
            'time_features': {'mean': 0.5, 'std': 0.29, 'min': 0.0, 'max': 1.0}
        }
    
    def process_state(self, raw_state: Dict[str, float]) -> np.ndarray:
        """
        Traite et normalise l'état brut
        
        Args:
            raw_state: Dict avec les variables d'état brutes
            
        Returns:
            État normalisé sous forme de vecteur numpy
        """
        processed_features = []
        
        # Températures
        T_int = self._normalize(raw_state['T_int'], 'T_int')
        T_ext = self._normalize(raw_state['T_ext'], 'T_ext')
        T_target = self._normalize(raw_state['T_target'], 'T_target')
        
        processed_features.extend([T_int, T_ext, T_target])
        
        # Erreur de température et ses dérivées
        temp_error = T_int - T_target
        temp_error_abs = abs(temp_error)
        processed_features.extend([temp_error, temp_error_abs])
        
        # Prévisions météo (6h, 12h)
        T_ext_6h = self._normalize(raw_state.get('T_ext_forecast_6h', T_ext), 'T_ext')
        T_ext_12h = self._normalize(raw_state.get('T_ext_forecast_12h', T_ext), 'T_ext')
        processed_features.extend([T_ext_6h, T_ext_12h])
        
        # Tendances
        temp_ext_trend = T_ext_6h - T_ext
        processed_features.append(temp_ext_trend)
        
        # Apports solaires
        solar = self._normalize(raw_state.get('solar_gains', 0), 'solar_gains')
        processed_features.append(solar)
        
        # Occupation
        occupancy = raw_state.get('occupancy', 0)
        occupancy_forecast_2h = raw_state.get('occupancy_forecast_2h', occupancy)
        processed_features.extend([occupancy, occupancy_forecast_2h])
        
        # Features temporelles cycliques
        hour = raw_state.get('hour', 12)
        day_of_week = raw_state.get('day_of_week', 3)
        
        # Encodage cyclique du temps
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)
        
        processed_features.extend([hour_sin, hour_cos, day_sin, day_cos])
        
        # État du système de chauffage
        heating_power_prev = self._normalize(raw_state.get('heating_power_prev', 0), 
                                           {'mean': 0.5, 'std': 0.3, 'min': 0.0, 'max': 1.0})
        energy_24h = self._normalize(raw_state.get('energy_consumed_24h', 50), 
                                   {'mean': 100, 'std': 50, 'min': 0, 'max': 300})
        
        processed_features.extend([heating_power_prev, energy_24h])
        
        # Indicateurs de confort et efficacité
        comfort_violation = 1.0 if temp_error_abs > 0.1 else 0.0  # Violation confort
        efficiency_indicator = max(0, 1 - energy_24h)  # Plus c'est efficace, plus c'est proche de 1
        
        processed_features.extend([comfort_violation, efficiency_indicator])
        
        return np.array(processed_features, dtype=np.float32)
    
    def _normalize(self, value: float, feature_name: str) -> float:
        """Normalise une valeur selon ses statistiques"""
        if isinstance(feature_name, dict):
            stats = feature_name
        else:
            stats = self.state_stats.get(feature_name, {'mean': 0, 'std': 1, 'min': -1, 'max': 1})
        
        # Normalisation min-max suivie d'une standardisation
        normalized = (value - stats['min']) / (stats['max'] - stats['min'])
        normalized = np.clip(normalized, 0, 1)  # Assure [0,1]
        
        # Centrage et réduction
        standardized = (normalized - 0.5) * 2  # Ramène à [-1, 1]
        
        return standardized

class LSTMDQNAgent:
    """Agent LSTM-DQN pour le contrôle thermique intelligent"""
    
    def __init__(self, state_size: int, action_size: int, config: TrainingConfig):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation du device: {self.device}")
        
        # Réseaux de neurones
        self.q_network = LSTMThermalController(state_size, action_size, config).to(self.device)
        self.target_network = LSTMThermalController(state_size, action_size, config).to(self.device)
        
        # Optimiseur
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)
        
        # Replay buffer avec priorités
        self.memory = PrioritizedReplayBuffer(config.memory_size)
        
        # Processeur d'état
        self.state_processor = ThermalStateProcessor()
        
        # Variables d'exploration
        self.epsilon = config.epsilon_start
        self.epsilon_decay_rate = (config.epsilon_start - config.epsilon_end) / config.epsilon_decay
        
        # Variables d'entraînement
        self.steps_done = 0
        self.episode_rewards = []
        self.losses = []
        self.q_values_history = []
        
        # État LSTM
        self.hidden_state = None
        self.state_sequence = deque(maxlen=config.sequence_length)
        
        # Copie initiale des poids vers le réseau target
        self.update_target_network()
    
    def get_state_size(self) -> int:
        """Retourne la taille du vecteur d'état traité"""
        dummy_state = {
            'T_int': 20.0, 'T_ext': 15.0, 'T_target': 21.0,
            'T_ext_forecast_6h': 14.0, 'T_ext_forecast_12h': 13.0,
            'solar_gains': 200.0, 'occupancy': 1.0, 'occupancy_forecast_2h': 1.0,
            'hour': 12, 'day_of_week': 3,
            'heating_power_prev': 0.5, 'energy_consumed_24h': 80.0
        }
        return len(self.state_processor.process_state(dummy_state))
    
    def select_action(self, state: Dict[str, float], training: bool = True) -> int:
        """
        Sélection d'action avec exploration epsilon-greedy
        
        Args:
            state: État du système
            training: Mode entraînement ou évaluation
            
        Returns:
            Action sélectionnée (index)
        """
        processed_state = self.state_processor.process_state(state)
        self.state_sequence.append(processed_state)
        
        # Compléter la séquence si nécessaire
        while len(self.state_sequence) < self.config.sequence_length:
            self.state_sequence.appendleft(processed_state)
        
        if training and random.random() < self.epsilon:
            # Exploration aléatoire
            return random.randrange(self.action_size)
        
        # Exploitation : sélection greedy
        with torch.no_grad():
            state_tensor = torch.FloatTensor([list(self.state_sequence)]).to(self.device)
            
            output = self.q_network(state_tensor, self.hidden_state)
            q_values = output['q_values']
            self.hidden_state = output['hidden_state']
            
            # Sauvegarde pour analyse
            if training:
                self.q_values_history.append(q_values.cpu().numpy().flatten())
            
            return q_values.argmax().item()
    
    def step(self, state: Dict[str, float], action: int, reward: float, 
             next_state: Dict[str, float], done: bool):
        """
        Enregistre une expérience et déclenche l'apprentissage
        
        Args:
            state, action, reward, next_state, done: Transition SARS'
        """
        # Traitement des états
        processed_state = self.state_processor.process_state(state)
        processed_next_state = self.state_processor.process_state(next_state)
        
        # Stockage de l'expérience
        experience = Experience(processed_state, action, reward, processed_next_state, done)
        
        # Calcul approximatif de l'erreur TD pour la priorité
        with torch.no_grad():
            state_tensor = torch.FloatTensor([processed_state]).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor([processed_next_state]).unsqueeze(0).to(self.device)
            
            current_q = self.q_network(state_tensor)['q_values'][0, action]
            next_q = self.target_network(next_state_tensor)['q_values'].max(1)[0]
            target_q = reward + (self.config.gamma * next_q * (1 - done))
            td_error = abs(current_q.item() - target_q.item())
        
        self.memory.add(experience, td_error)
        
        # Apprentissage
        if len(self.memory) >= self.config.batch_size:
            self.learn()
        
        # Mise à jour epsilon
        if self.epsilon > self.config.epsilon_end:
            self.epsilon = max(self.config.epsilon_end, 
                             self.epsilon - self.epsilon_decay_rate)
        
        self.steps_done += 1
        
        # Mise à jour du réseau target
        if self.steps_done % self.config.target_update == 0:
            self.update_target_network()
    
    def learn(self):
        """Apprentissage à partir d'un batch d'expériences"""
        # Échantillonnage du buffer
        experiences, weights, indices = self.memory.sample(
            self.config.batch_size, self.config.sequence_length
        )
        
        if not experiences:
            return
        
        # Préparation des tensors
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for sequence in experiences:
            # Extraction des séquences d'états
            seq_states = [exp.state for exp in sequence]
            seq_next_states = [exp.next_state for exp in sequence]
            
            states.append(seq_states)
            next_states.append(seq_next_states)
            
            # Utilise la dernière transition de la séquence pour action/reward/done
            last_exp = sequence[-1]
            actions.append(last_exp.action)
            rewards.append(last_exp.reward)
            dones.append(last_exp.done)
        
        # Conversion en tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.BoolTensor(dones).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        
        # Forward pass
        current_output = self.q_network(states_tensor)
        current_q_values = current_output['q_values'].gather(1, actions_tensor.unsqueeze(1))
        
        # Double DQN : utilise le réseau principal pour sélectionner, target pour évaluer
        with torch.no_grad():
            next_output = self.q_network(next_states_tensor)
            next_actions = next_output['q_values'].argmax(1)
            
            target_output = self.target_network(next_states_tensor)
            next_q_values = target_output['q_values'].gather(1, next_actions.unsqueeze(1))
            
            target_q_values = rewards_tensor.unsqueeze(1) + \
                            (self.config.gamma * next_q_values * (~dones_tensor.unsqueeze(1)))
        
        # Perte principale (DQN)
        dqn_loss = F.mse_loss(current_q_values, target_q_values, reduction='none')
        dqn_loss = (dqn_loss.squeeze() * weights_tensor).mean()
        
        # Perte auxiliaire (prédiction de température)
        if 'temp_prediction' in current_output:
            # Approximation : température cible comme vraie valeur
            temp_targets = torch.FloatTensor([s[-1][2] for s in states]).to(self.device)  # T_target normalisé
            temp_loss = F.mse_loss(current_output['temp_prediction'].squeeze(), temp_targets)
        else:
            temp_loss = 0
        
        # Perte totale
        total_loss = dqn_loss + 0.1 * temp_loss
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping pour stabilité
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Mise à jour des priorités
        td_errors = (current_q_values - target_q_values).abs().detach().cpu().numpy().flatten()
        self.memory.update_priorities(indices, td_errors)
        
        # Sauvegarde des métriques
        self.losses.append(total_loss.item())
    
    def update_target_network(self):
        """Copie les poids du réseau principal vers le réseau target"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath: str):
        """Sauvegarde l'agent"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'config': self.config,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses
        }, filepath)
    
    def load(self, filepath: str):
        """Charge un agent sauvegardé"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.episode_rewards = checkpoint['episode_rewards']
        self.losses = checkpoint['losses']
    
    def get_training_stats(self) -> Dict[str, float]:
        """Retourne les statistiques d'entraînement"""
        return {
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'memory_size': len(self.memory),
            'avg_reward_100': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else 0,
            'avg_loss_100': np.mean(self.losses[-100:]) if len(self.losses) >= 100 else 0,
            'avg_q_value': np.mean([np.mean(q) for q in self.q_values_history[-100:]]) if self.q_values_history else 0
        }

# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration
    config = TrainingConfig(
        learning_rate=1e-3,
        batch_size=16,
        sequence_length=12,
        memory_size=5000
    )
    
    # Création de l'agent
    # Calcule automatiquement la taille de l'état
    dummy_state = {
        'T_int': 20.5,
        'T_ext': 5.0,
        'T_target': 21.0,
        'T_ext_forecast_6h': 4.0,
        'T_ext_forecast_12h': 3.5,
        'solar_gains': 150.0,
        'occupancy': 1.0,
        'occupancy_forecast_2h': 1.0,
        'hour': 14,
        'day_of_week': 2,
        'heating_power_prev': 0.4,
        'energy_consumed_24h': 85.0
    }
    state_size = ThermalStateProcessor().process_state(dummy_state).shape[0]
    action_size = 6  # 6 niveaux de puissance de chauffage
    
    agent = LSTMDQNAgent(state_size, action_size, config)
    
    # Test de traitement d'état
    test_state = {
        'T_int': 20.5,
        'T_ext': 5.0,
        'T_target': 21.0,
        'T_ext_forecast_6h': 4.0,
        'T_ext_forecast_12h': 3.5,
        'solar_gains': 150.0,
        'occupancy': 1.0,
        'occupancy_forecast_2h': 1.0,
        'hour': 14,
        'day_of_week': 2,
        'heating_power_prev': 0.4,
        'energy_consumed_24h': 85.0
    }
    
    processed = agent.state_processor.process_state(test_state)
    print(f"État original: {len(test_state)} features")
    print(f"État traité: {len(processed)} features")
    print(f"Exemple d'état traité: {processed[:5]}...")
    
    # Test de sélection d'action
    action = agent.select_action(test_state, training=True)
    print(f"Action sélectionnée: {action}")
    
    # Simulation d'une transition
    next_state = test_state.copy()
    next_state['T_int'] = 20.7
    reward = -0.5  # Exemple de récompense
    
    agent.step(test_state, action, reward, next_state, False)
    
    print(f"Statistiques: {agent.get_training_stats()}")