
import numpy as np
import math
from gym import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback


from simulation import AdvancedRocket


class RocketEnv(Env):
    
    def __init__(self, rocket=None, max_steps=15000):
        super(RocketEnv, self).__init__()
        
        # Используем переданную ракету или создаём новую
        self.rocket = rocket if rocket else AdvancedRocket()
        self.max_steps = max_steps
        self.step_count = 0
        
        # Пространство наблюдений: [ошибка тангажа, скорость тангажа, высота, скорость]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        
        # Пространство действий: [команда руля высоты] от -1 до 1
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # История для анализа
        self.history = {
            'pitch_error': [],
            'altitude': [],
            'velocity': [],
            'action': [],
            'reward': []
        }
    
    def reset(self):
        
        self.rocket.__init__()  # Полный сброс ракеты
        self.step_count = 0
        self.history = {key: [] for key in self.history}
        
        # Начинаем с небольшого случайного возмущения для разнообразия
        self.rocket.omega[1] = np.random.uniform(-0.05, 0.05)
        
        return self._get_observation()
    
    def _get_observation(self):
        """Получение текущего состояния"""
        pitch, _, _ = self.rocket.get_euler_angles()
        
        # Целевой угол (программа полёта)
        target_pitch = math.radians(self.rocket.target_pitch)
        
        # Ошибка по тангажу (нормализованная)
        pitch_error = (target_pitch - pitch) / math.radians(30)  # Нормализация
        
        return np.array([
            pitch_error,
            self.rocket.omega[1] / 2.0,  # Угловая скорость (нормализованная)
            self.rocket.pos[2] / 10000.0,  # Высота (нормализованная)
            np.linalg.norm(self.rocket.vel) / 500.0  # Скорость (нормализованная)
        ], dtype=np.float32)
    
    def step(self, action):
        """Выполнение шага симуляции"""
        # Применяем действие (команда руля высоты)
        elevator_cmd = np.clip(action[0], -1.0, 1.0)
        self.rocket.elevator_cmd = elevator_cmd
        
        # Выполняем шаг физики (10 шагов для стабильности)
        for _ in range(10):
            self.rocket.physics_update(0.02)
        
        # Получаем текущее состояние
        pitch, _, _ = self.rocket.get_euler_angles()
        target_pitch = math.radians(self.rocket.target_pitch)
        pitch_error = abs(target_pitch - pitch)
        
        # Расчёт награды
        reward = self._calculate_reward(pitch_error, elevator_cmd)
        
        # Проверка завершения эпизода
        done = self._check_done()
        
        # Сохраняем историю
        self.history['pitch_error'].append(pitch_error)
        self.history['altitude'].append(self.rocket.pos[2])
        self.history['velocity'].append(np.linalg.norm(self.rocket.vel))
        self.history['action'].append(elevator_cmd)
        self.history['reward'].append(reward)
        
        self.step_count += 1
        
        return self._get_observation(), reward, done, {}
    
    def _calculate_reward(self, pitch_error, action):
        """Расчёт награды для агента"""
        # Основная цель: минимизировать ошибку по тангажу
        pitch_reward = -pitch_error * 5.0
        
        # Штраф за резкие движения руля (плавность)
        smoothness_penalty = -0.1 * abs(action)
        
        # Бонус за набор высоты
        altitude_bonus = self.rocket.pos[2] / 10000.0
        
        # Штраф за падение
        if self.rocket.pos[2] < 0:
            return -100.0
        
        # Бонус за успешное завершение миссии
        if self.rocket.mission_complete:
            return 200.0
        
        return pitch_reward + smoothness_penalty + altitude_bonus * 0.5
    
    def _check_done(self):
        """Проверка условий завершения эпизода"""
        # Падение
        if self.rocket.pos[2] < 0:
            return True
        
        # Превышение максимальной длины эпизода
        if self.step_count >= self.max_steps:
            return True
        
        # Успешное завершение миссии
        if self.rocket.mission_complete:
            return True
        
        # Потеря управления (слишком большая ошибка)
        pitch, _, _ = self.rocket.get_euler_angles()
        if abs(pitch) > math.radians(45):
            return True
        
        return False
    
    def render(self, mode='human'):
        """Визуализация (опционально)"""
        pass


def train_rl_controller(total_timesteps=200000):
   
    print("🚀 Начинаем обучение RL-агента...")
    
    # Создаём среду
    env = DummyVecEnv([lambda: RocketEnv()])
    
    
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./rl_logs/"
    )
    
    # Callback для оценки
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Обучение
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Сохраняем модель
    model.save("models/rl_controller_final")
    print("✅ Обучение завершено! Модель сохранена.")
    
    return model


def load_rl_controller(model_path="models/rl_controller_final"):
    """Загрузка обученного RL-контроллера"""
    try:
        model = PPO.load(model_path)
        print("✅ RL-модель загружена успешно!")
        return model
    except:
        print("⚠️ Модель не найдена. Сначала обучите агента.")
        return None


class RLController:
    """
    Класс-обёртка для использования RL-агента в симуляции
    """
    
    def __init__(self, model_path="models/rl_controller_final"):
        self.model = load_rl_controller(model_path)
        self.observation = None
        self.action = 0.0
    
    def calculate(self, rocket, dt):
        """Расчёт управляющего сигнала с помощью RL"""
        if self.model is None:
            return 0.0
        
        # Получаем наблюдения
        pitch, _, _ = rocket.get_euler_angles()
        target_pitch = math.radians(rocket.target_pitch)
        
        observation = np.array([
            (target_pitch - pitch) / math.radians(30),
            rocket.omega[1] / 2.0,
            rocket.pos[2] / 10000.0,
            np.linalg.norm(rocket.vel) / 500.0
        ], dtype=np.float32)
        
        # Предсказание действия
        action, _ = self.model.predict(observation, deterministic=True)
        self.action = np.clip(action[0], -1.0, 1.0)
        
        return self.action


# Пример использования
if __name__ == "__main__":
    # Обучение агента
    model = train_rl_controller(total_timesteps=100000)
    
    # Тестирование
    env = RocketEnv()
    obs = env.reset()
    
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        
        if done:
            break
    
    print("✅ Тестирование завершено!")
