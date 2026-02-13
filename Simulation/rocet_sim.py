# advanced_rocket_simulator.py
import pygame
import numpy as np
import math
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from datetime import datetime
from scipy import signal

pygame.init()

# Константы экрана
WIDTH, HEIGHT = 1400, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ПРОФЕССИОНАЛЬНЫЙ СИМУЛЯТОР: ГОРИЗОНТАЛЬНЫЙ СТАРТ С КАСКАДНЫМ ПИД")

# Цвета (те же, что и в оригинале)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_BLUE = (15, 20, 35)
PANEL_GRAY = (40, 45, 60)
PANEL_BORDER = (70, 75, 90)
BLUE = (0, 150, 255)
GREEN = (0, 200, 100)
YELLOW = (255, 200, 0)
ORANGE = (255, 140, 0)
RED = (255, 60, 60)
CYAN = (0, 200, 220)
PURPLE = (180, 100, 220)

# Нормализованные цвета для Matplotlib
BLUE_NORM = (0/255, 150/255, 255/255)
GREEN_NORM = (0/255, 200/255, 100/255)
YELLOW_NORM = (255/255, 200/255, 0/255)
ORANGE_NORM = (255/255, 140/255, 0/255)
RED_NORM = (255/255, 60/255, 60/255)
CYAN_NORM = (0/255, 200/255, 220/255)
PURPLE_NORM = (180/255, 100/255, 220/255)
WHITE_NORM = (1.0, 1.0, 1.0)
DARK_BLUE_NORM = (15/255, 20/255, 35/255)


class CascadePID:
    """
    Каскадный ПИД-регулятор с двумя контурами:
    - Внешний контур: управление углом (медленный)
    - Внутренний контур: управление угловой скоростью (быстрый)
    """
    
    def __init__(self, 
                 Kp_angle=2.0, Ki_angle=0.01, Kd_angle=0.5,
                 Kp_rate=1.5, Ki_rate=0.05, Kd_rate=0.1,
                 max_output=1.0, max_rate=2.0, dt=0.02):
        
        # Внешний контур (угол)
        self.angle_pid = PID(Kp=Kp_angle, Ki=Ki_angle, Kd=Kd_angle, 
                             max_output=max_rate, dt=dt)
        
        # Внутренний контур (угловая скорость)
        self.rate_pid = PID(Kp=Kp_rate, Ki=Ki_rate, Kd=Kd_rate,
                           max_output=max_output, dt=dt)
        
        self.max_rate = max_rate
        self.target_rate = 0.0
        self.dt = dt
        
        # Для анализа
        self.angle_error = 0.0
        self.rate_error = 0.0
        self.angle_output = 0.0
        self.rate_output = 0.0
        
    def calculate(self, angle_error, current_rate, dt):
        """
        Расчет управляющего сигнала
        angle_error: ошибка по углу (рад)
        current_rate: текущая угловая скорость (рад/с)
        """
        # Внешний контур: ошибка угла -> желаемая угловая скорость
        self.target_rate = self.angle_pid.calculate(angle_error, dt)
        self.target_rate = np.clip(self.target_rate, -self.max_rate, self.max_rate)
        self.angle_output = self.target_rate
        self.angle_error = angle_error
        
        # Внутренний контур: ошибка угловой скорости -> сигнал управления
        rate_error = self.target_rate - current_rate
        output = self.rate_pid.calculate(rate_error, dt)
        self.rate_output = output
        self.rate_error = rate_error
        
        return output
    
    def reset(self):
        """Сброс регулятора"""
        self.angle_pid.reset()
        self.rate_pid.reset()
        self.target_rate = 0.0


class PID:
    """Класс ПИД-регулятора с фильтром производной"""
    
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, max_output=1.0, dt=0.02):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_output = max_output
        self.dt = dt
        
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_output = 0.0
        
        # Фильтр для производной
        self.deriv_filter = 0.0
        self.filter_coeff = 0.1  # Коэффициент фильтрации
        
        # Антивиндовер
        self.integral_limit = max_output / (Ki + 1e-6) if Ki > 0 else 0
        
        # Для анализа
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0
        
    def calculate(self, error, dt):
        self.dt = dt
        
        # Пропорциональная составляющая
        proportional = self.Kp * error
        self.last_p = proportional
        
        # Интегральная составляющая с антивиндовером
        self.integral += error * dt
        if abs(self.integral) > self.integral_limit:
            self.integral = np.sign(self.integral) * self.integral_limit
        integral_term = self.Ki * self.integral
        self.last_i = integral_term
        
        # Дифференциальная составляющая с фильтром
        if dt > 0:
            raw_derivative = (error - self.previous_error) / dt
            # Фильтр низких частот
            self.deriv_filter = (1 - self.filter_coeff) * self.deriv_filter + \
                                 self.filter_coeff * raw_derivative
            derivative_term = self.Kd * self.deriv_filter
        else:
            derivative_term = 0.0
        self.last_d = derivative_term
        
        # Суммарный сигнал
        output = proportional + integral_term + derivative_term
        
        # Ограничение с антивиндовером через обратное вычисление
        if abs(output) > self.max_output:
            output = np.sign(output) * self.max_output
            # Антивиндовер: останавливаем интегратор если он увеличивает ошибку
            if abs(integral_term) > abs(proportional + derivative_term):
                self.integral -= error * dt * 0.5  # Частичный откат
        
        # Сохранение состояния
        self.previous_error = error
        self.previous_output = output
        
        return output
    
    def reset(self):
        """Сброс регулятора"""
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_output = 0.0
        self.deriv_filter = 0.0


class AdvancedRocket:
    """Усовершенствованная модель ракеты с горизонтальным взлетом"""
    
    def __init__(self):
        # ФИЗИЧЕСКИЕ ПАРАМЕТРЫ
        self.mass = 25000.0  # кг
        self.length = 32.4  # м
        self.diameter = 2.95  # м
        
        # Моменты инерции
        self.Ixx = self.mass * self.diameter**2 / 4
        self.Iyy = self.mass * (3 * self.diameter**2 + self.length**2) / 12
        self.Izz = self.Iyy
        
        # Аэродинамика
        self.S_ref = math.pi * (self.diameter/2)**2
        self.Cd0 = 0.25  # Базовое сопротивление
        self.Cd_alpha = 0.5  # Прирост сопротивления от угла атаки
        self.Cl_alpha = 3.0  # Подъемная сила
        self.Cm_alpha = -0.5  # Момент от угла атаки (продольная устойчивость)
        self.Cn_beta = -0.3  # Момент от угла скольжения (путевая устойчивость)
        
        # Двигательная установка
        self.thrust_max = 250000.0  # Н
        self.mass_flow = 1000.0  # кг/с
        self.isp = 800.0  # с
        
        # Шасси и колеса
        self.wheel_friction = 0.03  # Коэффициент трения качения
        self.brake_force = 50000.0  # Сила тормозов
        self.nose_gear_angle = 0.0  # Угол поворота передней стойки
        self.max_steering_angle = math.radians(30)  # Макс. угол поворота колес
        
        # Система управления
        self.max_elevator = math.radians(25)  # Руль высоты
        self.max_rudder = math.radians(20)    # Руль направления
        self.max_aileron = math.radians(15)    # Элероны
        
        # СОСТОЯНИЕ СИСТЕМЫ
        self.pos = np.array([0.0, 0.0, 0.0])  # x, y, z (м)
        self.vel = np.array([0.0, 0.0, 0.0])  # м/с
        self.accel = np.array([0.0, 0.0, 0.0])  # м/с²
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # Кватернион
        self.omega = np.array([0.0, 0.0, 0.0])  # Угл. скорость (рад/с)
        
        # УПРАВЛЕНИЕ
        self.throttle = 0.0
        self.elevator = 0.0  # Руль высоты
        self.rudder = 0.0     # Руль направления
        self.aileron = 0.0    # Элероны
        self.steering = 0.0   # Управление передним колесом
        self.brakes = 0.0     # Тормоза
        
        # ЦЕЛЕВАЯ ТРАЕКТОРИЯ ДЛЯ ГОРИЗОНТАЛЬНОГО ВЗЛЕТА
        self.generate_horizontal_takeoff_trajectory()
        
        # ТЕКУЩИЕ ЦЕЛИ
        self.target_pitch = 0.0
        self.target_yaw = 0.0
        self.target_roll = 0.0
        self.target_altitude = 100000.0  # 100 км
        self.target_velocity = 90.0  # Целевая скорость отрыва
        
        # КАСКАДНЫЕ ПИД-РЕГУЛЯТОРЫ
        self.pitch_cascade = CascadePID(
            Kp_angle=3.0, Ki_angle=0.05, Kd_angle=0.8,    # Внешний контур
            Kp_rate=2.0, Ki_rate=0.1, Kd_rate=0.3,        # Внутренний контур
            max_output=1.0, max_rate=1.5, dt=0.02
        )
        
        self.yaw_cascade = CascadePID(
            Kp_angle=2.5, Ki_angle=0.03, Kd_angle=0.6,
            Kp_rate=1.5, Ki_rate=0.08, Kd_rate=0.2,
            max_output=1.0, max_rate=1.2, dt=0.02
        )
        
        self.roll_cascade = CascadePID(
            Kp_angle=2.0, Ki_angle=0.02, Kd_angle=0.4,
            Kp_rate=1.2, Ki_rate=0.05, Kd_rate=0.15,
            max_output=1.0, max_rate=1.0, dt=0.02
        )
        
        # Классические ПИД для сравнения (опционально)
        self.pitch_pid_classic = PID(Kp=0.9, Ki=0.0, Kd=1.7, max_output=1.0)
        self.yaw_pid_classic = PID(Kp=0.6, Ki=0.015, Kd=0.12, max_output=1.0)
        self.roll_pid_classic = PID(Kp=0.4, Ki=0.01, Kd=0.08, max_output=1.0)
        
        self.use_cascade = True  # Переключатель между каскадным и классическим
        
        # ДАННЫЕ
        self.time = 0.0
        self.dt = 0.02
        self.simulation_speed = 1.0
        self.mission_time = 0.0
        
        # ТЕЛЕМЕТРИЯ
        self.telemetry = {
            'time': [], 'altitude': [], 'velocity': [], 'mach': [],
            'pitch': [], 'yaw': [], 'roll': [], 'alpha': [], 'beta': [],
            'thrust': [], 'mass': [], 'q_dyn': [], 'accel': [],
            'pitch_error': [], 'yaw_error': [], 'roll_error': [],
            'elevator': [], 'rudder': [], 'aileron': [],
            'throttle': [], 'steering': [], 'brakes': [],
            'pitch_rate': [], 'yaw_rate': [], 'roll_rate': [],
            'ground_contact': [], 'aoa': [], 'slip': []
        }
        
        # РЕЖИМ РАБОТЫ
        self.mode = "PRELAUNCH"
        self.events = []
        self.event_times = []
        
        # ПАРАМЕТРЫ ГОРИЗОНТАЛЬНОГО ВЗЛЕТА
        self.on_ground = True
        self.gear_down = True
        self.runway_length = 3000.0
        self.runway_width = 60.0
        self.runway_heading = 0.0  # Направление ВПП (град)
        self.takeoff_velocity = 90.0  # м/с
        self.rotation_velocity = 70.0  # м/с - скорость подъема носа
        self.v1_velocity = 60.0  # м/с - скорость принятия решения
        self.vr_velocity = 75.0  # м/с - скорость подъема передней стойки
        self.v2_velocity = 85.0  # м/с - безопасная скорость взлета
        
        # ВИЗУАЛИЗАЦИЯ
        self.trajectory_points = []
        self.max_trajectory_points = 500
        self.show_trajectory = True
        self.show_target_path = True
        
        # ГРАФИКИ
        self.figures = {}
        self.graph_images = {}
        self.init_figures()
        
        # ИСТОРИЯ ДЛЯ АНАЛИЗА
        self.control_history = {
            'time': [],
            'pitch_error': [], 'yaw_error': [], 'roll_error': [],
            'pitch_rate_error': [], 'yaw_rate_error': [], 'roll_rate_error': [],
            'pitch_output': [], 'yaw_output': [], 'roll_output': [],
            'pitch_rate_target': [], 'yaw_rate_target': [], 'roll_rate_target': [],
            'pitch_p': [], 'pitch_i': [], 'pitch_d': [],
            'pitch_rate_p': [], 'pitch_rate_i': [], 'pitch_rate_d': []
        }
        
        self.mission_complete = False
        
    def generate_horizontal_takeoff_trajectory(self):
        """Генерация траектории горизонтального взлета"""
        self.target_trajectory = []
        
        for t in np.linspace(0, 400, 200):
            # Фаза 1: Разбег (0-20 сек)
            if t < 20:
                pitch = 0.0  # Горизонтально
                altitude = 0.0
                
            # Фаза 2: Подъем передней стойки (20-25 сек)
            elif t < 25:
                pitch = 5.0 * (t - 20) / 5.0  # Плавный подъем до 5°
                altitude = 0.0
                
            # Фаза 3: Начальный набор высоты (25-40 сек)
            elif t < 40:
                pitch = 5.0 + 5.0 * (t - 25) / 15.0  # До 10°
                altitude = 50.0 * (t - 25)  # Простая модель высоты
                
            # Фаза 4: Основной набор (40-120 сек)
            elif t < 120:
                pitch = 10.0 + 20.0 * (t - 40) / 80.0  # До 30°
                altitude = 750.0 + 150.0 * (t - 40)  # 150 м/с набор
                
            # Фаза 5: Гравитационный разворот (120-250 сек)
            elif t < 250:
                pitch = 30.0 + 30.0 * (t - 120) / 130.0  # До 60°
                altitude = 12750.0 + 250.0 * (t - 120)  # Ускоряем набор
                
            # Фаза 6: Выход на орбиту (250-400 сек)
            else:
                pitch = 60.0 - 20.0 * (t - 250) / 150.0  # Плавное снижение до 40°
                altitude = 45250.0 + 350.0 * (t - 250)  # Финал разгона
                
            self.target_trajectory.append({
                'time': t,
                'pitch': pitch,
                'altitude': altitude
            })
    
    def init_figures(self):
        """Инициализация графиков"""
        plt.style.use('dark_background')
        
        self.figures['trajectory'] = plt.figure(figsize=(5, 3.5), facecolor=DARK_BLUE_NORM)
        self.figures['dynamics'] = plt.figure(figsize=(5, 3.5), facecolor=DARK_BLUE_NORM)
        self.figures['control'] = plt.figure(figsize=(5, 3.5), facecolor=DARK_BLUE_NORM)
        self.figures['aerodynamics'] = plt.figure(figsize=(5, 3.5), facecolor=DARK_BLUE_NORM)
        self.figures['cascade'] = plt.figure(figsize=(5, 3.5), facecolor=DARK_BLUE_NORM)  # Новый график для каскада
    
    def update(self):
        """Обновление состояния системы"""
        dt = self.dt * self.simulation_speed
        
        if self.mission_complete:
            return
        
        # Обновление программы полета
        self.update_flight_program()
        
        # Система стабилизации с каскадным ПИД
        self.stabilization_system(dt)
        
        # Физика
        self.physics_update(dt)
        
        # Сбор данных
        self.collect_telemetry()
        self.collect_control_history()
        
        # Время
        self.time += dt
        self.mission_time += dt
        
        # Проверка событий
        self.check_mission_events()
        
        # Визуализация
        self.update_trajectory_visualization()
        self.update_figures()
        
        # Проверка завершения миссии
        if self.pos[2] >= self.target_altitude and not self.mission_complete:
            self.mission_complete = True
            self.mode = "MISSION_COMPLETE"
            print(f"🎉 МИССИЯ ВЫПОЛНЕНА! Высота: {self.pos[2]/1000:.1f} км")
    
    def update_flight_program(self):
        """Программа полета для горизонтального взлета"""
        if self.mode == "LAUNCH":
            t = self.mission_time
            velocity = np.linalg.norm(self.vel)
            
            # Фаза 0: Подготовка (двигатели на взлетный режим)
            if t < 5:
                self.throttle = 0.9  # Взлетный режим
                self.target_pitch = 0.0
                self.brakes = 1.0  # Держим тормоза до готовности
                
            # Фаза 1: Разгон по ВПП (5-25 сек)
            elif t < 25:
                self.throttle = 1.0  # Максимальная тяга
                self.brakes = 0.0  # Отпускаем тормоза
                self.target_pitch = 0.0
                
                # Управление передним колесом для удержания направления
                if velocity > 10:
                    self.steering = -0.1 * self.get_euler_angles()[1]  # Простая обратная связь
                else:
                    self.steering = 0.0
                    
                # Проверка скорости принятия решения
                if velocity > self.v1_velocity:
                    self.events.append("V1")
                    
                # Подъем передней стойки при достижении VR
                if velocity > self.vr_velocity:
                    self.target_pitch = 3.0  # Начинаем поднимать нос
                    
            # Фаза 2: Отрыв от ВПП (25-30 сек)
            elif t < 30:
                if velocity > self.v2_velocity and self.on_ground:
                    self.on_ground = False
                    self.events.append("LIFTOFF")
                    print(f"✈️ ОТРЫВ! Скорость: {velocity:.1f} м/с, Время: {t:.1f}с")
                    
                self.target_pitch = min(10.0, 3.0 + 7.0 * (t - 25) / 5.0)
                self.throttle = 1.0
                
            # Фаза 3: Набор высоты с уборкой шасси (30-50 сек)
            elif t < 50:
                if self.gear_down and not self.on_ground:
                    self.gear_down = False
                    self.events.append("GEAR UP")
                    
                # Плавный набор тангажа
                self.target_pitch = min(15.0, 10.0 + 5.0 * (t - 30) / 20.0)
                self.throttle = 1.0
                
            # Фаза 4: Разгон и набор (50-100 сек)
            elif t < 100:
                self.target_pitch = min(25.0, 15.0 + 10.0 * (t - 50) / 50.0)
                self.throttle = 1.0
                
            # Фаза 5: Гравитационный разворот (100-200 сек)
            elif t < 200:
                # Целевой тангаж зависит от высоты
                altitude_km = self.pos[2] / 1000
                self.target_pitch = min(60.0, 25.0 + 35.0 * min(altitude_km / 50.0, 1.0))
                self.throttle = 1.0
                
            # Фаза 6: Разгон на орбиту (200-350 сек)
            elif t < 350:
                self.target_pitch = max(30.0, 60.0 - 30.0 * (t - 200) / 150.0)
                # Постепенное снижение тяги по мере уменьшения массы
                if self.mass < 10000:
                    self.throttle = 0.8
                else:
                    self.throttle = 1.0
                    
            # Фаза 7: Финал
            else:
                self.target_pitch = 30.0
                self.throttle = 0.6
    
    def stabilization_system(self, dt):
        """Система стабилизации с каскадным ПИД"""
        # Текущие углы и угловые скорости
        pitch, yaw, roll = self.get_euler_angles()
        
        # Ошибки по углам
        pitch_error = math.radians(self.target_pitch) - pitch
        yaw_error = math.radians(self.target_yaw) - yaw
        roll_error = math.radians(self.target_roll) - roll
        
        # Нормализация ошибок
        pitch_error = self.normalize_angle(pitch_error)
        yaw_error = self.normalize_angle(yaw_error)
        roll_error = self.normalize_angle(roll_error)
        
        if self.use_cascade:
            # Каскадные ПИД-регуляторы
            self.elevator = self.pitch_cascade.calculate(pitch_error, self.omega[1], dt)
            self.rudder = self.yaw_cascade.calculate(yaw_error, self.omega[2], dt)
            self.aileron = self.roll_cascade.calculate(roll_error, self.omega[0], dt)
        else:
            # Классические ПИД (для сравнения)
            self.elevator = self.pitch_pid_classic.calculate(pitch_error, dt)
            self.rudder = self.yaw_pid_classic.calculate(yaw_error, dt)
            self.aileron = self.roll_pid_classic.calculate(roll_error, dt)
        
        # Ограничение сигналов
        self.elevator = np.clip(self.elevator, -1.0, 1.0)
        self.rudder = np.clip(self.rudder, -1.0, 1.0)
        self.aileron = np.clip(self.aileron, -1.0, 1.0)
        self.steering = np.clip(self.steering, -1.0, 1.0)
        self.brakes = np.clip(self.brakes, 0.0, 1.0)
        
        # Сохранение ошибок
        self.telemetry['pitch_error'].append(math.degrees(pitch_error))
        self.telemetry['yaw_error'].append(math.degrees(yaw_error))
        self.telemetry['roll_error'].append(math.degrees(roll_error))
        self.telemetry['elevator'].append(self.elevator)
        self.telemetry['rudder'].append(self.rudder)
        self.telemetry['aileron'].append(self.aileron)
        self.telemetry['throttle'].append(self.throttle)
        self.telemetry['steering'].append(self.steering)
        self.telemetry['brakes'].append(self.brakes)
        self.telemetry['pitch_rate'].append(math.degrees(self.omega[1]))
        self.telemetry['yaw_rate'].append(math.degrees(self.omega[2]))
        self.telemetry['roll_rate'].append(math.degrees(self.omega[0]))
        self.telemetry['ground_contact'].append(1.0 if self.on_ground else 0.0)
    
    def collect_control_history(self):
        """Сбор истории для анализа каскадного ПИД"""
        self.control_history['time'].append(self.mission_time)
        
        pitch, yaw, roll = self.get_euler_angles()
        pitch_error = math.radians(self.target_pitch) - pitch
        yaw_error = math.radians(self.target_yaw) - yaw
        roll_error = math.radians(self.target_roll) - roll
        
        self.control_history['pitch_error'].append(math.degrees(pitch_error))
        self.control_history['yaw_error'].append(math.degrees(yaw_error))
        self.control_history['roll_error'].append(math.degrees(roll_error))
        
        if self.use_cascade:
            self.control_history['pitch_rate_error'].append(
                math.degrees(self.pitch_cascade.rate_error))
            self.control_history['pitch_rate_target'].append(
                math.degrees(self.pitch_cascade.target_rate))
            self.control_history['pitch_output'].append(self.pitch_cascade.rate_output)
            
            # Составляющие ПИД для внешнего контура
            self.control_history['pitch_p'].append(self.pitch_cascade.angle_pid.last_p)
            self.control_history['pitch_i'].append(self.pitch_cascade.angle_pid.last_i)
            self.control_history['pitch_d'].append(self.pitch_cascade.angle_pid.last_d)
            
            # Составляющие для внутреннего контура
            self.control_history['pitch_rate_p'].append(self.pitch_cascade.rate_pid.last_p)
            self.control_history['pitch_rate_i'].append(self.pitch_cascade.rate_pid.last_i)
            self.control_history['pitch_rate_d'].append(self.pitch_cascade.rate_pid.last_d)
        
        # Ограничение размера истории
        max_history = 2000
        for key in self.control_history:
            if len(self.control_history[key]) > max_history:
                self.control_history[key] = self.control_history[key][-max_history:]
    
    def normalize_angle(self, angle):
        """Нормализация угла в диапазон [-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def physics_update(self, dt):
        """Обновление физики с учетом горизонтального взлета"""
        # СИЛЫ
        
        # 1. Тяга двигателей
        thrust_mag = self.throttle * self.get_thrust_at_altitude()
        thrust_body = np.array([thrust_mag, 0.0, 0.0])
        
        # 2. Гравитация
        g = 9.81
        gravity_force = np.array([0.0, 0.0, -self.mass * g])
        
        # 3. Аэродинамические силы
        aero_forces, aero_moments = self.calculate_aerodynamics()
        
        # 4. Силы от шасси (только на земле)
        ground_forces = np.zeros(3)
        ground_moments = np.zeros(3)
        
        if self.on_ground:
            ground_forces, ground_moments = self.calculate_ground_forces()
        
        # Преобразование сил в инерциальную СК
        thrust_inertial = self.body_to_inertial(thrust_body)
        
        # Суммарная сила
        total_force = thrust_inertial + gravity_force + aero_forces + ground_forces
        
        # Ускорение
        self.accel = total_force / self.mass
        
        # Интегрирование скорости и позиции
        self.vel += self.accel * dt
        self.pos += self.vel * dt
        
        # Ограничение движения на земле
        if self.on_ground:
            if self.pos[2] < 0:
                self.pos[2] = 0
                if self.vel[2] < 0:
                    self.vel[2] = 0
            
            # Торможение колесами
            if self.brakes > 0 and np.linalg.norm(self.vel) > 0.1:
                brake_decel = self.brakes * self.brake_force / self.mass
                self.vel *= max(0, 1 - brake_decel * dt / np.linalg.norm(self.vel))
        
        # УГЛОВОЕ ДВИЖЕНИЕ
        
        # Моменты от аэродинамических поверхностей
        control_moments = np.array([
            self.aileron * 50000.0,   # Крен
            self.elevator * 80000.0,   # Тангаж
            self.rudder * 40000.0      # Рыскание
        ])
        
        # Суммарный момент
        total_moment = control_moments + aero_moments + ground_moments
        
        # Угловое ускорение
        angular_accel = np.array([
            total_moment[0] / self.Ixx,
            total_moment[1] / self.Iyy,
            total_moment[2] / self.Izz
        ])
        
        # Интегрирование угловой скорости
        self.omega += angular_accel * dt
        
        # Демпфирование (стабилизация)
        self.omega *= 0.995
        
        # Интегрирование ориентации
        self.integrate_orientation(dt)
        
        # Расход топлива
        if self.throttle > 0 and not self.on_ground:
            self.mass -= self.mass_flow * self.throttle * dt
            self.mass = max(self.mass, 7500.0)
    
    def calculate_aerodynamics(self):
        """Расчет аэродинамических сил и моментов"""
        velocity = np.linalg.norm(self.vel)
        
        if velocity < 1.0:
            return np.zeros(3), np.zeros(3)
        
        # Плотность воздуха
        rho = self.get_atmospheric_density(self.pos[2])
        
        if rho < 1e-6:
            return np.zeros(3), np.zeros(3)
        
        # Скоростной напор
        q = 0.5 * rho * velocity**2
        
        # Направление скорости в связанной СК
        v_body = self.inertial_to_body(self.vel)
        v_body_norm = v_body / (velocity + 1e-6)
        
        # Угол атаки и скольжения
        alpha = math.atan2(v_body[2], v_body[0]) if abs(v_body[0]) > 0.1 else 0
        beta = math.asin(v_body[1] / (velocity + 1e-6))
        
        # Сохраняем для телеметрии
        self.telemetry['alpha'].append(math.degrees(alpha))
        self.telemetry['beta'].append(math.degrees(beta))
        self.telemetry['aoa'].append(math.degrees(alpha))
        self.telemetry['slip'].append(math.degrees(beta))
        
        # Коэффициенты
        Cd = self.Cd0 + self.Cd_alpha * abs(alpha)
        Cl = self.Cl_alpha * alpha
        Cy = -0.5 * beta  # Боковая сила
        
        # Силы в связанной СК
        Fa_body = np.array([
            -q * self.S_ref * Cd,           # Сопротивление
            q * self.S_ref * Cy,             # Боковая сила
            -q * self.S_ref * Cl             # Подъемная сила (отрицательная по Z)
        ])
        
        # Преобразование в инерциальную СК
        Fa_inertial = self.body_to_inertial(Fa_body)
        
        # Аэродинамические моменты
        Ma_body = np.array([
            q * self.S_ref * self.length * (-0.01 * self.omega[0]),  # Демпфирование крена
            q * self.S_ref * self.length * (self.Cm_alpha * alpha - 0.05 * self.omega[1]),  # Тангаж
            q * self.S_ref * self.length * (self.Cn_beta * beta - 0.03 * self.omega[2])     # Рыскание
        ])
        
        return Fa_inertial, Ma_body
    
    def calculate_ground_forces(self):
        """Расчет сил взаимодействия с землей"""
        forces = np.zeros(3)
        moments = np.zeros(3)
        
        # Нормальная реакция опоры
        if self.pos[2] <= 0.1:
            # Вес, распределенный по трем стойкам шасси
            normal_force = self.mass * 9.81
            
            # Основные стойки (сзади)
            main_gear_pos = np.array([-5.0, 0.0, 0.0])  # Позади центра масс
            main_force = 0.8 * normal_force
            
            # Передняя стойка
            nose_gear_pos = np.array([5.0, 0.0, 0.0])  # Впереди центра масс
            nose_force = 0.2 * normal_force
            
            # Силы трения (зависят от скорости)
            vel_mag = np.linalg.norm(self.vel)
            if vel_mag > 0.1:
                vel_dir = self.vel / vel_mag
                
                # Трение качения
                rolling_friction = -self.wheel_friction * normal_force * vel_dir
                
                # Тормозное усилие (только на основных стойках)
                brake_force = -self.brakes * self.brake_force * vel_dir
                
                # Управление передним колесом (создает момент рыскания)
                if abs(self.vel[0]) > 1.0:
                    steering_angle = self.steering * self.max_steering_angle
                    # Боковая сила от повернутого колеса
                    side_force = 0.5 * normal_force * math.tan(steering_angle)
                    moments[2] += side_force * nose_gear_pos[0]  # Момент рыскания
                    
                    # Добавляем силу трения от поворота
                    friction_steering = -0.1 * side_force * np.array([0, 1, 0])
                    forces += friction_steering
                
                forces += rolling_friction + brake_force
            
            forces[2] += normal_force
            
            # Моменты от вертикальных сил (стабилизация)
            moments[1] += nose_force * nose_gear_pos[0] - main_force * main_gear_pos[0]  # Тангаж
        
        return forces, moments
    
    def integrate_orientation(self, dt):
        """Интегрирование кватерниона ориентации"""
        # Нормализованная угловая скорость
        omega = self.omega
        
        # Матрица для производной кватерниона
        Omega = np.array([
            [0, -omega[0], -omega[1], -omega[2]],
            [omega[0], 0, omega[2], -omega[1]],
            [omega[1], -omega[2], 0, omega[0]],
            [omega[2], omega[1], -omega[0], 0]
        ])
        
        # Производная
        q_dot = 0.5 * Omega @ self.q
        
        # Интегрирование
        self.q += q_dot * dt
        
        # Нормализация
        norm = np.linalg.norm(self.q)
        if norm > 0:
            self.q /= norm
    
    def body_to_inertial(self, v_body):
        """Преобразование из связанной в инерциальную СК"""
        w, x, y, z = self.q
        
        # Матрица поворота
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        
        return R @ v_body
    
    def inertial_to_body(self, v_inertial):
        """Преобразование из инерциальной в связанную СК"""
        w, x, y, z = self.q
        
        # Транспонированная матрица
        R_inv = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y + 2*w*z, 2*x*z - 2*w*y],
            [2*x*y - 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z + 2*w*x],
            [2*x*z + 2*w*y, 2*y*z - 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        
        return R_inv @ v_inertial
    
    def get_euler_angles(self):
        """Получение углов Эйлера из кватерниона"""
        w, x, y, z = self.q
        
        # Тангаж
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi/2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Рыскание
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y*y + z*z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Крен
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x*x + y*y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        return pitch, yaw, roll
    
    def get_atmospheric_density(self, altitude):
        """Плотность атмосферы"""
        if altitude < 11000:
            T = 288.15 - 0.0065 * altitude
            p = 101325 * (T / 288.15) ** 5.255
        elif altitude < 20000:
            T = 216.65
            p = 22632 * math.exp(-0.0001577 * (altitude - 11000))
        else:
            T = 216.65 + 0.001 * (altitude - 20000)
            p = 5474 * (216.65 / T) ** 34.163
        
        return p / (287.05 * T)
    
    def get_thrust_at_altitude(self):
        """Тяга на высоте"""
        altitude_km = self.pos[2] / 1000
        
        if altitude_km < 30:
            return self.thrust_max
        else:
            vacuum_factor = 1.0 + altitude_km * 0.01
            return self.thrust_max * min(vacuum_factor, 1.2)
    
    def get_mach_number(self):
        """Число Маха"""
        velocity = np.linalg.norm(self.vel)
        
        if self.pos[2] < 11000:
            T = 288.15 - 0.0065 * self.pos[2]
        else:
            T = 216.65
        
        speed_of_sound = 20.05 * math.sqrt(T)
        
        if speed_of_sound > 0:
            return velocity / speed_of_sound
        return 0.0
    
    def collect_telemetry(self):
        """Сбор телеметрии"""
        self.telemetry['time'].append(self.time)
        self.telemetry['altitude'].append(self.pos[2])
        self.telemetry['velocity'].append(np.linalg.norm(self.vel))
        self.telemetry['mach'].append(self.get_mach_number())
        
        pitch, yaw, roll = self.get_euler_angles()
        self.telemetry['pitch'].append(math.degrees(pitch))
        self.telemetry['yaw'].append(math.degrees(yaw))
        self.telemetry['roll'].append(math.degrees(roll))
        
        rho = self.get_atmospheric_density(self.pos[2])
        velocity = np.linalg.norm(self.vel)
        self.telemetry['q_dyn'].append(0.5 * rho * velocity**2)
        
        self.telemetry['mass'].append(self.mass)
        self.telemetry['thrust'].append(self.get_thrust_at_altitude() * self.throttle)
        self.telemetry['accel'].append(np.linalg.norm(self.accel) / 9.81)
        
        # Ограничение размера
        for key in self.telemetry:
            if len(self.telemetry[key]) > 1000:
                self.telemetry[key].pop(0)
    
    def check_mission_events(self):
        """Проверка событий миссии"""
        t = self.mission_time
        v = np.linalg.norm(self.vel)
        alt_km = self.pos[2] / 1000
        mach = self.get_mach_number()
        
        events_list = [
            (5, "ENGINE START", self.throttle > 0, "Запуск двигателей"),
            (10, "BRAKES OFF", self.brakes < 0.1 and v > 1, "Тормоза отпущены"),
            (15, "V1", v >= self.v1_velocity, "Скорость принятия решения"),
            (20, "VR", v >= self.vr_velocity, "Подъем передней стойки"),
            (25, "V2", v >= self.v2_velocity, "Безопасная скорость взлета"),
            (30, "LIFTOFF", not self.on_ground and self.pos[2] > 1, "Отрыв от ВПП"),
            (35, "GEAR UP", not self.gear_down, "Шасси убраны"),
            (60, "MACH 1", mach >= 0.95, "Приближение к звуковому барьеру"),
            (65, "SUPERSONIC", mach >= 1.05, "Сверхзвуковой полет"),
            (100, "MACH 2", mach >= 2.0, "Достигнута 2М"),
            (200, "KARMAN", alt_km >= 100, "Граница космоса"),
        ]
        
        for check_time, name, condition, desc in events_list:
            if t >= check_time and name not in self.events:
                if condition:
                    self.events.append(name)
                    self.event_times.append(t)
                    print(f"🎯 {name}: {desc} (T+{t:.1f}с)")
    
    def update_trajectory_visualization(self):
        """Обновление визуализации траектории"""
        self.trajectory_points.append({
            'x': self.pos[0],
            'y': self.pos[2],
            'time': self.time
        })
        
        if len(self.trajectory_points) > self.max_trajectory_points:
            self.trajectory_points.pop(0)
    
    def update_figures(self):
        """Обновление графиков"""
        if len(self.telemetry['time']) < 2:
            return
        
        t = self.telemetry['time']
        
        # ГРАФИК 1: Траектория и ориентация
        fig = self.figures['trajectory']
        fig.clear()
        
        ax1 = fig.add_subplot(121)
        ax1.set_facecolor(DARK_BLUE_NORM)
        ax1.set_title('ВЫСОТА', fontsize=9, color='white')
        ax1.set_xlabel('Время, с', color='gray', fontsize=8)
        ax1.set_ylabel('Высота, км', color='gray', fontsize=8)
        ax1.grid(True, alpha=0.2)
        ax1.tick_params(colors='gray', labelsize=7)
        
        if len(t) > 0:
            ax1.plot(t, [h/1000 for h in self.telemetry['altitude']],
                    color=CYAN_NORM, linewidth=1.5, label='Текущая')
            
            if self.show_target_path:
                target_t = [p['time'] for p in self.target_trajectory]
                target_h = [p['altitude']/1000 for p in self.target_trajectory]
                ax1.plot(target_t, target_h, '--', color=YELLOW_NORM,
                        linewidth=1, alpha=0.7, label='Целевая')
        
        ax1.legend(fontsize=7, facecolor=DARK_BLUE_NORM, labelcolor='white')
        
        ax2 = fig.add_subplot(122)
        ax2.set_facecolor(DARK_BLUE_NORM)
        ax2.set_title('ОРИЕНТАЦИЯ', fontsize=9, color='white')
        ax2.set_xlabel('Время, с', color='gray', fontsize=8)
        ax2.set_ylabel('Угол, °', color='gray', fontsize=8)
        ax2.grid(True, alpha=0.2)
        ax2.tick_params(colors='gray', labelsize=7)
        
        if len(t) > 0:
            ax2.plot(t, self.telemetry['pitch'], color=BLUE_NORM, linewidth=1.5, label='Тангаж')
            ax2.axhline(y=self.target_pitch, color=RED_NORM, linestyle=':', alpha=0.5, label='Цель')
        
        ax2.legend(fontsize=7, facecolor=DARK_BLUE_NORM, labelcolor='white')
        
        fig.tight_layout(pad=1.5)
        self.save_figure_to_image('trajectory', fig)
        
        # ГРАФИК 2: Динамика
        fig = self.figures['dynamics']
        fig.clear()
        
        ax1 = fig.add_subplot(111)
        ax1.set_facecolor(DARK_BLUE_NORM)
        ax1.set_title('ДИНАМИКА', fontsize=9, color='white')
        ax1.set_xlabel('Время, с', color='gray', fontsize=8)
        ax1.set_ylabel('Скорость, м/с', color=CYAN_NORM, fontsize=8)
        ax1.grid(True, alpha=0.2)
        ax1.tick_params(colors='gray', labelsize=7)
        
        if len(t) > 0:
            ax1.plot(t, self.telemetry['velocity'], color=CYAN_NORM, linewidth=1.5, label='Скорость')
            
            ax2 = ax1.twinx()
            ax2.set_ylabel('Перегрузка, g', color=ORANGE_NORM, fontsize=8)
            ax2.plot(t, self.telemetry['accel'], color=ORANGE_NORM, linewidth=1.5, label='Перегрузка', alpha=0.8)
            ax2.tick_params(colors='gray', labelsize=7)
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1+lines2, labels1+labels2, fontsize=7,
                      facecolor=DARK_BLUE_NORM, labelcolor='white')
        
        fig.tight_layout(pad=1.5)
        self.save_figure_to_image('dynamics', fig)
        
        # ГРАФИК 3: Управление
        fig = self.figures['control']
        fig.clear()
        
        ax = fig.add_subplot(111)
        ax.set_facecolor(DARK_BLUE_NORM)
        ax.set_title('УПРАВЛЕНИЕ', fontsize=9, color='white')
        ax.set_xlabel('Время, с', color='gray', fontsize=8)
        ax.set_ylabel('Сигнал', color='gray', fontsize=8)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.2)
        ax.tick_params(colors='gray', labelsize=7)
        
        if len(t) > 0:
            n = min(len(t), len(self.telemetry['elevator']))
            if n > 0:
                ax.plot(t[-n:], self.telemetry['elevator'][-n:],
                       color=GREEN_NORM, linewidth=1.5, label='Руль высоты')
                ax.plot(t[-n:], self.telemetry['rudder'][-n:],
                       color=PURPLE_NORM, linewidth=1.5, label='Руль напр.', alpha=0.8)
                ax.plot(t[-n:], self.telemetry['throttle'][-n:],
                       color=YELLOW_NORM, linewidth=1.5, label='Дроссель', alpha=0.6)
                ax.legend(fontsize=7, facecolor=DARK_BLUE_NORM, labelcolor='white')
        
        fig.tight_layout(pad=1.5)
        self.save_figure_to_image('control', fig)
        
        # ГРАФИК 4: Аэродинамика
        fig = self.figures['aerodynamics']
        fig.clear()
        
        ax1 = fig.add_subplot(111)
        ax1.set_facecolor(DARK_BLUE_NORM)
        ax1.set_title('АЭРОДИНАМИКА', fontsize=9, color='white')
        ax1.set_xlabel('Время, с', color='gray', fontsize=8)
        ax1.set_ylabel('Число Маха', color=CYAN_NORM, fontsize=8)
        ax1.grid(True, alpha=0.2)
        ax1.tick_params(colors='gray', labelsize=7)
        
        if len(t) > 0 and len(self.telemetry['mach']) > 0:
            n = min(len(t), len(self.telemetry['mach']))
            ax1.plot(t[-n:], self.telemetry['mach'][-n:],
                    color=CYAN_NORM, linewidth=1.5, label='Число Маха')
            ax1.axhline(y=1.0, color=RED_NORM, linestyle='--', alpha=0.5, label='M=1.0')
            
            ax2 = ax1.twinx()
            ax2.set_ylabel('Угол атаки, °', color=ORANGE_NORM, fontsize=8)
            if len(self.telemetry['alpha']) > 0:
                ax2.plot(t[-n:], self.telemetry['alpha'][-n:],
                        color=ORANGE_NORM, linewidth=1.5, label='Угол атаки', alpha=0.8)
            ax2.tick_params(colors='gray', labelsize=7)
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1+lines2, labels1+labels2, fontsize=7,
                      facecolor=DARK_BLUE_NORM, labelcolor='white')
        
        fig.tight_layout(pad=1.5)
        self.save_figure_to_image('aerodynamics', fig)
        
        # ГРАФИК 5: Каскадный ПИД
        if self.use_cascade:
            fig = self.figures['cascade']
            fig.clear()
            
            ax = fig.add_subplot(111)
            ax.set_facecolor(DARK_BLUE_NORM)
            ax.set_title('КАСКАДНЫЙ ПИД (ТАНГАЖ)', fontsize=9, color='white')
            ax.set_xlabel('Время, с', color='gray', fontsize=8)
            ax.set_ylabel('Значение', color='gray', fontsize=8)
            ax.grid(True, alpha=0.2)
            ax.tick_params(colors='gray', labelsize=7)
            
            if len(self.control_history['time']) > 10:
                t_hist = self.control_history['time']
                n = min(len(t_hist), 200)
                
                ax.plot(t_hist[-n:], self.control_history['pitch_error'][-n:],
                       color=BLUE_NORM, linewidth=1.5, label='Ошибка угла, °')
                ax.plot(t_hist[-n:], self.control_history['pitch_rate_target'][-n:],
                       color=GREEN_NORM, linewidth=1.5, label='Целевая скорость, °/с')
                ax.plot(t_hist[-n:], [math.degrees(self.omega[1])]*n,
                       color=RED_NORM, linewidth=1.5, label='Тек. скорость, °/с', alpha=0.7)
                
                ax.legend(fontsize=7, facecolor=DARK_BLUE_NORM, labelcolor='white')
            
            fig.tight_layout(pad=1.5)
            self.save_figure_to_image('cascade', fig)
    
    def save_figure_to_image(self, name, fig):
        """Сохранение графика в изображение"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, facecolor=DARK_BLUE_NORM,
                   edgecolor='none', bbox_inches='tight')
        buf.seek(0)
        
        image = pygame.image.load(buf)
        image = pygame.transform.scale(image, (400, 280))
        
        self.graph_images[name] = image
        buf.close()
    
    def get_status_text(self):
        """Текст статуса"""
        pitch, yaw, roll = self.get_euler_angles()
        v = np.linalg.norm(self.vel)
        alt_km = self.pos[2] / 1000
        mach = self.get_mach_number()
        
        phase = "РАЗБЕГ" if self.on_ground else "ПОЛЕТ"
        
        lines = []
        lines.append(f"РЕЖИМ: {self.mode:12s} {phase:10s} T+{self.mission_time:6.1f}с")
        lines.append(f"ВЫСОТА: {alt_km:6.1f} км   СКОРОСТЬ: {v:6.0f} м/с")
        lines.append(f"ТАНГАЖ: {math.degrees(pitch):5.1f}°   ЧИСЛО МАХА: {mach:5.2f}")
        lines.append(f"РЫСКАНИЕ: {math.degrees(yaw):5.1f}°   ПЕРЕГРУЗКА: {self.telemetry['accel'][-1] if self.telemetry['accel'] else 0:4.1f} g")
        lines.append(f"КРЕН: {math.degrees(roll):5.1f}°   УА: {self.telemetry['alpha'][-1] if self.telemetry['alpha'] else 0:4.1f}°")
        lines.append(f"МАССА: {self.mass/1000:5.1f} т   ТЯГА: {self.get_thrust_at_altitude()*self.throttle/1000:5.0f} кН")
        lines.append(f"ЦЕЛЬ: {self.target_pitch:5.1f}°   ДРОССЕЛЬ: {self.throttle*100:3.0f}%")
        lines.append(f"РУЛИ: ВЫС={self.elevator:.2f} НАПР={self.rudder:.2f}")
        lines.append(f"ШАССИ: {'ВЫПУЩ' if self.gear_down else 'УБРАНЫ'}   НА ЗЕМЛЕ: {'ДА' if self.on_ground else 'НЕТ'}")
        
        if self.on_ground:
            lines.append(f"РАЗБЕГ: {v:.0f}/{self.v2_velocity:.0f} м/с   ТОРМОЗ: {self.brakes*100:.0f}%")
        
        return lines
    
    def draw(self, surface):
        """Отрисовка"""
        surface.fill(DARK_BLUE)
        
        # Графики
        self.draw_graphs(surface)
        
        # Панели
        self.draw_status_panel(surface)
        self.draw_events_panel(surface)
        
        # Траектория и ВПП
        if self.show_trajectory:
            self.draw_trajectory_and_runway(surface)
        
        # Ракета
        self.draw_rocket(surface)
    
    def draw_rocket(self, surface):
        """Отрисовка ракеты с крыльями (самолетная схема)"""
        rocket_x = 150
        rocket_y = HEIGHT - 250
        
        # Получаем углы
        pitch, yaw, roll = self.get_euler_angles()
        pitch_deg = math.degrees(pitch)
        
        # Создаем поверхность для ракеты
        rocket_surf = pygame.Surface((100, 250), pygame.SRCALPHA)
        
        # Фюзеляж
        body_rect = pygame.Rect(35, 20, 30, 200)
        pygame.draw.ellipse(rocket_surf, PANEL_GRAY, (30, 10, 40, 30))  # Нос
        pygame.draw.rect(rocket_surf, PANEL_GRAY, body_rect)
        pygame.draw.rect(rocket_surf, PANEL_BORDER, body_rect, 2)
        
        # Крылья (для горизонтального взлета)
        wing_rect = pygame.Rect(20, 100, 60, 15)
        pygame.draw.rect(rocket_surf, PANEL_GRAY, wing_rect)
        pygame.draw.rect(rocket_surf, PANEL_BORDER, wing_rect, 2)
        
        # Хвостовое оперение
        tail_rect = pygame.Rect(45, 150, 10, 40)
        pygame.draw.rect(rocket_surf, PANEL_GRAY, tail_rect)
        pygame.draw.rect(rocket_surf, PANEL_BORDER, tail_rect, 2)
        
        # Шасси
        if self.gear_down:
            gear_color = GREEN if self.on_ground else YELLOW
            pygame.draw.circle(rocket_surf, gear_color, (40, 220), 5)  # Левое
            pygame.draw.circle(rocket_surf, gear_color, (60, 220), 5)  # Правое
            pygame.draw.circle(rocket_surf, gear_color, (50, 180), 4)  # Переднее
        
        # Двигатели
        if self.throttle > 0:
            flame_len = 30 + 20 * self.throttle
            for dx in [40, 60]:
                flame_points = [(dx, 220), (dx-5, 220+flame_len), (dx+5, 220+flame_len)]
                colors = [YELLOW, ORANGE, RED]
                for i, color in enumerate(colors):
                    alpha = 150 - i*30
                    flame_surf = pygame.Surface((10, flame_len), pygame.SRCALPHA)
                    pygame.draw.polygon(flame_surf, (*color, alpha),
                                       [(5, flame_len), (0, 0), (10, 0)])
                    rocket_surf.blit(flame_surf, (dx-5, 220))
        
        # Поворот
        rotated = pygame.transform.rotate(rocket_surf, -pitch_deg)
        rect = rotated.get_rect(center=(rocket_x, rocket_y))
        surface.blit(rotated, rect)
        
        # Информация
        font = pygame.font.SysFont('Arial', 12)
        if self.on_ground:
            text = font.render(f"РАЗБЕГ: {np.linalg.norm(self.vel):.0f} м/с", True, YELLOW)
            surface.blit(text, (rocket_x-50, rocket_y+120))
    
    def draw_graphs(self, surface):
        """Отрисовка графиков"""
        graph_width, graph_height = 400, 280
        graphs = ['trajectory', 'dynamics', 'control', 'aerodynamics', 'cascade']
        titles = ['ТРАЕКТОРИЯ', 'ДИНАМИКА', 'УПРАВЛЕНИЕ', 'АЭРОДИНАМИКА', 'КАСКАДНЫЙ ПИД']
        
        positions = [
            (WIDTH//2 - 420, 40),
            (WIDTH//2 + 10, 40),
            (WIDTH//2 - 420, 340),
            (WIDTH//2 + 10, 340),
            (WIDTH - 420, HEIGHT - 320)
        ]
        
        for i, (name, pos) in enumerate(zip(graphs, positions)):
            if name in self.graph_images and i < 4:  # Первые 4 графика
                # Рамка
                rect = pygame.Rect(pos[0]-5, pos[1]-5, graph_width+10, graph_height+10)
                pygame.draw.rect(surface, PANEL_GRAY, rect, border_radius=6)
                pygame.draw.rect(surface, PANEL_BORDER, rect, 2, border_radius=6)
                
                # Заголовок
                font = pygame.font.SysFont('Arial', 12, bold=True)
                title = font.render(titles[i], True, CYAN)
                surface.blit(title, (pos[0] + 10, pos[1] - 20))
                
                # График
                surface.blit(self.graph_images[name], pos)
            
            elif name == 'cascade' and name in self.graph_images and self.use_cascade:
                # 5-й график внизу справа
                rect = pygame.Rect(pos[0]-5, pos[1]-5, graph_width+10, graph_height+10)
                pygame.draw.rect(surface, PANEL_GRAY, rect, border_radius=6)
                pygame.draw.rect(surface, PANEL_BORDER, rect, 2, border_radius=6)
                
                font = pygame.font.SysFont('Arial', 12, bold=True)
                title = font.render(titles[4], True, CYAN)
                surface.blit(title, (pos[0] + 10, pos[1] - 20))
                
                surface.blit(self.graph_images[name], pos)
    
    def draw_status_panel(self, surface):
        """Панель статуса"""
        panel_x, panel_y = 20, 40
        panel_width, panel_height = 450, 250
        
        # Фон
        pygame.draw.rect(surface, PANEL_GRAY,
                        (panel_x, panel_y, panel_width, panel_height),
                        border_radius=12)
        pygame.draw.rect(surface, PANEL_BORDER,
                        (panel_x, panel_y, panel_width, panel_height),
                        2, border_radius=12)
        
        # Заголовок
        font_title = pygame.font.SysFont('Arial', 18, bold=True)
        title = font_title.render("СТАТУС СИСТЕМЫ", True, BLUE)
        surface.blit(title, (panel_x + 20, panel_y + 15))
        
        # Статус
        font = pygame.font.SysFont('Consolas', 12)
        status_lines = self.get_status_text()
        
        for i, line in enumerate(status_lines):
            color = WHITE
            if 'РЕЖИМ:' in line:
                color = GREEN if self.mode == "LAUNCH" else YELLOW
            elif 'ЦЕЛЬ:' in line:
                color = CYAN
            elif 'ШАССИ:' in line:
                color = GREEN if not self.gear_down else YELLOW
            elif 'РАЗБЕГ:' in line:
                v = np.linalg.norm(self.vel)
                color = GREEN if v >= self.v2_velocity else YELLOW
            
            text = font.render(line, True, color)
            surface.blit(text, (panel_x + 20, panel_y + 50 + i * 20))
    
    def draw_events_panel(self, surface):
        """Панель событий"""
        panel_x, panel_y = WIDTH - 250, 40
        panel_width, panel_height = 230, 150
        
        pygame.draw.rect(surface, PANEL_GRAY,
                        (panel_x, panel_y, panel_width, panel_height),
                        border_radius=8)
        pygame.draw.rect(surface, PANEL_BORDER,
                        (panel_x, panel_y, panel_width, panel_height),
                        2, border_radius=8)
        
        font_title = pygame.font.SysFont('Arial', 14, bold=True)
        title = font_title.render("СОБЫТИЯ ПОЛЕТА", True, YELLOW)
        surface.blit(title, (panel_x + 20, panel_y + 10))
        
        font = pygame.font.SysFont('Arial', 10)
        events_to_show = self.events[-5:]
        
        if events_to_show:
            for i, event in enumerate(events_to_show):
                idx = self.events.index(event)
                time_str = f"T+{self.event_times[idx]:.1f}с"
                text = font.render(f"{time_str} - {event}", True, WHITE)
                surface.blit(text, (panel_x + 15, panel_y + 35 + i * 18))
        else:
            text = font.render("Событий пока нет", True, (150,150,150))
            surface.blit(text, (panel_x + 15, panel_y + 50))
    
    def draw_trajectory_and_runway(self, surface):
        """Отрисовка траектории и ВПП"""
        traj_x, traj_y = 20, HEIGHT - 180
        traj_width, traj_height = 450, 120
        
        # Фон
        pygame.draw.rect(surface, PANEL_GRAY,
                        (traj_x, traj_y, traj_width, traj_height),
                        border_radius=8)
        pygame.draw.rect(surface, PANEL_BORDER,
                        (traj_x, traj_y, traj_width, traj_height),
                        2, border_radius=8)
        
        font = pygame.font.SysFont('Arial', 12, bold=True)
        title = font.render("ТРАЕКТОРИЯ И ВПП", True, CYAN)
        surface.blit(title, (traj_x + 10, traj_y - 18))
        
        # ВПП
        runway_x = traj_x + 20
        runway_y = traj_y + traj_height - 20
        runway_len = traj_width - 40
        runway_width = 8
        
        pygame.draw.rect(surface, (100,100,100),
                        (runway_x, runway_y, runway_len, runway_width))
        pygame.draw.rect(surface, (150,150,150),
                        (runway_x, runway_y, runway_len, runway_width), 2)
        
        # Разметка
        for i in range(0, int(runway_len), 30):
            mark_x = runway_x + i
            pygame.draw.rect(surface, WHITE,
                            (mark_x, runway_y + runway_width//2 - 2, 15, 4))
        
        # Позиция на ВПП
        if self.on_ground:
            progress = min(self.pos[0] / self.runway_length, 1.0)
            rocket_runway_x = runway_x + progress * runway_len
            rocket_runway_y = runway_y + runway_width//2
            
            pygame.draw.circle(surface, RED,
                              (int(rocket_runway_x), int(rocket_runway_y)), 6)
            
            font_small = pygame.font.SysFont('Arial', 10)
            speed_text = font_small.render(f"{np.linalg.norm(self.vel):.0f} м/с", True, YELLOW)
            surface.blit(speed_text, (rocket_runway_x - 20, rocket_runway_y - 20))
        
        # Траектория
        if self.trajectory_points:
            points = []
            min_time = min(p['time'] for p in self.trajectory_points)
            max_time = max(p['time'] for p in self.trajectory_points)
            max_alt = max(p['y'] for p in self.trajectory_points) / 1000
            
            scale_x = traj_width / max(max_time - min_time, 1)
            scale_y = (traj_height - 40) / max(max_alt, 1)
            
            for p in self.trajectory_points:
                x = traj_x + (p['time'] - min_time) * scale_x
                y = traj_y + traj_height - 20 - p['y']/1000 * scale_y
                if traj_x <= x <= traj_x + traj_width:
                    points.append((x, y))
            
            if len(points) >= 2:
                pygame.draw.lines(surface, (*CYAN, 150), False, points, 2)
                pygame.draw.circle(surface, RED,
                                  (int(points[-1][0]), int(points[-1][1])), 4)


class ControlPanel:
    """Панель управления"""
    
    def __init__(self, rocket):
        self.rocket = rocket
        self.buttons = []
        self.init_buttons()
    
    def init_buttons(self):
        button_y = HEIGHT - 80
        
        self.buttons.append({
            'rect': pygame.Rect(50, button_y, 100, 35),
            'text': 'СТАРТ',
            'action': self.start_mission,
            'color': GREEN,
            'active': rocket.mode == "PRELAUNCH"
        })
        
        self.buttons.append({
            'rect': pygame.Rect(160, button_y, 100, 35),
            'text': 'ПАУЗА',
            'action': self.toggle_pause,
            'color': YELLOW,
            'active': True
        })
        
        self.buttons.append({
            'rect': pygame.Rect(270, button_y, 100, 35),
            'text': 'СБРОС',
            'action': self.reset_simulation,
            'color': RED,
            'active': True
        })
        
        self.buttons.append({
            'rect': pygame.Rect(380, button_y, 100, 35),
            'text': 'АНАЛИЗ',
            'action': self.show_analysis,
            'color': PURPLE,
            'active': True
        })
        
        self.buttons.append({
            'rect': pygame.Rect(490, button_y, 100, 35),
            'text': 'ПИД',
            'action': self.toggle_pid,
            'color': ORANGE,
            'active': True
        })
        
        # Скорость
        for i, (x, text, speed) in enumerate([(600, '1x', 1.0), (655, '2x', 2.0),
                                              (710, '5x', 5.0), (765, '10x', 10.0)]):
            self.buttons.append({
                'rect': pygame.Rect(x, button_y - 40, 45, 25),
                'text': text,
                'action': lambda s=speed: setattr(self.rocket, 'simulation_speed', s),
                'color': PANEL_GRAY,
                'active': True
            })
    
    def start_mission(self):
        if self.rocket.mode == "PRELAUNCH":
            self.rocket.mode = "LAUNCH"
            self.rocket.mission_time = 0.0
            self.rocket.throttle = 0.9
            print("🚀 ЗАПУСК! Горизонтальный взлет")
    
    def toggle_pause(self):
        self.rocket.simulation_speed = 0.0 if self.rocket.simulation_speed > 0 else 1.0
        print("⏸ ПАУЗА" if self.rocket.simulation_speed == 0 else "▶ ПРОДОЛЖЕНИЕ")
    
    def reset_simulation(self):
        self.rocket.__init__()
        print("🔄 СБРОС")
    
    def toggle_pid(self):
        self.rocket.use_cascade = not self.rocket.use_cascade
        mode = "КАСКАДНЫЙ" if self.rocket.use_cascade else "КЛАССИЧЕСКИЙ"
        print(f"🔄 Режим ПИД: {mode}")
    
    def show_analysis(self):
        if len(self.rocket.control_history['time']) > 10:
            analysis = PostFlightAnalysis(self.rocket)
            analysis.show_control_analysis()
    
    def draw(self, surface):
        # Фон
        pygame.draw.rect(surface, PANEL_GRAY, (0, HEIGHT-100, WIDTH, 100))
        pygame.draw.line(surface, PANEL_BORDER, (0, HEIGHT-100), (WIDTH, HEIGHT-100), 2)
        
        font_title = pygame.font.SysFont('Arial', 14, bold=True)
        title = font_title.render("УПРАВЛЕНИЕ", True, WHITE)
        surface.blit(title, (WIDTH//2 - 50, HEIGHT-95))
        
        font = pygame.font.SysFont('Arial', 12, bold=True)
        
        for button in self.buttons:
            color = button['color']
            if not button['active']:
                color = tuple(c//2 for c in color)
            
            pygame.draw.rect(surface, color, button['rect'], border_radius=4)
            pygame.draw.rect(surface, PANEL_BORDER, button['rect'], 2, border_radius=4)
            
            text = font.render(button['text'], True, WHITE)
            text_rect = text.get_rect(center=button['rect'].center)
            surface.blit(text, text_rect)
        
        # Информация
        font_small = pygame.font.SysFont('Arial', 10)
        speed_text = font_small.render(f"СКОРОСТЬ: {self.rocket.simulation_speed:.1f}x", True, CYAN)
        surface.blit(speed_text, (600, HEIGHT-85))
        
        pid_text = font_small.render(f"ПИД: {'КАСКАД' if self.rocket.use_cascade else 'КЛАССИК'}", 
                                     True, ORANGE)
        surface.blit(pid_text, (490, HEIGHT-85))
    
    def handle_click(self, pos):
        for button in self.buttons:
            if button['rect'].collidepoint(pos) and button['active']:
                button['action']()
                return True
        return False


class PostFlightAnalysis:
    """Анализ полета"""
    
    def __init__(self, rocket):
        self.rocket = rocket
    
    def show_control_analysis(self):
        print("\n" + "="*80)
        print("📊 АНАЛИЗ СИСТЕМЫ УПРАВЛЕНИЯ")
        print("="*80)
        
        self.create_analysis_figures()
        self.print_statistics()
        
        print("\n💡 РЕКОМЕНДАЦИИ:")
        if self.rocket.mission_complete:
            print("✅ Миссия выполнена успешно")
            print("✅ Каскадный ПИД обеспечил качественное управление")
        else:
            print("⚠ Требуется настройка ПИД-коэффициентов")
        
        print("="*80)
    
    def create_analysis_figures(self):
        """Создание графиков анализа"""
        t = self.rocket.control_history['time']
        
        fig1 = plt.figure(figsize=(12, 8), facecolor=DARK_BLUE_NORM)
        fig1.suptitle('АНАЛИЗ КАСКАДНОГО ПИД-РЕГУЛЯТОРА', fontsize=16, color='white')
        
        # Ошибки
        ax1 = plt.subplot(2, 2, 1)
        ax1.set_facecolor(DARK_BLUE_NORM)
        ax1.set_title('ОШИБКИ УПРАВЛЕНИЯ', fontsize=12, color='white')
        ax1.set_xlabel('Время, с', color='gray')
        ax1.set_ylabel('Ошибка, °', color='gray')
        ax1.grid(True, alpha=0.2)
        ax1.tick_params(colors='gray')
        
        if len(t) > 0:
            ax1.plot(t, self.rocket.control_history['pitch_error'],
                    color=BLUE_NORM, linewidth=2, label='Тангаж')
            ax1.plot(t, self.rocket.control_history['yaw_error'],
                    color=GREEN_NORM, linewidth=2, label='Рыскание', alpha=0.8)
            ax1.plot(t, self.rocket.control_history['roll_error'],
                    color=PURPLE_NORM, linewidth=2, label='Крен', alpha=0.6)
        ax1.legend(facecolor=DARK_BLUE_NORM, labelcolor='white')
        
        # Угловые скорости
        ax2 = plt.subplot(2, 2, 2)
        ax2.set_facecolor(DARK_BLUE_NORM)
        ax2.set_title('УГЛОВЫЕ СКОРОСТИ', fontsize=12, color='white')
        ax2.set_xlabel('Время, с', color='gray')
        ax2.set_ylabel('Скорость, °/с', color='gray')
        ax2.grid(True, alpha=0.2)
        ax2.tick_params(colors='gray')
        
        if len(t) > 0 and len(self.rocket.telemetry['pitch_rate']) > 0:
            n = min(len(t), len(self.rocket.telemetry['pitch_rate']))
            ax2.plot(t[-n:], self.rocket.telemetry['pitch_rate'][-n:],
                    color=BLUE_NORM, linewidth=2, label='Тангаж')
            ax2.plot(t[-n:], self.rocket.telemetry['yaw_rate'][-n:],
                    color=GREEN_NORM, linewidth=2, label='Рыскание', alpha=0.8)
            ax2.plot(t[-n:], self.rocket.telemetry['roll_rate'][-n:],
                    color=PURPLE_NORM, linewidth=2, label='Крен', alpha=0.6)
        ax2.legend(facecolor=DARK_BLUE_NORM, labelcolor='white')
        
        # Составляющие ПИД
        ax3 = plt.subplot(2, 2, 3)
        ax3.set_facecolor(DARK_BLUE_NORM)
        ax3.set_title('ПИД-СОСТАВЛЯЮЩИЕ (ВНЕШНИЙ КОНТУР)', fontsize=12, color='white')
        ax3.set_xlabel('Время, с', color='gray')
        ax3.set_ylabel('Значение', color='gray')
        ax3.grid(True, alpha=0.2)
        ax3.tick_params(colors='gray')
        
        if len(self.rocket.control_history['pitch_p']) > 0:
            n = min(len(t), len(self.rocket.control_history['pitch_p']))
            ax3.plot(t[-n:], self.rocket.control_history['pitch_p'][-n:],
                    color=BLUE_NORM, linewidth=2, label='P')
            ax3.plot(t[-n:], self.rocket.control_history['pitch_i'][-n:],
                    color=GREEN_NORM, linewidth=2, label='I')
            ax3.plot(t[-n:], self.rocket.control_history['pitch_d'][-n:],
                    color=PURPLE_NORM, linewidth=2, label='D')
        ax3.legend(facecolor=DARK_BLUE_NORM, labelcolor='white')
        
        # Качество
        ax4 = plt.subplot(2, 2, 4)
        ax4.set_facecolor(DARK_BLUE_NORM)
        ax4.set_title('КАЧЕСТВО УПРАВЛЕНИЯ', fontsize=12, color='white')
        
        if len(self.rocket.control_history['pitch_error']) > 0:
            pitch_rmse = np.sqrt(np.mean(np.array(self.rocket.control_history['pitch_error'])**2))
            yaw_rmse = np.sqrt(np.mean(np.array(self.rocket.control_history['yaw_error'])**2))
            roll_rmse = np.sqrt(np.mean(np.array(self.rocket.control_history['roll_error'])**2))
            
            metrics = ['Тангаж', 'Рыскание', 'Крен']
            values = [pitch_rmse, yaw_rmse, roll_rmse]
            colors = [BLUE_NORM, GREEN_NORM, PURPLE_NORM]
            
            bars = ax4.barh(metrics, values, color=colors)
            ax4.set_xlabel('СКО ошибки, °', color='gray')
            ax4.tick_params(colors='gray')
            ax4.grid(True, alpha=0.2, axis='x')
            
            for bar, val in zip(bars, values):
                ax4.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                        f'{val:.2f}°', va='center', color='white')
        
        plt.tight_layout(pad=3.0)
        
        # Сохраняем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig1.savefig(f'cascade_analysis_{timestamp}.png', dpi=150, facecolor=DARK_BLUE_NORM)
        print(f"📁 График сохранен: cascade_analysis_{timestamp}.png")
        
        plt.show()
        plt.close('all')
    
    def print_statistics(self):
        """Статистика"""
        print("\n📈 СТАТИСТИКА УПРАВЛЕНИЯ:")
        print("-"*60)
        
        pitch_err = self.rocket.control_history['pitch_error']
        yaw_err = self.rocket.control_history['yaw_error']
        roll_err = self.rocket.control_history['roll_error']
        
        if len(pitch_err) > 0:
            print(f"\n{'Канал':<12} {'Макс.':<8} {'Сред.':<8} {'СКО':<8}")
            print("-"*40)
            
            for name, err in zip(['Тангаж', 'Рыскание', 'Крен'],
                                 [pitch_err, yaw_err, roll_err]):
                max_err = max(abs(e) for e in err)
                mean_err = np.mean(np.abs(err))
                rms_err = np.sqrt(np.mean(np.array(err)**2))
                print(f"{name:<12} {max_err:>6.2f}°  {mean_err:>6.2f}°  {rms_err:>6.2f}°")


def main():
    rocket = AdvancedRocket()
    control_panel = ControlPanel(rocket)
    
    clock = pygame.time.Clock()
    running = True
    
    print("="*80)
    print("🚀 СИМУЛЯТОР: ГОРИЗОНТАЛЬНЫЙ СТАРТ С КАСКАДНЫМ ПИД")
    print("="*80)
    print("УПРАВЛЕНИЕ:")
    print("  ПРОБЕЛ - пауза")
    print("  R - сброс")
    print("  A - анализ")
    print("  P - переключение ПИД (каскадный/классический)")
    print("  1/2/5/0 - скорость симуляции")
    print("  ESC - выход")
    print("="*80)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    control_panel.toggle_pause()
                elif event.key == pygame.K_r:
                    control_panel.reset_simulation()
                elif event.key == pygame.K_a:
                    control_panel.show_analysis()
                elif event.key == pygame.K_p:
                    control_panel.toggle_pid()
                elif event.key == pygame.K_1:
                    rocket.simulation_speed = 1.0
                elif event.key == pygame.K_2:
                    rocket.simulation_speed = 2.0
                elif event.key == pygame.K_5:
                    rocket.simulation_speed = 5.0
                elif event.key == pygame.K_0:
                    rocket.simulation_speed = 10.0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    control_panel.handle_click(event.pos)
        
        rocket.update()
        rocket.draw(screen)
        control_panel.draw(screen)
        
        # Подсказки
        font = pygame.font.SysFont('Arial', 12)
        controls = [
            "ПРОБЕЛ-пауза R-сброс A-анализ P-ПИД 1/2/5/0-скорость ESC-выход"
        ]
        for i, text in enumerate(controls):
            surf = font.render(text, True, (200,200,200))
            screen.blit(surf, (20, HEIGHT-30 - i*20))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    print("\n✅ СИМУЛЯЦИЯ ЗАВЕРШЕНА")


if __name__ == "__main__":
    main()