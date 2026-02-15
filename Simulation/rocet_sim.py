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
BLUE = (0, 150, 255)  # Основной цвет
GREEN = (0, 200, 100)  # Успех/норма
YELLOW = (255, 200, 0)  # Внимание
ORANGE = (255, 140, 0)  # Предупреждение
RED = (255, 60, 60)  # Критично
CYAN = (0, 200, 220)  # Данные
PURPLE = (180, 100, 220)  # Траектория

# Нормализованные цвета для Matplotlib (0-1 вместо 0-255)
BLUE_NORM = (0 / 255, 150 / 255, 255 / 255)
GREEN_NORM = (0 / 255, 200 / 255, 100 / 255)
YELLOW_NORM = (255 / 255, 200 / 255, 0 / 255)
ORANGE_NORM = (255 / 255, 140 / 255, 0 / 255)
RED_NORM = (255 / 255, 60 / 255, 60 / 255)
CYAN_NORM = (0 / 255, 200 / 255, 220 / 255)
PURPLE_NORM = (180 / 255, 100 / 255, 220 / 255)
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
    """Усовершенствованная модель ракеты"""

    def __init__(self):
        # ФИЗИЧЕСКИЕ ПАРАМЕТРЫ (реалистичные)
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
        self.max_deflection = math.radians(18)  # Макс. угол рулей

        # СОСТОЯНИЕ СИСТЕМЫ
        self.pos = np.array([0.0, 0.0, 0.0])  # x, y, z (м)
        self.vel = np.array([0.0, 0.0, 0.0])  # м/с
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # Кватернион
        self.omega = np.array([0.0, 0.0, 0.0])  # Угл. скорость (рад/с)

        # УПРАВЛЕНИЕ
        self.throttle = 0.0
        self.elevator_cmd = 0.0
        self.rudder_cmd = 0.0
        self.aileron_cmd = 0.0

        # ЦЕЛЕВАЯ ТРАЕКТОРИЯ
        self.target_trajectory = []
        self.generate_target_trajectory()

        # ТЕКУЩИЕ ЦЕЛИ
        self.target_pitch = 0.0  # Старт горизонтально (для разбега)
        self.target_yaw = 0.0
        self.target_roll = 0.0
        self.target_altitude = 100000.0  # 100 км
        self.target_velocity = 80.0  # Целевая скорость для взлета (м/с)

        # ПИД-РЕГУЛЯТОРЫ
        self.pitch_pid = PID(Kp=0.9, Ki=0.0, Kd=1.7, max_output=1.0)
        self.yaw_pid = PID(Kp=0.6, Ki=0.015, Kd=0.12, max_output=1.0)
        self.roll_pid = PID(Kp=0.4, Ki=0.01, Kd=0.08, max_output=1.0)

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

        # ВИЗУАЛИЗАЦИЯ ТРАЕКТОРИИ
        self.trajectory_points = []
        self.max_trajectory_points = 500

        # ГРАФИКИ
        self.figures = {}
        self.graph_images = {}
        self.init_figures()

        # ДЕМОНСТРАЦИОННЫЕ ПАРАМЕТРЫ
        self.show_trajectory = True
        self.show_target_path = True
        self.show_control_forces = True

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

        # ФЛАГ ЗАВЕРШЕНИЯ
        self.mission_complete = False

        # ДЛЯ САМОЛЕТНОГО СТАРТА
        self.takeoff_velocity = 90.0  # Скорость отрыва (м/с)
        self.rotation_speed = 70.0  # Скорость начала подъема носа (м/с)
        self.on_runway = True  # Находится ли на полосе
        self.runway_length = 3000.0  # Длина ВПП (м)

    def generate_target_trajectory(self):
        """Генерация целевой траектории (гравитационный разворот)"""
        self.target_trajectory = []
        for t in np.linspace(0, 300, 100):  # 300 секунд полета
            if t < 10:
                pitch = 0.0  # Горизонтальный разбег
            elif t < 20:
                pitch = 15.0  # Взлетный угол
            elif t < 60:
                pitch = 15.0 + (t - 20) * 1.5  # Начало разворота
            elif t < 120:
                pitch = 75.0 - (t - 60) * 0.5  # Плавный разворот
            elif t < 180:
                pitch = 45.0 - (t - 120) * 0.3  # Завершение разворота
            else:
                pitch = 5.0  # Почти горизонтально

            altitude = 100 * t  # Примерная высота
            self.target_trajectory.append({
                'time': t,
                'pitch': pitch,
                'altitude': altitude
            })
    
    def init_figures(self):
        """Инициализация профессиональных графиков"""
        plt.style.use('dark_background')

        # ГРАФИК 1: Траектория и ориентация
        self.figures['trajectory'] = plt.figure(figsize=(5, 3.5), facecolor=DARK_BLUE_NORM)

        # ГРАФИК 2: Скорость и ускорение
        self.figures['dynamics'] = plt.figure(figsize=(5, 3.5), facecolor=DARK_BLUE_NORM)

        # ГРАФИК 3: Управление и ошибки
        self.figures['control'] = plt.figure(figsize=(5, 3.5), facecolor=DARK_BLUE_NORM)

        # ГРАФИК 4: Аэродинамика
        self.figures['aerodynamics'] = plt.figure(figsize=(5, 3.5), facecolor=DARK_BLUE_NORM)
        self.figures['cascade'] = plt.figure(figsize=(5, 3.5), facecolor=DARK_BLUE_NORM)  # Новый график для каскада
    
    def update(self):
        
        dt = self.dt * self.simulation_speed

        # Проверка завершения миссии
        if self.pos[2] >= self.target_altitude and not self.mission_complete:
            self.mission_complete = True
            self.mode = "MISSION_COMPLETE"
            print(f"🎉 МИССИЯ ВЫПОЛНЕНА! Достигнута высота {self.pos[2] / 1000:.1f} км")
            print(f"⏱ Время выполнения: {self.mission_time:.1f} сек")
            print(f"🚀 Скорость в апогее: {np.linalg.norm(self.vel):.0f} м/с")

        if self.mission_complete:
            return

        # УПРАВЛЕНИЕ ПО ПРОГРАММЕ
        self.update_flight_program()

        # СИСТЕМА СТАБИЛИЗАЦИИ
        self.stabilization_system(dt)

        # ФИЗИКА
        self.physics_update(dt)

        # ДАННЫЕ
        self.collect_telemetry()

        # ИСТОРИЯ УПРАВЛЕНИЯ
        self.collect_control_history()

        # ВРЕМЯ
        self.time += dt
        self.mission_time += dt

        # СОБЫТИЯ
        self.check_mission_events()

        # ВИЗУАЛИЗАЦИЯ
        self.update_trajectory_visualization()
        self.update_figures()
        
        # Проверка завершения миссии
        if self.pos[2] >= self.target_altitude and not self.mission_complete:
            self.mission_complete = True
            self.mode = "MISSION_COMPLETE"
            print(f"🎉 МИССИЯ ВЫПОЛНЕНА! Высота: {self.pos[2]/1000:.1f} км")
    
    def update_flight_program(self):
        """Обновление программы полета - САМОЛЕТНЫЙ СТАРТ"""
        if self.mode == "LAUNCH":
            # САМОЛЕТНЫЙ СТАРТ: разбег по полосе -> взлет -> гравитационный разворот
            t = self.mission_time
            velocity = np.linalg.norm(self.vel)

            # Фаза 1: Разбег по полосе (первые 10 секунд)
            if t < 10:
                self.target_pitch = 0.0  # Горизонтально
                self.target_yaw = 0.0
                self.target_roll = 0.0
                self.throttle = 1.0  # Максимальная тяга
                self.on_runway = True

                # Когда достигаем скорости вращения, начинаем поднимать нос
                if velocity >= self.rotation_speed:
                    self.target_pitch = 10.0  # Подъем носа для взлета

            # Фаза 2: Взлет (10-20 секунд)
            elif t < 20:
                if self.pos[2] < 10:  # Если еще на земле или только оторвались
                    self.target_pitch = 15.0  # Угол набора высоты
                else:
                    self.target_pitch = 20.0  # Более крутой набор
                self.throttle = 1.0
                self.on_runway = False

            # Фаза 3: Набор высоты и начало гравитационного разворота (20-60 секунд)
            elif t < 60:
                altitude_km = self.pos[2] / 1000
                # Плавный разворот в зависимости от высоты
                self.target_pitch = min(75.0, 20.0 + altitude_km * 3.0)
                self.throttle = 1.0

            # Фаза 4: Активный гравитационный разворот (60-120 секунд)
            elif t < 120:
                self.target_pitch = max(30.0, 75.0 - (t - 60) * 0.5)
                self.throttle = 1.0

            # Фаза 5: Завершение разворота (120-180 секунд)
            elif t < 180:
                self.target_pitch = max(5.0, 30.0 - (t - 120) * 0.2)
                self.throttle = 0.8

            # Фаза 6: Полет по орбите
            else:
                self.target_pitch = 5.0
                # Регулируем тягу в зависимости от высоты
                altitude_km = self.pos[2] / 1000
                if altitude_km > 80:
                    self.throttle = 0.6
                else:
                    self.throttle = 0.8

    def stabilization_system(self, dt):
        """Система стабилизации (ПИД-регуляторы)"""
        # Текущие углы
        pitch, yaw, roll = self.get_euler_angles()
        
        # Ошибки по углам
        pitch_error = math.radians(self.target_pitch) - pitch
        yaw_error = math.radians(self.target_yaw) - yaw
        roll_error = math.radians(self.target_roll) - roll

        # ПИД-регуляторы
        self.elevator_cmd = self.pitch_pid.calculate(pitch_error, dt)
        self.rudder_cmd = self.yaw_pid.calculate(yaw_error, dt)
        self.aileron_cmd = self.roll_pid.calculate(roll_error, dt)

        # Ограничение команд
        self.elevator_cmd = max(-1.0, min(1.0, self.elevator_cmd))
        self.rudder_cmd = max(-1.0, min(1.0, self.rudder_cmd))
        self.aileron_cmd = max(-1.0, min(1.0, self.aileron_cmd))

        # Сохранение ошибок для анализа
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
        """Сбор истории управления для анализа"""
        self.control_history['time'].append(self.mission_time)
        
        pitch, yaw, roll = self.get_euler_angles()
        pitch_error = math.radians(self.target_pitch) - pitch
        yaw_error = math.radians(self.target_yaw) - yaw
        roll_error = math.radians(self.target_roll) - roll
        
        self.control_history['pitch_error'].append(math.degrees(pitch_error))
        self.control_history['yaw_error'].append(math.degrees(yaw_error))
        self.control_history['roll_error'].append(math.degrees(roll_error))

        # Выходы регуляторов
        self.control_history['pitch_output'].append(self.elevator_cmd)
        self.control_history['yaw_output'].append(self.rudder_cmd)
        self.control_history['roll_output'].append(self.aileron_cmd)

        # Составляющие ПИД (для анализа)
        # Сохраняем последние значения (упрощенно)
        if hasattr(self.pitch_pid, 'last_p'):
            self.control_history['pitch_p'].append(self.pitch_pid.last_p)
            self.control_history['pitch_i'].append(self.pitch_pid.last_i)
            self.control_history['pitch_d'].append(self.pitch_pid.last_d)
        else:
            self.control_history['pitch_p'].append(0)
            self.control_history['pitch_i'].append(0)
            self.control_history['pitch_d'].append(0)

        # Ограничиваем размер истории
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
        """Обновление физики - САМОЛЕТНЫЙ ВЗЛЕТ"""
        # СИЛЫ

        # 1. Тяга двигателя
        thrust_mag = self.throttle * self.get_thrust_at_altitude()

        # Направление тяги (вперед по оси X ракеты для горизонтального старта)
        thrust_body = np.array([thrust_mag, 0.0, 0.0])

        # 2. Гравитация
        g = 9.81
        gravity_inertial = np.array([0.0, 0.0, -g * self.mass])

        # 3. Аэродинамические силы
        aero_forces = self.calculate_aerodynamic_forces()

        # 4. Сила реакции земли (только когда на полосе)
        ground_force = np.array([0.0, 0.0, 0.0])
        if self.on_runway and self.pos[2] <= 0.1:
            # Нормальная реакция опоры
            ground_normal = g * self.mass
            ground_force = np.array([0.0, 0.0, ground_normal])

            # Трение при разбеге (коэффициент трения 0.02 для бетона)
            if np.linalg.norm(self.vel) > 0.1:
                friction_force = -0.02 * ground_normal * (self.vel / np.linalg.norm(self.vel))
                ground_force += friction_force

        # ПРЕОБРАЗОВАНИЕ СИЛ
        thrust_inertial = self.body_to_inertial(thrust_body)

        # СУММАРНАЯ СИЛА
        total_force = thrust_inertial + gravity_inertial + aero_forces + ground_force

        # УСКОРЕНИЕ
        acceleration = total_force / self.mass

        # ИНТЕГРИРОВАНИЕ СКОРОСТИ И ПОЗИЦИИ
        self.vel += acceleration * dt
        self.pos += self.vel * dt

        # Если на полосе, ограничиваем вертикальное движение
        if self.on_runway and self.pos[2] < 0:
            self.pos[2] = 0
            if self.vel[2] < 0:
                self.vel[2] = 0

        # Автоматический отрыв от полосы при достаточной скорости
        if self.on_runway and np.linalg.norm(self.vel) >= self.takeoff_velocity:
            self.on_runway = False
            print(f"✈️ ОТРЫВ! Скорость: {np.linalg.norm(self.vel):.1f} м/с")

        # УГЛОВОЕ ДВИЖЕНИЕ

        # Моменты от органов управления
        control_moments = np.array([
            self.aileron_cmd * 40000.0,  # Крен
            self.elevator_cmd * 60000.0,  # Тангаж
            self.rudder_cmd * 30000.0  # Рыскание
        ])

        # Аэродинамические моменты (демпфирование)
        damping_moments = -0.15 * self.omega * np.array([self.Ixx, self.Iyy, self.Izz])

        # Суммарный момент
        total_moment = control_moments + damping_moments

        # Угловое ускорение
        angular_acceleration = np.array([
            total_moment[0] / self.Ixx,
            total_moment[1] / self.Iyy,
            total_moment[2] / self.Izz
        ])

        # Интегрирование угловой скорости
        self.omega += angular_acceleration * dt

        # Интегрирование ориентации
        self.integrate_orientation(dt)

        # РАСХОД ТОПЛИВА
        if self.throttle > 0:
            self.mass -= self.mass_flow * self.throttle * dt
            self.mass = max(self.mass, 7500.0)  # Сухая масса

        # ЗАЩИТА ОТ ЗЕМЛИ
        if self.pos[2] < 0 and not self.on_runway:
            self.pos[2] = 0
            self.vel[2] = max(self.vel[2], 0)

    def integrate_orientation(self, dt):
        """Интегрирование ориентации через кватернионы"""
        # Матрица угловой скорости
        omega = self.omega
        Omega = np.array([
            [0, -omega[0], -omega[1], -omega[2]],
            [omega[0], 0, omega[2], -omega[1]],
            [omega[1], -omega[2], 0, omega[0]],
            [omega[2], omega[1], -omega[0], 0]
        ])

        # Производная кватерниона
        q_dot = 0.5 * Omega @ self.q

        # Интегрирование (метод Эйлера)
        self.q += q_dot * dt

        # Нормализация
        norm = np.linalg.norm(self.q)
        if norm > 0:
            self.q /= norm

    def body_to_inertial(self, vector_body):
        """Преобразование из связанной СК в инерциальную"""
        w, x, y, z = self.q

        # Матрица поворота из кватерниона
        R = np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]
        ])

        return R @ vector_body

    def calculate_aerodynamic_forces(self):
        """Расчет аэродинамических сил - УЧИТЫВАЕМ ПОДЪЕМНУЮ СИЛУ"""
        velocity_mag = np.linalg.norm(self.vel)

        if velocity_mag < 1.0:
            return np.array([0.0, 0.0, 0.0])

        # Плотность воздуха (модель атмосферы)
        rho = self.get_atmospheric_density(self.pos[2])

        if rho < 1e-6:
            return np.array([0.0, 0.0, 0.0])

        # Скоростной напор
        dynamic_pressure = 0.5 * rho * velocity_mag ** 2

        # Угол атаки
        alpha = self.get_angle_of_attack()

        # Коэффициенты
        # Сопротивление увеличивается с углом атаки
        Cd = self.Cd + 0.15 * abs(alpha)

        # Подъемная сила пропорциональна углу атаки и квадрату скорости
        Cl = self.Cl_alpha * alpha

        # Направление скорости
        velocity_dir = self.vel / velocity_mag

        # Сила сопротивления (против движения)
        drag_mag = dynamic_pressure * self.S_ref * Cd
        drag_force = -drag_mag * velocity_dir

        # ПОДЪЕМНАЯ СИЛА (перпендикулярно скорости, вверх)
        # Для самолетного взлета критически важна!
        # Направление подъемной силы: перпендикулярно скорости и в сторону от земли
        lift_direction = np.array([0, 0, 1])  # Вверх в инерциальной СК

        # Проецируем на плоскость, перпендикулярную скорости
        velocity_component = np.dot(lift_direction, velocity_dir) * velocity_dir
        lift_direction_perp = lift_direction - velocity_component

        # Нормализуем
        lift_dir_norm = np.linalg.norm(lift_direction_perp)
        if lift_dir_norm > 0:
            lift_direction = lift_direction_perp / lift_dir_norm

        lift_mag = dynamic_pressure * self.S_ref * Cl
        lift_force = lift_mag * lift_direction

        return drag_force + lift_force

    def get_angle_of_attack(self):
        """Расчет угла атаки"""
        if np.linalg.norm(self.vel) < 1.0:
            return 0.0

        # Направление скорости в связанной СК
        velocity_body = self.inertial_to_body(self.vel)

        # Угол атаки = arctan(w/u) (w - вертикальная составляющая, u - продольная)
        u = velocity_body[0] if abs(velocity_body[0]) > 0.1 else 0.1
        w = velocity_body[2]

        return math.atan2(w, u)

    def inertial_to_body(self, vector_inertial):
        """Преобразование из инерциальной СК в связанную"""
        w, x, y, z = self.q

        # Обратная матрица поворота (транспонированная, т.к. ортогональная)
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
        """Плотность воздуха на заданной высоте (упрощенная модель)"""
        if altitude < 11000:  # Тропосфера
            T = 288.15 - 0.0065 * altitude
            p = 101325 * (T / 288.15) ** 5.255
        elif altitude < 20000:
            T = 216.65
            p = 22632 * math.exp(-0.0001577 * (altitude - 11000))
        else:  # Высокие слои
            T = 216.65 + 0.001 * (altitude - 20000)
            p = 5474 * (216.65 / T) ** 34.163
        
        return p / (287.05 * T)
    
    def get_thrust_at_altitude(self):
        """Тяга на текущей высоте (зависит от давления)"""
        altitude_km = self.pos[2] / 1000
        
        if altitude_km < 30:
            # В атмосфере - номинальная тяга
            return self.thrust_max
        else:
            # В вакууме тяга увеличивается
            vacuum_factor = 1.0 + altitude_km * 0.01
            return self.thrust_max * min(vacuum_factor, 1.2)
    
    def get_mach_number(self):
        
        velocity = np.linalg.norm(self.vel)

        # Скорость звука (зависит от температуры)
        altitude = self.pos[2]
        if altitude < 11000:
            T = 288.15 - 0.0065 * altitude
        else:
            T = 216.65
        
        speed_of_sound = 20.05 * math.sqrt(T)
        
        if speed_of_sound > 0:
            return velocity / speed_of_sound
        return 0.0

    def get_euler_angles(self):
        """Получение углов Эйлера из кватерниона"""
        w, x, y, z = self.q

        # Тангаж (pitch)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Рыскание (yaw)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # Крен (roll)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        return pitch, yaw, roll

    def collect_telemetry(self):
        
        self.telemetry['time'].append(self.time)
        self.telemetry['altitude'].append(self.pos[2])
        self.telemetry['velocity'].append(np.linalg.norm(self.vel))
        self.telemetry['mach'].append(self.get_mach_number())
        
        pitch, yaw, roll = self.get_euler_angles()
        self.telemetry['pitch'].append(math.degrees(pitch))
        self.telemetry['yaw'].append(math.degrees(yaw))
        self.telemetry['roll'].append(math.degrees(roll))

        self.telemetry['alpha'].append(math.degrees(self.get_angle_of_attack()))
        self.telemetry['thrust'].append(self.get_thrust_at_altitude() * self.throttle)
        self.telemetry['mass'].append(self.mass)

        # Скоростной напор
        rho = self.get_atmospheric_density(self.pos[2])
        velocity = np.linalg.norm(self.vel)
        self.telemetry['q_dyn'].append(0.5 * rho * velocity ** 2)

        # Ускорение (в g)
        aero_forces = self.calculate_aerodynamic_forces()
        accel_mag = np.linalg.norm(aero_forces) / self.mass
        self.telemetry['accel'].append(accel_mag / 9.81)

        # Ограничение длины массивов
        for key in self.telemetry:
            if len(self.telemetry[key]) > 1000:
                self.telemetry[key].pop(0)
    
    def check_mission_events(self):
        """Проверка событий миссии"""
        altitude_km = self.pos[2] / 1000
        mach = self.get_mach_number()
        q_dyn = self.telemetry['q_dyn'][-1] if self.telemetry['q_dyn'] else 0
        velocity = np.linalg.norm(self.vel)

        events = [
            (5, "V1", velocity >= 60, "Скорость принятия решения"),
            (8, "ROTATION", velocity >= self.rotation_speed, "Подъем носа"),
            (10, "LIFTOFF", self.pos[2] > 1.0 and not self.on_runway, "Отрыв от ВПП"),
            (15, "GEAR UP", self.pos[2] > 10, "Уборка шасси"),
            (60, "MACH 1", mach >= 0.95, "Приближение к звуковому барьеру"),
            (65, "TRANSONIC", 0.95 <= mach <= 1.05, "Трансзвуковой режим"),
            (70, "SUPERSONIC", mach >= 1.05, "Сверхзвуковой полет"),
            (120, "MACH 2", mach >= 2.0, "Достигнута 2М"),
            (200, "MACH 3", mach >= 3.0, "Достигнута 3М"),
            (100, "КАРМАН ЛИНИЯ", altitude_km >= 100, "Граница космоса"),
        ]
        
        for check_time, name, condition, desc in events_list:
            if t >= check_time and name not in self.events:
                if condition:
                    self.events.append(name)
                    self.event_times.append(t)
                    print(f"🎯 {name}: {desc} (T+{t:.1f}с)")
    
    def update_trajectory_visualization(self):
        """Обновление визуализации траектории"""
        # Добавляем текущую точку
        self.trajectory_points.append({
            'x': self.pos[0],
            'y': self.pos[2],
            'time': self.time
        })

        # Ограничиваем количество точек
        if len(self.trajectory_points) > self.max_trajectory_points:
            self.trajectory_points.pop(0)
    
    def update_figures(self):
        
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

        # Линия текущей высоты
        if len(t) > 0:
            ax1.plot(t, [h / 1000 for h in self.telemetry['altitude']],
                     color=CYAN_NORM, linewidth=1.5, label='Текущая')

            # Целевая траектория
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

        # ГРАФИК 2: Скорость и ускорение
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

            # Добавляем легенду
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
            # Сигналы управления
            if len(self.telemetry['elevator_cmd']) > 0:
                # Берем последние N значений
                n_points = min(len(t), len(self.telemetry['elevator_cmd']))
                t_plot = t[-n_points:]
                elevator = self.telemetry['elevator_cmd'][-n_points:]
                rudder = self.telemetry['rudder_cmd'][-n_points:]
                throttle = self.telemetry['throttle_cmd'][-n_points:]

                ax.plot(t_plot, elevator, color=GREEN_NORM, linewidth=1.5, label='Руль высоты')
                ax.plot(t_plot, rudder, color=PURPLE_NORM, linewidth=1.5, label='Руль напр.', alpha=0.8)
                ax.plot(t_plot, throttle, color=YELLOW_NORM, linewidth=1.5, label='Дроссель', alpha=0.6)

                ax.legend(fontsize=7, facecolor=DARK_BLUE_NORM, edgecolor='none', labelcolor='white')

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

            # Отметка звукового барьера
            ax1.axhline(y=1.0, color=RED_NORM, linestyle='--', linewidth=1, alpha=0.5, label='M=1.0')

            # Добавляем легенду
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
        """Сохранение графика в изображение для PyGame"""
        # Сохраняем в буфер
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, facecolor=DARK_BLUE_NORM,
                   edgecolor='none', bbox_inches='tight')
        buf.seek(0)

        # Загружаем в PyGame
        image = pygame.image.load(buf)

        # Масштабируем изображение
        target_width = 400
        target_height = 280
        image = pygame.transform.scale(image, (target_width, target_height))

        self.graph_images[name] = image

        # Закрываем буфер
        buf.close()
    
    def get_status_text(self):
        """Получение текста статуса для отображения"""
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
        """Отрисовка ракеты и интерфейса"""
        # Фон
        surface.fill(DARK_BLUE)

        # Сначала рисуем графики (они на заднем плане)
        self.draw_graphs(surface)

        # Затем панели (они поверх графиков)
        self.draw_status_panel(surface)
        self.draw_events_panel(surface)

        # Отрисовка траектории и ВПП
        if self.show_trajectory:
            self.draw_trajectory_and_runway(surface)

        # В самом конце - ракету и управление (они поверх всего)
        self.draw_rocket_and_controls(surface)

    def draw_rocket_and_controls(self, surface):
        """Отрисовка ракеты и элементов управления"""
        # Расчет позиции ракеты на экране
        rocket_x = 150
        rocket_y = HEIGHT - 250

        # Отрисовка ракеты
        self.draw_rocket(surface, rocket_x, rocket_y)

    def draw_rocket(self, surface, x, y):
        """Отрисовка 3D-модели ракеты"""
        # Тело ракеты
        rocket_width = 30
        rocket_height = 200

        # Поворот ракеты по тангажу
        pitch, _, _ = self.get_euler_angles()
        pitch_deg = math.degrees(pitch)

        # Создаем поверхность для ракеты
        rocket_surf = pygame.Surface((rocket_width + 20, rocket_height + 20), pygame.SRCALPHA)

        # Корпус (более аэродинамическая форма для самолета)
        points = [
            (rocket_width // 2, 0),
            (rocket_width, rocket_height // 4),
            (rocket_width * 3 // 4, rocket_height // 2),
            (rocket_width, rocket_height * 3 // 4),
            (rocket_width // 2, rocket_height),
            (0, rocket_height * 3 // 4),
            (rocket_width // 4, rocket_height // 2),
            (0, rocket_height // 4)
        ]

        pygame.draw.polygon(rocket_surf, PANEL_GRAY, points)
        pygame.draw.polygon(rocket_surf, PANEL_BORDER, points, 2)

        # Крылья (если это самолет-носитель)
        wing_width = 40
        wing_height = 10
        wing_y = rocket_height // 3

        # Левое крыло
        pygame.draw.rect(rocket_surf, PANEL_GRAY,
                         (0, wing_y, wing_width, wing_height))
        # Правое крыло
        pygame.draw.rect(rocket_surf, PANEL_GRAY,
                         (rocket_width - wing_width, wing_y, wing_width, wing_height))

        # Двигатель (с анимацией)
        if self.throttle > 0:
            flame_height = 30 + 20 * self.throttle
            flame_width = 15

            # Два двигателя по бокам (как у самолета)
            for offset in [-flame_width, flame_width]:
                flame_points = [
                    (rocket_width // 2 + offset, rocket_height),
                    (rocket_width // 2 + offset - flame_width // 2, rocket_height + flame_height),
                    (rocket_width // 2 + offset, rocket_height + flame_height * 0.8),
                    (rocket_width // 2 + offset + flame_width // 2, rocket_height + flame_height)
                ]

                # Пламя с градиентом
                colors = [(255, 255, 0), (255, 140, 0), (255, 60, 60)]
                for i in range(len(colors)):
                    flame_surf = pygame.Surface((flame_width, flame_height), pygame.SRCALPHA)
                    pygame.draw.polygon(flame_surf, (*colors[i], 150),
                                        [(0, flame_height), (flame_width // 2, 0), (flame_width, flame_height)])
                    rocket_surf.blit(flame_surf, (rocket_width // 2 + offset - flame_width // 2, rocket_height))

        # Поворачиваем ракету
        rotated_rocket = pygame.transform.rotate(rocket_surf, -pitch_deg)
        rocket_rect = rotated_rocket.get_rect(center=(x, y))

        # Отображаем ракету
        surface.blit(rotated_rocket, rocket_rect)

        # Маркер цели
        target_y = y - self.target_pitch * 2
        pygame.draw.circle(surface, YELLOW, (x, int(target_y)), 8, 2)
        pygame.draw.line(surface, YELLOW, (x - 10, target_y), (x + 10, target_y), 1)
        pygame.draw.line(surface, YELLOW, (x, target_y - 10), (x, target_y + 10), 1)

        # Подпись
        font = pygame.font.SysFont('Arial', 14, bold=True)
        text = font.render("САМОЛЕТ-НОСИТЕЛЬ", True, CYAN)
        surface.blit(text, (x - 70, y - 120))

        # Индикатор на ВПП
        if self.on_runway:
            font_small = pygame.font.SysFont('Arial', 12)
            runway_text = font_small.render(f"РАЗБЕГ: {np.linalg.norm(self.vel):.0f}/{self.takeoff_velocity:.0f} м/с",
                                            True, GREEN if np.linalg.norm(self.vel) >= self.rotation_speed else YELLOW)
            surface.blit(runway_text, (x - 60, y + 100))

    def draw_graphs(self, surface):
        """Отрисовка графиков - 2x2 сетка (ИСПРАВЛЕНО РАСПОЛОЖЕНИЕ)"""
        graph_width = 400
        graph_height = 280

        # Позиции графиков в сетке 2x2 - СДВИНУТО ВПРАВО И ВНИЗ
        graph_positions = {
            'trajectory': (WIDTH // 2 - 420, 40),  # Левый верхний (поднят выше)
            'dynamics': (WIDTH // 2 + 10, 40),  # Правый верхний (поднят выше)
            'control': (WIDTH // 2 - 420, 340),  # Левый нижний (опущен ниже)
            'aerodynamics': (WIDTH // 2 + 10, 340)  # Правый нижний (опущен ниже)
        }

        # Названия графиков для отображения
        graph_titles = {
            'trajectory': 'ТРАЕКТОРИЯ И ОРИЕНТАЦИЯ',
            'dynamics': 'ДИНАМИКА ПОЛЕТА',
            'control': 'СИСТЕМА УПРАВЛЕНИЯ',
            'aerodynamics': 'АЭРОДИНАМИКА'
        }

        for name, pos in graph_positions.items():
            if name in self.graph_images:
                # Полупрозрачный фон для графика
                graph_rect = pygame.Rect(pos[0] - 5, pos[1] - 5,
                                         graph_width + 10,
                                         graph_height + 10)
                pygame.draw.rect(surface, PANEL_GRAY, graph_rect, border_radius=6)
                pygame.draw.rect(surface, PANEL_BORDER, graph_rect, 2, border_radius=6)

                # Заголовок графика
                font = pygame.font.SysFont('Arial', 12, bold=True)
                title = font.render(titles[4], True, CYAN)
                surface.blit(title, (pos[0] + 10, pos[1] - 20))

                # График
                surface.blit(self.graph_images[name], pos)
    
    def draw_status_panel(self, surface):
        """Отрисовка панели состояния - УМЕНЬШЕНА И СДВИНУТА"""
        panel_x = 20
        panel_y = 40  # Сдвинута вниз
        panel_width = 450
        panel_height = 250  # Уменьшена

        # Фон панели
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

        # Текст статуса
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
        """Отрисовка панели событий - УМЕНЬШЕНА И СДВИНУТА"""
        panel_x = WIDTH - 250
        panel_y = 40  # Сдвинута вниз
        panel_width = 230
        panel_height = 150

        # Фон
        pygame.draw.rect(surface, PANEL_GRAY,
                        (panel_x, panel_y, panel_width, panel_height),
                        border_radius=8)
        pygame.draw.rect(surface, PANEL_BORDER,
                         (panel_x, panel_y, panel_width, panel_height),
                         2, border_radius=8)

        # Заголовок
        font_title = pygame.font.SysFont('Arial', 14, bold=True)
        title = font_title.render("СОБЫТИЯ ПОЛЕТА", True, YELLOW)
        surface.blit(title, (panel_x + 20, panel_y + 10))

        # Список событий
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
        # Область для отрисовки траектории
        traj_x = 20
        traj_y = HEIGHT - 180
        traj_width = 450
        traj_height = 120

        # Фон для траектории
        pygame.draw.rect(surface, PANEL_GRAY,
                        (traj_x, traj_y, traj_width, traj_height),
                        border_radius=8)
        pygame.draw.rect(surface, PANEL_BORDER,
                         (traj_x, traj_y, traj_width, traj_height),
                         2, border_radius=8)

        # Заголовок
        font = pygame.font.SysFont('Arial', 12, bold=True)
        title = font.render("ТРАЕКТОРИЯ И ВПП", True, CYAN)
        surface.blit(title, (traj_x + 10, traj_y - 18))

        # Рисуем ВПП (полосу разбега)
        runway_length_px = traj_width * 0.8
        runway_x = traj_x + (traj_width - runway_length_px) / 2
        runway_y = traj_y + traj_height - 20
        runway_width = 15

        # Полоса ВПП
        pygame.draw.rect(surface, (100, 100, 100),
                         (runway_x, runway_y, runway_length_px, runway_width))
        pygame.draw.rect(surface, (150, 150, 150),
                         (runway_x, runway_y, runway_length_px, runway_width), 2)

        # Разметка ВПП
        for i in range(0, int(runway_length_px), 30):
            mark_x = runway_x + i
            pygame.draw.rect(surface, WHITE,
                             (mark_x, runway_y + runway_width // 2 - 2, mark_width, 4))

        # Позиция ракеты на ВПП (если на полосе)
        if self.on_runway:
            # Пройденное расстояние по ВПП
            distance_traveled = min(self.pos[0], self.runway_length)
            runway_progress = distance_traveled / self.runway_length

            rocket_runway_x = runway_x + runway_length_px * runway_progress
            rocket_runway_y = runway_y + runway_width // 2

            # Ракета на ВПП
            pygame.draw.circle(surface, RED, (int(rocket_runway_x), int(rocket_runway_y)), 6)

            # Скорость разбега
            font_small = pygame.font.SysFont('Arial', 10)
            speed_text = font_small.render(f"{np.linalg.norm(self.vel):.0f} м/с", True, YELLOW)
            surface.blit(speed_text, (rocket_runway_x - 20, rocket_runway_y - 20))

        # Преобразуем координаты для отображения траектории
        if self.trajectory_points:
            points = []
            min_time = min(p['time'] for p in self.trajectory_points)
            max_time = max(p['time'] for p in self.trajectory_points)
            max_alt = max(p['y'] for p in self.trajectory_points) / 1000

            # Автомасштабирование
            scale_x = traj_width / max(max_time - min_time, 1)
            scale_y = (traj_height - 40) / max(max_alt, 1)  # Оставляем место для ВПП
        else:
            scale_x = traj_width / 600
            scale_y = (traj_height - 40) / 200

        # Рисуем траекторию
        points = []
        for point in self.trajectory_points:
            screen_x = traj_x + (point['time'] - min_time) * scale_x
            screen_y = traj_y + (traj_height - 20) - point['y'] / 1000 * scale_y

            # Проверка на видимость
            if (traj_x <= screen_x <= traj_x + traj_width and
                    traj_y <= screen_y <= traj_y + traj_height):
                points.append((screen_x, screen_y))

        if len(points) >= 2:
            # Рисуем линию
            for i in range(len(points) - 1):
                alpha = 150 + 105 * (i / len(points))
                color = (*CYAN, int(alpha))

                pygame.draw.line(surface, color, points[i], points[i + 1], 2)

            # Текущая позиция
            if points:
                pygame.draw.circle(surface, RED, (int(points[-1][0]), int(points[-1][1])), 4)

        # Целевая траектория
        if self.show_target_path and self.target_trajectory:
            target_points = []
            for point in self.target_trajectory:
                screen_x = traj_x + (point['time'] - min_time) * scale_x
                screen_y = traj_y + (traj_height - 20) - point['altitude'] / 1000 * scale_y
                target_points.append((screen_x, screen_y))

            if len(target_points) >= 2:
                pygame.draw.lines(surface, (*YELLOW, 100), False, target_points, 1)


class PID:
    """Класс ПИД-регулятора"""

    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, max_output=1.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_output = max_output

        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = 0.0

        # Для анализа
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0

    def calculate(self, error, dt):
        """Расчет управляющего сигнала"""
        # Пропорциональная составляющая
        proportional = self.Kp * error
        self.last_p = proportional

        # Интегральная составляющая
        self.integral += error * dt
        integral_term = self.Ki * self.integral
        self.last_i = integral_term

        # Дифференциальная составляющая
        derivative = 0.0
        if dt > 0:
            derivative = (error - self.previous_error) / dt
        derivative_term = self.Kd * derivative
        self.last_d = derivative_term

        # Суммарный сигнал
        output = proportional + integral_term + derivative_term

        # Ограничение
        output = max(-self.max_output, min(self.max_output, output))

        # Сохранение состояния
        self.previous_error = error

        return output

    def reset(self):
        """Сброс регулятора"""
        self.integral = 0.0
        self.previous_error = 0.0


class ControlPanel:
    """Панель управления симуляцией"""

    def __init__(self, rocket):
        self.rocket = rocket
        self.buttons = []
        self.init_buttons()
    
    def init_buttons(self):
        """Инициализация кнопок управления"""
        button_y = HEIGHT - 80

        # Кнопка запуска
        self.buttons.append({
            'rect': pygame.Rect(50, button_y, 100, 35),
            'text': 'СТАРТ',
            'action': self.start_mission,
            'color': GREEN,
            'active': rocket.mode == "PRELAUNCH"
        })

        # Кнопка паузы
        self.buttons.append({
            'rect': pygame.Rect(160, button_y, 100, 35),
            'text': 'ПАУЗА',
            'action': self.toggle_pause,
            'color': YELLOW,
            'active': True
        })

        # Кнопка сброса
        self.buttons.append({
            'rect': pygame.Rect(270, button_y, 100, 35),
            'text': 'СБРОС',
            'action': self.reset_simulation,
            'color': RED,
            'active': True
        })

        # Кнопка анализа
        self.buttons.append({
            'rect': pygame.Rect(380, button_y, 100, 35),
            'text': 'АНАЛИЗ',
            'action': self.show_analysis,
            'color': PURPLE,
            'active': True
        })

        # Скорость симуляции
        speed_buttons = [
            (500, '1x', 1.0, PANEL_GRAY),
            (550, '2x', 2.0, PANEL_GRAY),
            (600, '5x', 5.0, PANEL_GRAY),
            (650, '10x', 10.0, PANEL_GRAY)
        ]

        for x, text, speed, color in speed_buttons:
            self.buttons.append({
                'rect': pygame.Rect(x, button_y - 40, 45, 25),
                'text': text,
                'action': lambda s=speed: setattr(self.rocket, 'simulation_speed', s),
                'color': PANEL_GRAY,
                'active': True
            })
    
    def start_mission(self):
        """Запуск миссии"""
        if self.rocket.mode == "PRELAUNCH":
            self.rocket.mode = "LAUNCH"
            self.rocket.mission_time = 0.0
            self.rocket.throttle = 1.0  # Включаем тягу!
            print("🚀 ЗАПУСК! Начало миссии.")
            print("✈️ Фаза 1: Разбег по ВПП")

    def toggle_pause(self):
        """Пауза/продолжение"""
        if self.rocket.simulation_speed > 0:
            self.rocket.simulation_speed = 0.0
            print("⏸ ПАУЗА")
        else:
            self.rocket.simulation_speed = 1.0
            print("▶ ПРОДОЛЖЕНИЕ")

    def reset_simulation(self):
        """Сброс симуляции"""
        self.rocket.__init__()
        print("🔄 СИМУЛЯЦИЯ СБРОШЕНА")

    def show_analysis(self):
        """Показать анализ управления"""
        if len(self.rocket.control_history['time']) > 10:
            analysis = PostFlightAnalysis(self.rocket)
            analysis.show_control_analysis()
        else:
            print("⚠ Недостаточно данных для анализа")

    def draw(self, surface):
        """Отрисовка панели управления"""
        # Фон
        pygame.draw.rect(surface, PANEL_GRAY, (0, HEIGHT - 100, WIDTH, 100))
        pygame.draw.line(surface, PANEL_BORDER, (0, HEIGHT - 100), (WIDTH, HEIGHT - 100), 2)

        # Заголовок панели управления
        font_title = pygame.font.SysFont('Arial', 14, bold=True)
        title = font_title.render("УПРАВЛЕНИЕ СИМУЛЯЦИЕЙ", True, WHITE)
        surface.blit(title, (WIDTH // 2 - 80, HEIGHT - 95))

        # Кнопки
        font = pygame.font.SysFont('Arial', 12, bold=True)
        
        for button in self.buttons:
            # Цвет кнопки
            color = button['color']
            if not button['active']:
                color = tuple(c // 2 for c in color)  # Темнее

            # Рисуем кнопку
            pygame.draw.rect(surface, color, button['rect'], border_radius=4)
            pygame.draw.rect(surface, PANEL_BORDER, button['rect'], 2, border_radius=4)

            # Текст
            text = font.render(button['text'], True, WHITE)
            text_rect = text.get_rect(center=button['rect'].center)
            surface.blit(text, text_rect)

        # Индикатор скорости симуляции
        font_small = pygame.font.SysFont('Arial', 10)
        speed_text = font_small.render(f"СКОРОСТЬ: {self.rocket.simulation_speed:.1f}x", True, CYAN)
        surface.blit(speed_text, (500, HEIGHT - 85))

        # Индикатор миссии
        font_status = pygame.font.SysFont('Arial', 12, bold=True)
        if self.rocket.mission_complete:
            mission_text = font_status.render("МИССИЯ ВЫПОЛНЕНА!", True, GREEN)
            surface.blit(mission_text, (WIDTH - 180, HEIGHT - 85))
        elif self.rocket.mode == "LAUNCH":
            if self.rocket.on_runway:
                mission_text = font_status.render("РАЗБЕГ ПО ВПП", True, YELLOW)
            else:
                mission_text = font_status.render("НАБОР ВЫСОТЫ", True, YELLOW)
            surface.blit(mission_text, (WIDTH - 180, HEIGHT - 85))
        elif self.rocket.mode == "PRELAUNCH":
            mission_text = font_status.render("ГОТОВ К ЗАПУСКУ", True, CYAN)
            surface.blit(mission_text, (WIDTH - 180, HEIGHT - 85))

    def handle_click(self, pos):
        """Обработка кликов"""
        for button in self.buttons:
            if button['rect'].collidepoint(pos) and button['active']:
                button['action']()
                return True
        return False


class PostFlightAnalysis:
    """Анализ полета после завершения миссии"""

    def __init__(self, rocket):
        self.rocket = rocket
    
    def show_control_analysis(self):
        """Показать анализ системы управления"""
        if len(self.rocket.control_history['time']) < 10:
            print("⚠ Недостаточно данных для анализа")
            return

        print("\n" + "=" * 80)
        print("📊 АНАЛИЗ СИСТЕМЫ УПРАВЛЕНИЯ ПОЛЕТОМ")
        print("=" * 80)

        # Создаем профессиональные графики
        self.create_analysis_figures()

        # Выводим статистику
        self.print_control_statistics()

        print("\n💡 РЕКОМЕНДАЦИИ:")
        if self.rocket.mission_complete:
            print("✅ Система управления успешно выполнила миссию")
        else:
            print("⚠ Есть проблемы с управлением, требуется настройка ПИД")

        print("=" * 80)

    def create_analysis_figures(self):
        """Создание графиков для анализа"""
        t = self.rocket.control_history['time']

        # ГРАФИК 1: Ошибки и выходы регуляторов
        fig1 = plt.figure(figsize=(12, 8), facecolor=DARK_BLUE_NORM)
        fig1.suptitle('АНАЛИЗ СИСТЕМЫ СТАБИЛИЗАЦИИ', fontsize=16, color='white')

        # Ошибки по каналам
        ax1 = plt.subplot(2, 2, 1)
        ax1.set_facecolor(DARK_BLUE_NORM)
        ax1.set_title('ОШИБКИ УПРАВЛЕНИЯ', fontsize=12, color='white')
        ax1.set_xlabel('Время, с', color='gray')
        ax1.set_ylabel('Ошибка, °', color='gray')
        ax1.grid(True, alpha=0.2)
        ax1.tick_params(colors='gray')

        ax1.plot(t, self.rocket.control_history['pitch_error'],
                 color=BLUE_NORM, linewidth=2, label='Тангаж')
        ax1.plot(t, self.rocket.control_history['yaw_error'],
                 color=GREEN_NORM, linewidth=2, label='Рыскание', alpha=0.8)
        ax1.plot(t, self.rocket.control_history['roll_error'],
                 color=PURPLE_NORM, linewidth=2, label='Крен', alpha=0.6)
        ax1.legend(facecolor=DARK_BLUE_NORM, edgecolor='none', labelcolor='white')

        # Выходы регуляторов
        ax2 = plt.subplot(2, 2, 2)
        ax2.set_facecolor(DARK_BLUE_NORM)
        ax2.set_title('УГЛОВЫЕ СКОРОСТИ', fontsize=12, color='white')
        ax2.set_xlabel('Время, с', color='gray')
        ax2.set_ylabel('Скорость, °/с', color='gray')
        ax2.grid(True, alpha=0.2)
        ax2.tick_params(colors='gray')

        ax2.plot(t, self.rocket.control_history['pitch_output'],
                 color=BLUE_NORM, linewidth=2, label='Тангаж')
        ax2.plot(t, self.rocket.control_history['yaw_output'],
                 color=GREEN_NORM, linewidth=2, label='Рыскание', alpha=0.8)
        ax2.plot(t, self.rocket.control_history['roll_output'],
                 color=PURPLE_NORM, linewidth=2, label='Крен', alpha=0.6)
        ax2.legend(facecolor=DARK_BLUE_NORM, edgecolor='none', labelcolor='white')

        # Гистограмма использования органов управления
        ax3 = plt.subplot(2, 2, 3)
        ax3.set_facecolor(DARK_BLUE_NORM)
        ax3.set_title('ИСПОЛЬЗОВАНИЕ ОРГАНОВ УПРАВЛЕНИЯ', fontsize=12, color='white')
        ax3.set_xlabel('Канал управления', color='gray')
        ax3.set_ylabel('Средний сигнал', color='gray')
        ax3.grid(True, alpha=0.2, color='gray', axis='y')
        ax3.tick_params(colors='gray')

        channels = ['Тангаж', 'Рыскание', 'Крен']
        avg_outputs = [
            np.mean(np.abs(self.rocket.control_history['pitch_output'])),
            np.mean(np.abs(self.rocket.control_history['yaw_output'])),
            np.mean(np.abs(self.rocket.control_history['roll_output']))
        ]

        colors = [BLUE_NORM, GREEN_NORM, PURPLE_NORM]
        bars = ax3.bar(channels, avg_outputs, color=colors)

        for bar, value in zip(bars, avg_outputs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{value:.3f}', ha='center', va='bottom', color='white')

        # График качества стабилизации
        ax4 = plt.subplot(2, 2, 4)
        ax4.set_facecolor(DARK_BLUE_NORM)
        ax4.set_title('КАЧЕСТВО СТАБИЛИЗАЦИИ', fontsize=12, color='white')
        ax4.set_xlabel('Максимальная ошибка, °', color='gray')
        ax4.set_ylabel('Средняя ошибка, °', color='gray')
        ax4.grid(True, alpha=0.2, color='gray')
        ax4.tick_params(colors='gray')

        # Точки для каждого канала
        max_errors = [
            max(abs(e) for e in self.rocket.control_history['pitch_error']),
            max(abs(e) for e in self.rocket.control_history['yaw_error']),
            max(abs(e) for e in self.rocket.control_history['roll_error'])
        ]

        mean_errors = [
            np.mean(np.abs(self.rocket.control_history['pitch_error'])),
            np.mean(np.abs(self.rocket.control_history['yaw_error'])),
            np.mean(np.abs(self.rocket.control_history['roll_error']))
        ]

        for i, (max_err, mean_err, color, label) in enumerate(zip(
                max_errors, mean_errors, [BLUE_NORM, GREEN_NORM, PURPLE_NORM], channels)):
            ax4.scatter(max_err, mean_err, color=color,
                        s=200, alpha=0.7, label=label)
            ax4.text(max_err + 0.1, mean_err, f'{label}\n{max_err:.1f}°/ {mean_err:.1f}°',
                     color='white', fontsize=9)

        ax4.legend(facecolor=DARK_BLUE_NORM, edgecolor='none', labelcolor='white')

        plt.tight_layout(pad=3.0)

        # ГРАФИК 2: Детальный анализ ПИД-регуляторов
        fig2 = plt.figure(figsize=(12, 8), facecolor=DARK_BLUE_NORM)
        fig2.suptitle('АНАЛИЗ ПИД-РЕГУЛЯТОРОВ', fontsize=16, color='white')

        # Составляющие ПИД для тангажа
        ax5 = plt.subplot(2, 2, 1)
        ax5.set_facecolor(DARK_BLUE_NORM)
        ax5.set_title('ПИД-СОСТАВЛЯЮЩИЕ (ТАНГАЖ)', fontsize=12, color='white')
        ax5.set_xlabel('Время, с', color='gray')
        ax5.set_ylabel('Значение', color='gray')
        ax5.grid(True, alpha=0.2, color='gray')
        ax5.tick_params(colors='gray')

        if len(self.rocket.control_history['pitch_p']) > 0:
            ax5.plot(t[:len(self.rocket.control_history['pitch_p'])],
                     self.rocket.control_history['pitch_p'],
                     color=BLUE_NORM, linewidth=2, label='P')
            ax5.plot(t[:len(self.rocket.control_history['pitch_i'])],
                     self.rocket.control_history['pitch_i'],
                     color=GREEN_NORM, linewidth=2, label='I', alpha=0.8)
            ax5.plot(t[:len(self.rocket.control_history['pitch_d'])],
                     self.rocket.control_history['pitch_d'],
                     color=PURPLE_NORM, linewidth=2, label='D', alpha=0.6)
        ax5.legend(facecolor=DARK_BLUE_NORM, edgecolor='none', labelcolor='white')

        # Частотный анализ ошибок
        ax6 = plt.subplot(2, 2, 2)
        ax6.set_facecolor(DARK_BLUE_NORM)
        ax6.set_title('СПЕКТРАЛЬНЫЙ АНАЛИЗ ОШИБОК', fontsize=12, color='white')
        ax6.set_xlabel('Частота, Гц', color='gray')
        ax6.set_ylabel('Амплитуда', color='gray')
        ax6.grid(True, alpha=0.2, color='gray')
        ax6.tick_params(colors='gray')

        # Простой спектральный анализ
        pitch_errors = self.rocket.control_history['pitch_error']
        if len(pitch_errors) > 10:
            N = len(pitch_errors)
            T = t[1] - t[0] if len(t) > 1 else 0.02
            yf = np.fft.fft(pitch_errors)
            xf = np.fft.fftfreq(N, T)[:N // 2]

            ax6.plot(xf[1:], 2.0 / N * np.abs(yf[0:N // 2])[1:],
                     color=CYAN_NORM, linewidth=2)

        # Корреляция между каналам
        ax7 = plt.subplot(2, 2, 3)
        ax7.set_facecolor(DARK_BLUE_NORM)
        ax7.set_title('КОРРЕЛЯЦИЯ МЕЖДУ КАНАЛАМИ', fontsize=12, color='white')
        ax7.set_xlabel('Канал', color='gray')
        ax7.set_ylabel('Канал', color='gray')
        ax7.grid(False)
        ax7.tick_params(colors='gray')

        # Матрица корреляции
        channels_data = np.array([
            self.rocket.control_history['pitch_error'][:1000],
            self.rocket.control_history['yaw_error'][:1000],
            self.rocket.control_history['roll_error'][:1000]
        ])

        if channels_data.shape[1] > 10:
            corr_matrix = np.corrcoef(channels_data)
            im = ax7.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax7)

            # Подписи
            channels = ['Танг.', 'Рыск.', 'Крен']
            ax7.set_xticks(range(3))
            ax7.set_yticks(range(3))
            ax7.set_xticklabels(channels)
            ax7.set_yticklabels(channels)

            # Добавляем значения
            for i in range(3):
                for j in range(3):
                    ax7.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha='center', va='center', color='white')

        # Статистика работы регуляторов
        ax8 = plt.subplot(2, 2, 4)
        ax8.set_facecolor(DARK_BLUE_NORM)
        ax8.set_title('ЭФФЕКТИВНОСТЬ УПРАВЛЕНИЯ', fontsize=12, color='white')

        # Рассчитываем метрики
        pitch_rmse = np.sqrt(np.mean(np.array(self.rocket.control_history['pitch_error']) ** 2))
        yaw_rmse = np.sqrt(np.mean(np.array(self.rocket.control_history['yaw_error']) ** 2))
        roll_rmse = np.sqrt(np.mean(np.array(self.rocket.control_history['roll_error']) ** 2))

        metrics = ['СКО Тангаж', 'СКО Рыскание', 'СКО Крен']
        values = [pitch_rmse, yaw_rmse, roll_rmse]
        colors_metric = [BLUE_NORM, GREEN_NORM, PURPLE_NORM]

        bars = ax8.barh(metrics, values, color=colors_metric)
        ax8.set_xlabel('СКО ошибки, °', color='gray')
        ax8.tick_params(colors='gray')
        ax8.grid(True, alpha=0.2, color='gray', axis='x')

        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax8.text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                     f'{value:.2f}°', ha='left', va='center', color='white')

        plt.tight_layout(pad=3.0)

        # Сохраняем графики
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig1.savefig(f'cascade_analysis_{timestamp}.png', dpi=150, facecolor=DARK_BLUE_NORM)
        print(f"📁 График сохранен: cascade_analysis_{timestamp}.png")
        
        plt.show()
        plt.close('all')

    def print_control_statistics(self):
        """Вывод статистики управления"""
        print("\n📈 СТАТИСТИКА УПРАВЛЕНИЯ:")
        print("-" * 60)

        # Статистика ошибок
        pitch_errors = self.rocket.control_history['pitch_error']
        yaw_errors = self.rocket.control_history['yaw_error']
        roll_errors = self.rocket.control_history['roll_error']

        if len(pitch_errors) > 0:
            print("\n📊 ОШИБКИ СТАБИЛИЗАЦИИ:")
            print(f"{'Канал':<12} {'Макс.':<8} {'Мин.':<8} {'Сред.':<8} {'СКО':<8}")
            print("-" * 44)

            for name, errors in zip(['Тангаж', 'Рыскание', 'Крен'],
                                    [pitch_errors, yaw_errors, roll_errors]):
                max_err = max(errors) if errors else 0
                min_err = min(errors) if errors else 0
                mean_err = np.mean(errors) if errors else 0
                std_err = np.std(errors) if errors else 0

                print(f"{name:<12} {max_err:>7.2f}° {min_err:>7.2f}° "
                      f"{mean_err:>7.2f}° {std_err:>7.2f}°")

        # Статистика управляющих сигналов
        pitch_out = self.rocket.control_history['pitch_output']
        yaw_out = self.rocket.control_history['yaw_output']
        roll_out = self.rocket.control_history['roll_output']

        if len(pitch_out) > 0:
            print("\n🎛 СИГНАЛЫ УПРАВЛЕНИЯ:")
            print(f"{'Канал':<12} {'Сред.':<8} {'Макс.':<8} {'Активность':<12}")
            print("-" * 44)

            for name, outputs in zip(['Тангаж', 'Рыскание', 'Крен'],
                                     [pitch_out, yaw_out, roll_out]):
                avg_out = np.mean(np.abs(outputs)) if outputs else 0
                max_out = max(np.abs(outputs)) if outputs else 0
                activity = avg_out / max_out if max_out > 0 else 0

                print(f"{name:<12} {avg_out:>7.3f}  {max_out:>7.3f}  "
                      f"{activity:>10.1%}")

        # Эффективность стабилизации
        print("\n📐 ЭФФЕКТИВНОСТЬ СТАБИЛИЗАЦИИ:")

        # Время в допуске
        tolerance = 2.0  # градусы
        if len(pitch_errors) > 0:
            pitch_in_tol = sum(1 for e in pitch_errors if abs(e) <= tolerance) / len(pitch_errors)
            yaw_in_tol = sum(1 for e in yaw_errors if abs(e) <= tolerance) / len(yaw_errors)
            roll_in_tol = sum(1 for e in roll_errors if abs(e) <= tolerance) / len(roll_errors)

            print(f"Время в допуске ±{tolerance}°:")
            print(f"  Тангаж: {pitch_in_tol:>6.1%}")
            print(f"  Рыскание: {yaw_in_tol:>4.1%}")
            print(f"  Крен: {roll_in_tol:>8.1%}")

        # Оценка качества
        print("\n⭐ ОЦЕНКА СИСТЕМЫ УПРАВЛЕНИЯ:")

        # Простая оценка
        avg_error = np.mean([np.mean(np.abs(pitch_errors)),
                             np.mean(np.abs(yaw_errors)),
                             np.mean(np.abs(roll_errors))])

        if avg_error < 1.0:
            rating = "ОТЛИЧНО"
            color_code = "🟢"
        elif avg_error < 3.0:
            rating = "ХОРОШО"
            color_code = "🟡"
        elif avg_error < 5.0:
            rating = "УДОВЛЕТВОРИТЕЛЬНО"
            color_code = "🟠"
        else:
            rating = "ТРЕБУЕТ НАСТРОЙКИ"
            color_code = "🔴"

        print(f"{color_code} {rating} (средняя ошибка: {avg_error:.2f}°)")


# Основной цикл
def main():
    rocket = AdvancedRocket()
    control_panel = ControlPanel(rocket)
    
    clock = pygame.time.Clock()
    running = True

    # Основной шрифт
    font = pygame.font.SysFont('Arial', 16)

    print("=" * 80)
    print("🚀 ПРОФЕССИОНАЛЬНЫЙ СИМУЛЯТОР СИСТЕМЫ СТАБИЛИЗАЦИИ САМОЛЕТА-НОСИТЕЛЯ")
    print("=" * 80)
    print("\nУПРАВЛЕНИЕ:")
    print("• Нажмите СТАРТ для запуска (разбег по ВПП -> взлет -> гравитационный разворот)")
    print("• ПАУЗА для приостановки симуляции")
    print("• СБРОС для перезапуска")
    print("• АНАЛИЗ - подробный анализ системы управления")
    print("• 1x/2x/5x/10x для изменения скорости симуляции")
    print("\nФАЗЫ ПОЛЕТА:")
    print("1. Разбег по ВПП (0-10 сек): горизонтально, набор скорости")
    print("2. Взлет (10-20 сек): подъем носа, отрыв от ВПП")
    print("3. Набор высоты (20-60 сек): гравитационный разворот")
    print("4. Выход на орбиту (60+ сек): почти горизонтальный полет")
    print("=" * 80)

    while running:
        # Обработка событий
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
                if event.button == 1:  # Левая кнопка
                    control_panel.handle_click(event.pos)

        # Обновление
        rocket.update()

        # Отрисовка
        rocket.draw(screen)
        control_panel.draw(screen)

        # Информация об управлении
        controls_text = [
            "УПРАВЛЕНИЕ: ПРОБЕЛ - пауза, R - сброс, A - анализ, 1/2/5/0 - скорость",
            "T - траектория, P - целевой путь, C - силы управления, ESC - выход"
        ]

        for i, text in enumerate(controls_text):
            text_surface = font.render(text, True, (200, 200, 200))
            screen.blit(text_surface, (20, HEIGHT - 30 - i * 25))

        # Обновление экрана
        pygame.display.flip()

        # Ограничение FPS
        clock.tick(60)

    # Завершение - показываем итоговый анализ
    if rocket.mission_complete or len(rocket.control_history['time']) > 100:
        print("\n" + "=" * 80)
        print("📊 ЗАВЕРШЕНИЕ СИМУЛЯЦИИ - ИТОГОВЫЙ АНАЛИЗ")
        print("=" * 80)

        analysis = PostFlightAnalysis(rocket)
        analysis.show_control_analysis()

    # Завершение Pygame
    pygame.quit()

    print("\n✅ СИМУЛЯЦИЯ ЗАВЕРШЕНА")
    print("=" * 80)


if __name__ == "__main__":
    main()