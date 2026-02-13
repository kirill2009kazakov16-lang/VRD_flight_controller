import pygame
import numpy as np
import math
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from datetime import datetime


pygame.init()


WIDTH, HEIGHT = 1400, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ПРОФЕССИОНАЛЬНЫЙ СИМУЛЯТОР: Система стабилизации ракеты-носителя")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_BLUE = (15, 20, 35)
PANEL_GRAY = (40, 45, 60)
PANEL_BORDER = (70, 75, 90)
BLUE = (0, 150, 255)  
GREEN = (0, 200, 100)  
YELLOW = (255, 200, 0)  
ORANGE = (255, 140, 0)  
RED = (255, 60, 60)  # Критично
CYAN = (0, 200, 220)  
PURPLE = (180, 100, 220)  


BLUE_NORM = (0 / 255, 150 / 255, 255 / 255)
GREEN_NORM = (0 / 255, 200 / 255, 100 / 255)
YELLOW_NORM = (255 / 255, 200 / 255, 0 / 255)
ORANGE_NORM = (255 / 255, 140 / 255, 0 / 255)
RED_NORM = (255 / 255, 60 / 255, 60 / 255)
CYAN_NORM = (0 / 255, 200 / 255, 220 / 255)
PURPLE_NORM = (180 / 255, 100 / 255, 220 / 255)
WHITE_NORM = (1.0, 1.0, 1.0)
PANEL_GRAY_NORM = (40 / 255, 45 / 255, 60 / 255)
PANEL_BORDER_NORM = (70 / 255, 75 / 255, 90 / 255)
DARK_BLUE_NORM = (15 / 255, 20 / 255, 35 / 255)


class AdvancedRocket:
    

    def __init__(self):
        
        self.mass = 25000.0  # кг
        self.length = 32.4  # м (как Союз)
        self.diameter = 2.95  # м

        # Моменты инерции
        self.Ixx = self.mass * self.diameter ** 2 / 4
        self.Iyy = self.mass * (3 * self.diameter ** 2 + self.length ** 2) / 12
        self.Izz = self.Iyy

        # Аэродинамика
        self.S_ref = math.pi * (self.diameter / 2) ** 2
        self.Cd = 0.25  # Сопротивление
        self.Cl_alpha = 3.0  # Подъемная сила

        # Двигательная установка
        self.thrust_max = 250000.0  # Н (как РД-107)
        self.mass_flow = 1000.0  # кг/с
        self.isp = 800.0  # Удельный импульс

        # Система управления
        self.max_deflection = math.radians(18)  # Макс. угол рулей

        
        self.pos = np.array([0.0, 0.0, 0.0])  # x, y, z (м)
        self.vel = np.array([0.0, 0.0, 0.0])  # м/с
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  
        self.omega = np.array([0.0, 0.0, 0.0])  # Угл. скорость (рад/с)

        
        self.throttle = 0.0
        self.elevator_cmd = 0.0
        self.rudder_cmd = 0.0
        self.aileron_cmd = 0.0

        
        self.target_trajectory = []
        self.generate_target_trajectory()

        
        self.target_pitch = 0.0  
        self.target_yaw = 0.0
        self.target_roll = 0.0
        self.target_altitude = 100000.0  
        self.target_velocity = 80.0  

        # ПИД
        self.pitch_pid = PID(Kp=0.9, Ki=0.0, Kd=1.7, max_output=1.0)
        self.yaw_pid = PID(Kp=0.6, Ki=0.015, Kd=0.12, max_output=1.0)
        self.roll_pid = PID(Kp=0.4, Ki=0.01, Kd=0.08, max_output=1.0)

       
        self.time = 0.0
        self.dt = 0.02  # 50 Гц
        self.simulation_speed = 1.0
        self.mission_time = 0.0

        
        self.telemetry = {
            'time': [], 'altitude': [], 'velocity': [], 'mach': [],
            'pitch': [], 'yaw': [], 'roll': [], 'alpha': [], 'beta': [],
            'thrust': [], 'mass': [], 'q_dyn': [], 'accel': [],
            'pitch_error': [], 'yaw_error': [], 'roll_error': [],
            'elevator_cmd': [], 'rudder_cmd': [], 'aileron_cmd': [],
            'throttle_cmd': [], 'pid_p': [], 'pid_i': [], 'pid_d': []
        }

        
        self.mode = "PRELAUNCH"
        self.events = []
        self.event_times = []

        
        self.trajectory_points = []
        self.max_trajectory_points = 500

        self.figures = {}
        self.graph_images = {}
        self.init_figures()

        
        self.show_trajectory = True
        self.show_target_path = True
        self.show_control_forces = True

        
        self.control_history = {
            'time': [],
            'pitch_error': [], 'yaw_error': [], 'roll_error': [],
            'pitch_output': [], 'yaw_output': [], 'roll_output': [],
            'pitch_p': [], 'pitch_i': [], 'pitch_d': [],
            'yaw_p': [], 'yaw_i': [], 'yaw_d': [],
            'roll_p': [], 'roll_i': [], 'roll_d': []
        }

        
        self.mission_complete = False

        
        self.takeoff_velocity = 90.0  # Скорость отрыва (м/с)
        self.rotation_speed = 70.0  # Скорость начала подъема носа (м/с)
        self.on_runway = True  
        self.runway_length = 3000.0  # Длина ВПП (м)

    def generate_target_trajectory(self):
        
        self.target_trajectory = []
        for t in np.linspace(0, 300, 100):  
            if t < 10:
                pitch = 0.0  
            elif t < 20:
                pitch = 15.0  # Взлетный угол
            elif t < 60:
                pitch = 15.0 + (t - 20) * 1.5  
            elif t < 120:
                pitch = 75.0 - (t - 60) * 0.5  
            elif t < 180:
                pitch = 45.0 - (t - 120) * 0.3 
            else:
                pitch = 5.0 

            altitude = 100 * t  
            self.target_trajectory.append({
                'time': t,
                'pitch': pitch,
                'altitude': altitude
            })

    def init_figures(self):
        
        plt.style.use('dark_background')

        
        self.figures['trajectory'] = plt.figure(figsize=(5, 3.5), facecolor=DARK_BLUE_NORM)

        
        self.figures['dynamics'] = plt.figure(figsize=(5, 3.5), facecolor=DARK_BLUE_NORM)

        
        self.figures['control'] = plt.figure(figsize=(5, 3.5), facecolor=DARK_BLUE_NORM)

        
        self.figures['aerodynamics'] = plt.figure(figsize=(5, 3.5), facecolor=DARK_BLUE_NORM)

    def update(self):
        
        dt = self.dt * self.simulation_speed

        
        if self.pos[2] >= self.target_altitude and not self.mission_complete:
            self.mission_complete = True
            self.mode = "MISSION_COMPLETE"
            print(f"Mission complite! Достигнута высота {self.pos[2] / 1000:.1f} км")
            print(f"Time: {self.mission_time:.1f} сек")
            print(f"Speed: {np.linalg.norm(self.vel):.0f} м/с")

        if self.mission_complete:
            return

        
        self.update_flight_program()

        
        self.stabilization_system(dt)

        
        self.physics_update(dt)

        
        self.collect_telemetry()

        
        self.collect_control_history()

        
        self.time += dt
        self.mission_time += dt

        
        self.check_mission_events()

        
        self.update_trajectory_visualization()
        self.update_figures()

    def update_flight_program(self):
        
        if self.mode == "LAUNCH":
            
            t = self.mission_time
            velocity = np.linalg.norm(self.vel)

            
            if t < 10:
                self.target_pitch = 0.0  
                self.target_yaw = 0.0
                self.target_roll = 0.0
                self.throttle = 1.0  
                self.on_runway = True

                
                if velocity >= self.rotation_speed:
                    self.target_pitch = 10.0  

            
            elif t < 20:
                if self.pos[2] < 10:  
                    self.target_pitch = 15.0  
                else:
                    self.target_pitch = 20.0  
                self.throttle = 1.0
                self.on_runway = False

            
            elif t < 60:
                altitude_km = self.pos[2] / 1000
                
                self.target_pitch = min(75.0, 20.0 + altitude_km * 3.0)
                self.throttle = 1.0

            
            elif t < 120:
                self.target_pitch = max(30.0, 75.0 - (t - 60) * 0.5)
                self.throttle = 1.0

            
            elif t < 180:
                self.target_pitch = max(5.0, 30.0 - (t - 120) * 0.2)
                self.throttle = 0.8

            
            else:
                self.target_pitch = 5.0
                
                altitude_km = self.pos[2] / 1000
                if altitude_km > 80:
                    self.throttle = 0.6
                else:
                    self.throttle = 0.8

    def stabilization_system(self, dt):
        
        
        pitch, yaw, roll = self.get_euler_angles()

        # Ошибки
        pitch_error = math.radians(self.target_pitch) - pitch
        yaw_error = math.radians(self.target_yaw) - yaw
        roll_error = math.radians(self.target_roll) - roll

        # ПИД
        self.elevator_cmd = self.pitch_pid.calculate(pitch_error, dt)
        self.rudder_cmd = self.yaw_pid.calculate(yaw_error, dt)
        self.aileron_cmd = self.roll_pid.calculate(roll_error, dt)

        
        self.elevator_cmd = max(-1.0, min(1.0, self.elevator_cmd))
        self.rudder_cmd = max(-1.0, min(1.0, self.rudder_cmd))
        self.aileron_cmd = max(-1.0, min(1.0, self.aileron_cmd))

        
        self.telemetry['pitch_error'].append(math.degrees(pitch_error))
        self.telemetry['yaw_error'].append(math.degrees(yaw_error))
        self.telemetry['roll_error'].append(math.degrees(roll_error))
        self.telemetry['elevator_cmd'].append(self.elevator_cmd)
        self.telemetry['rudder_cmd'].append(self.rudder_cmd)
        self.telemetry['aileron_cmd'].append(self.aileron_cmd)
        self.telemetry['throttle_cmd'].append(self.throttle)

    def collect_control_history(self):
        
        self.control_history['time'].append(self.mission_time)

        # Ошибки
        pitch, yaw, roll = self.get_euler_angles()
        pitch_error = math.radians(self.target_pitch) - pitch
        yaw_error = math.radians(self.target_yaw) - yaw
        roll_error = math.radians(self.target_roll) - roll

        self.control_history['pitch_error'].append(math.degrees(pitch_error))
        self.control_history['yaw_error'].append(math.degrees(yaw_error))
        self.control_history['roll_error'].append(math.degrees(roll_error))

        
        self.control_history['pitch_output'].append(self.elevator_cmd)
        self.control_history['yaw_output'].append(self.rudder_cmd)
        self.control_history['roll_output'].append(self.aileron_cmd)

        
        if hasattr(self.pitch_pid, 'last_p'):
            self.control_history['pitch_p'].append(self.pitch_pid.last_p)
            self.control_history['pitch_i'].append(self.pitch_pid.last_i)
            self.control_history['pitch_d'].append(self.pitch_pid.last_d)
        else:
            self.control_history['pitch_p'].append(0)
            self.control_history['pitch_i'].append(0)
            self.control_history['pitch_d'].append(0)

        
        max_history = 2000
        for key in self.control_history:
            if len(self.control_history[key]) > max_history:
                self.control_history[key] = self.control_history[key][-max_history:]

    def physics_update(self, dt):
        

        
        thrust_mag = self.throttle * self.get_thrust_at_altitude()

        
        thrust_body = np.array([thrust_mag, 0.0, 0.0])

        
        g = 9.81
        gravity_inertial = np.array([0.0, 0.0, -g * self.mass])

        
        aero_forces = self.calculate_aerodynamic_forces()

        
        ground_force = np.array([0.0, 0.0, 0.0])
        if self.on_runway and self.pos[2] <= 0.1:
            
            ground_normal = g * self.mass
            ground_force = np.array([0.0, 0.0, ground_normal])

            
            if np.linalg.norm(self.vel) > 0.1:
                friction_force = -0.02 * ground_normal * (self.vel / np.linalg.norm(self.vel))
                ground_force += friction_force

        
        thrust_inertial = self.body_to_inertial(thrust_body)

        
        total_force = thrust_inertial + gravity_inertial + aero_forces + ground_force

        
        acceleration = total_force / self.mass

        
        self.vel += acceleration * dt
        self.pos += self.vel * dt

        if self.on_runway and self.pos[2] < 0:
            self.pos[2] = 0
            if self.vel[2] < 0:
                self.vel[2] = 0

        
        if self.on_runway and np.linalg.norm(self.vel) >= self.takeoff_velocity:
            self.on_runway = False
            

        

        
        control_moments = np.array([
            self.aileron_cmd * 40000.0,  
            self.elevator_cmd * 60000.0,  
            self.rudder_cmd * 30000.0  
        ])

        
        damping_moments = -0.15 * self.omega * np.array([self.Ixx, self.Iyy, self.Izz])

        
        total_moment = control_moments + damping_moments

        
        angular_acceleration = np.array([
            total_moment[0] / self.Ixx,
            total_moment[1] / self.Iyy,
            total_moment[2] / self.Izz
        ])

        
        self.omega += angular_acceleration * dt

        
        self.integrate_orientation(dt)

        
        if self.throttle > 0:
            self.mass -= self.mass_flow * self.throttle * dt
            self.mass = max(self.mass, 7500.0)  # Сухая масса

        
        if self.pos[2] < 0 and not self.on_runway:
            self.pos[2] = 0
            self.vel[2] = max(self.vel[2], 0)

    def integrate_orientation(self, dt):
        
        omega = self.omega
        Omega = np.array([
            [0, -omega[0], -omega[1], -omega[2]],
            [omega[0], 0, omega[2], -omega[1]],
            [omega[1], -omega[2], 0, omega[0]],
            [omega[2], omega[1], -omega[0], 0]
        ])

        
        q_dot = 0.5 * Omega @ self.q

        
        self.q += q_dot * dt

        
        norm = np.linalg.norm(self.q)
        if norm > 0:
            self.q /= norm

    def body_to_inertial(self, vector_body):
        
        w, x, y, z = self.q

        
        R = np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]
        ])

        return R @ vector_body

    def calculate_aerodynamic_forces(self):
        
        velocity_mag = np.linalg.norm(self.vel)

        if velocity_mag < 1.0:
            return np.array([0.0, 0.0, 0.0])

        
        rho = self.get_atmospheric_density(self.pos[2])

        if rho < 1e-6:
            return np.array([0.0, 0.0, 0.0])

        
        dynamic_pressure = 0.5 * rho * velocity_mag ** 2

        
        alpha = self.get_angle_of_attack()

        
        Cd = self.Cd + 0.15 * abs(alpha)

        
        Cl = self.Cl_alpha * alpha

        
        velocity_dir = self.vel / velocity_mag

        
        drag_mag = dynamic_pressure * self.S_ref * Cd
        drag_force = -drag_mag * velocity_dir

        
        lift_direction = np.array([0, 0, 1])  

        
        velocity_component = np.dot(lift_direction, velocity_dir) * velocity_dir
        lift_direction_perp = lift_direction - velocity_component

        
        lift_dir_norm = np.linalg.norm(lift_direction_perp)
        if lift_dir_norm > 0:
            lift_direction = lift_direction_perp / lift_dir_norm

        lift_mag = dynamic_pressure * self.S_ref * Cl
        lift_force = lift_mag * lift_direction

        return drag_force + lift_force

    def get_angle_of_attack(self):
        
        if np.linalg.norm(self.vel) < 1.0:
            return 0.0

        
        velocity_body = self.inertial_to_body(self.vel)

        
        u = velocity_body[0] if abs(velocity_body[0]) > 0.1 else 0.1
        w = velocity_body[2]

        return math.atan2(w, u)

    def inertial_to_body(self, vector_inertial):
        
        w, x, y, z = self.q

        
        R_inv = np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y + 2 * w * z, 2 * x * z - 2 * w * y],
            [2 * x * y - 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z + 2 * w * x],
            [2 * x * z + 2 * w * y, 2 * y * z - 2 * w * x, 1 - 2 * x * x - 2 * y * y]
        ])

        return R_inv @ vector_inertial

    def get_atmospheric_density(self, altitude):
        
        if altitude < 11000:  
            T = 288.15 - 0.0065 * altitude
            p = 101325 * (T / 288.15) ** 5.255
        elif altitude < 20000:  # Стратосфера
            T = 216.65
            p = 22632 * math.exp(-0.0001577 * (altitude - 11000))
        else:  
            T = 216.65 + 0.001 * (altitude - 20000)
            p = 5474 * (216.65 / T) ** 34.163

        return p / (287.05 * T)

    def get_thrust_at_altitude(self):
        
        altitude_km = self.pos[2] / 1000

        if altitude_km < 30:
            
            return self.thrust_max
        else:
            
            vacuum_factor = 1.0 + altitude_km * 0.01
            return self.thrust_max * min(vacuum_factor, 1.2)

    def get_mach_number(self):
        
        velocity = np.linalg.norm(self.vel)

        
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
        
        w, x, y, z = self.q

        
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        
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

        
        rho = self.get_atmospheric_density(self.pos[2])
        velocity = np.linalg.norm(self.vel)
        self.telemetry['q_dyn'].append(0.5 * rho * velocity ** 2)

        
        aero_forces = self.calculate_aerodynamic_forces()
        accel_mag = np.linalg.norm(aero_forces) / self.mass
        self.telemetry['accel'].append(accel_mag / 9.81)

        
        for key in self.telemetry:
            if len(self.telemetry[key]) > 1000:
                self.telemetry[key].pop(0)

    def check_mission_events(self):
        
        altitude_km = self.pos[2] / 1000
        mach = self.get_mach_number()
        q_dyn = self.telemetry['q_dyn'][-1] if self.telemetry['q_dyn'] else 0
        velocity = np.linalg.norm(self.vel)

        events = [
            (5, "V1", velocity >= 60, "Speed v1"),
            (8, "ROTATION", velocity >= self.rotation_speed, ""),
            (10, "LIFTOFF", self.pos[2] > 1.0 and not self.on_runway, ""),
            (15, "GEAR UP", self.pos[2] > 10, ""),
            (60, "MACH 1", mach >= 0.95, ""),
            (65, "TRANSONIC", 0.95 <= mach <= 1.05, ""),
            (70, "SUPERSONIC", mach >= 1.05, ""),
            (120, "MACH 2", mach >= 2.0, "Достигнута 2М"),
            (200, "MACH 3", mach >= 3.0, "Достигнута 3М"),
            (100, "", altitude_km >= 100, "Граница космоса"),
        ]

        for check_time, name, condition, description in events:
            if self.mission_time >= check_time and name not in self.events:
                if condition:
                    self.events.append(name)
                    self.event_times.append(self.mission_time)
                    print(f"🎯 {name}: {description} (T+{self.mission_time:.1f}с)")

    def update_trajectory_visualization(self):
        
        self.trajectory_points.append({
            'x': self.pos[0],
            'y': self.pos[2],  # Высота
            'time': self.time
        })

        
        if len(self.trajectory_points) > self.max_trajectory_points:
            self.trajectory_points.pop(0)

    def update_figures(self):
        
        if len(self.telemetry['time']) < 2:
            return

        t = self.telemetry['time']

        
        fig = self.figures['trajectory']
        fig.clear()

        ax1 = fig.add_subplot(121)
        ax1.set_facecolor(DARK_BLUE_NORM)
        ax1.set_title('ВЫСОТА ПОЛЕТА', fontsize=9, color='white', pad=8)
        ax1.set_xlabel('Время, с', color='gray', fontsize=8)
        ax1.set_ylabel('Высота, км', color='gray', fontsize=8)
        ax1.grid(True, alpha=0.2, color='gray')
        ax1.tick_params(colors='gray', labelsize=7)

        
        if len(t) > 0:
            ax1.plot(t, [h / 1000 for h in self.telemetry['altitude']],
                     color=CYAN_NORM, linewidth=1.5, label='Текущая')

            
            if self.show_target_path:
                target_t = [p['time'] for p in self.target_trajectory]
                target_h = [p['altitude'] / 1000 for p in self.target_trajectory]
                ax1.plot(target_t, target_h, '--', color=YELLOW_NORM,
                         linewidth=1, alpha=0.7, label='Целевая')

        ax1.legend(fontsize=7, facecolor=DARK_BLUE_NORM, edgecolor='none', labelcolor='white')

        ax2 = fig.add_subplot(122)
        ax2.set_facecolor(DARK_BLUE_NORM)
        ax2.set_title('ОРИЕНТАЦИЯ', fontsize=9, color='white', pad=8)
        ax2.set_xlabel('Время, с', color='gray', fontsize=8)
        ax2.set_ylabel('Угол, °', color='gray', fontsize=8)
        ax2.grid(True, alpha=0.2, color='gray')
        ax2.tick_params(colors='gray', labelsize=7)

        if len(t) > 0:
            ax2.plot(t, self.telemetry['pitch'], color=BLUE_NORM, linewidth=1.5, label='Тангаж')
            ax2.plot(t, self.telemetry['yaw'], color=GREEN_NORM, linewidth=1.5, label='Рыскание', alpha=0.8)
            ax2.axhline(y=self.target_pitch, color=RED_NORM, linestyle=':', linewidth=1, alpha=0.5, label='Цель')

        ax2.legend(fontsize=7, facecolor=DARK_BLUE_NORM, edgecolor='none', labelcolor='white')

        fig.tight_layout(pad=1.5)
        self.save_figure_to_image('trajectory', fig)

        
        fig = self.figures['dynamics']
        fig.clear()

        ax1 = fig.add_subplot(111)
        ax1.set_facecolor(DARK_BLUE_NORM)
        ax1.set_title('ДИНАМИКА ПОЛЕТА', fontsize=9, color='white', pad=8)
        ax1.set_xlabel('Время, с', color='gray', fontsize=8)
        ax1.set_ylabel('Скорость, м/с', color=CYAN_NORM, fontsize=8)
        ax1.grid(True, alpha=0.2, color='gray')
        ax1.tick_params(colors='gray', labelsize=7)

        if len(t) > 0:
            ax1.plot(t, self.telemetry['velocity'], color=CYAN_NORM, linewidth=1.5, label='Скорость')

            ax2 = ax1.twinx()
            ax2.set_ylabel('Перегрузка, g', color=ORANGE_NORM, fontsize=8)
            ax2.plot(t, self.telemetry['accel'], color=ORANGE_NORM, linewidth=1.5, label='Перегрузка', alpha=0.8)
            ax2.tick_params(colors='gray', labelsize=7)

            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7,
                       facecolor=DARK_BLUE_NORM, edgecolor='none', labelcolor='white')

        fig.tight_layout(pad=1.5)
        self.save_figure_to_image('dynamics', fig)

        
        fig = self.figures['control']
        fig.clear()

        ax = fig.add_subplot(111)
        ax.set_facecolor(DARK_BLUE_NORM)
        ax.set_title('СИСТЕМА УПРАВЛЕНИЯ', fontsize=9, color='white', pad=8)
        ax.set_xlabel('Время, с', color='gray', fontsize=8)
        ax.set_ylabel('Сигнал управления', color='gray', fontsize=8)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.2, color='gray')
        ax.tick_params(colors='gray', labelsize=7)

        if len(t) > 0:
            
            if len(self.telemetry['elevator_cmd']) > 0:
                
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

        
        fig = self.figures['aerodynamics']
        fig.clear()

        ax1 = fig.add_subplot(111)
        ax1.set_facecolor(DARK_BLUE_NORM)
        ax1.set_title('АЭРОДИНАМИКА', fontsize=9, color='white', pad=8)
        ax1.set_xlabel('Время, с', color='gray', fontsize=8)
        ax1.set_ylabel('Число Маха', color=CYAN_NORM, fontsize=8)
        ax1.grid(True, alpha=0.2, color='gray')
        ax1.tick_params(colors='gray', labelsize=7)

        if len(t) > 0 and len(self.telemetry['mach']) > 0:
            n_points = min(len(t), len(self.telemetry['mach']))
            t_plot = t[-n_points:]
            mach = self.telemetry['mach'][-n_points:]

            ax1.plot(t_plot, mach, color=CYAN_NORM, linewidth=1.5, label='Число Маха')

            ax2 = ax1.twinx()
            ax2.set_ylabel('Угол атаки, °', color=ORANGE_NORM, fontsize=8)

            if len(self.telemetry['alpha']) > 0:
                alpha = self.telemetry['alpha'][-n_points:]
                ax2.plot(t_plot, alpha, color=ORANGE_NORM, linewidth=1.5, label='Угол атаки', alpha=0.8)

            ax2.tick_params(colors='gray', labelsize=7)

            
            ax1.axhline(y=1.0, color=RED_NORM, linestyle='--', linewidth=1, alpha=0.5, label='M=1.0')

            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7,
                       facecolor=DARK_BLUE_NORM, edgecolor='none', labelcolor='white')

        fig.tight_layout(pad=1.5)
        self.save_figure_to_image('aerodynamics', fig)

    def save_figure_to_image(self, name, fig):
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, facecolor=DARK_BLUE_NORM,
                    edgecolor='none', bbox_inches='tight')
        buf.seek(0)

       
        image = pygame.image.load(buf)

        
        target_width = 400
        target_height = 280
        image = pygame.transform.scale(image, (target_width, target_height))

        self.graph_images[name] = image

        
        buf.close()

    def get_status_text(self):
        
        pitch, yaw, roll = self.get_euler_angles()
        mach = self.get_mach_number()
        altitude_km = self.pos[2] / 1000
        velocity = np.linalg.norm(self.vel)

        status_lines = []
        status_lines.append(f"РЕЖИМ: {self.mode:20s} T+{self.mission_time:7.1f}с")
        status_lines.append(f"ВЫСОТА: {altitude_km:7.1f} км{' ':5}СКОРОСТЬ: {velocity:7.0f} м/с")
        status_lines.append(f"ТАНГАЖ: {math.degrees(pitch):6.1f}°{' ':8}ЧИСЛО МАХА: {mach:6.2f}")
        status_lines.append(
            f"РЫСКАНИЕ: {math.degrees(yaw):6.1f}°{' ':7}ПЕРЕГРУЗКА: {self.telemetry['accel'][-1] if self.telemetry['accel'] else 0:5.2f} g")
        status_lines.append(
            f"КРЕН: {math.degrees(roll):6.1f}°{' ':10}УА: {self.telemetry['alpha'][-1] if self.telemetry['alpha'] else 0:5.1f}°")
        status_lines.append(
            f"МАССА: {self.mass / 1000:6.1f} т{' ':7}ТЯГА: {self.get_thrust_at_altitude() * self.throttle / 1000:6.0f} кН")
        status_lines.append(f"ЦЕЛЬ ТАНГАЖ: {self.target_pitch:5.1f}°{' ':6}ДРОССЕЛЬ: {self.throttle * 100:4.0f}%")
        status_lines.append(
            f"НА ВПП: {'ДА' if self.on_runway else 'НЕТ':<10} ОТРЫВ: {'ГОТОВ' if velocity >= self.takeoff_velocity else 'РАЗБЕГ'}")

        return status_lines

    def draw(self, surface):
        
        
        surface.fill(DARK_BLUE)

        
        self.draw_graphs(surface)

        
        self.draw_status_panel(surface)
        self.draw_events_panel(surface)

        
        if self.show_trajectory:
            self.draw_trajectory_and_runway(surface)

        
        self.draw_rocket_and_controls(surface)

    def draw_rocket_and_controls(self, surface):
        
        rocket_x = 150
        rocket_y = HEIGHT - 250

        
        self.draw_rocket(surface, rocket_x, rocket_y)

    def draw_rocket(self, surface, x, y):
        
        rocket_width = 30
        rocket_height = 200

        
        pitch, _, _ = self.get_euler_angles()
        pitch_deg = math.degrees(pitch)

        
        rocket_surf = pygame.Surface((rocket_width + 20, rocket_height + 20), pygame.SRCALPHA)

        
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

        
        wing_width = 40
        wing_height = 10
        wing_y = rocket_height // 3

        
        pygame.draw.rect(rocket_surf, PANEL_GRAY,
                         (0, wing_y, wing_width, wing_height))
        
        pygame.draw.rect(rocket_surf, PANEL_GRAY,
                         (rocket_width - wing_width, wing_y, wing_width, wing_height))

        
        if self.throttle > 0:
            flame_height = 30 + 20 * self.throttle
            flame_width = 15

            
            for offset in [-flame_width, flame_width]:
                flame_points = [
                    (rocket_width // 2 + offset, rocket_height),
                    (rocket_width // 2 + offset - flame_width // 2, rocket_height + flame_height),
                    (rocket_width // 2 + offset, rocket_height + flame_height * 0.8),
                    (rocket_width // 2 + offset + flame_width // 2, rocket_height + flame_height)
                ]

                
                colors = [(255, 255, 0), (255, 140, 0), (255, 60, 60)]
                for i in range(len(colors)):
                    flame_surf = pygame.Surface((flame_width, flame_height), pygame.SRCALPHA)
                    pygame.draw.polygon(flame_surf, (*colors[i], 150),
                                        [(0, flame_height), (flame_width // 2, 0), (flame_width, flame_height)])
                    rocket_surf.blit(flame_surf, (rocket_width // 2 + offset - flame_width // 2, rocket_height))

        
        rotated_rocket = pygame.transform.rotate(rocket_surf, -pitch_deg)
        rocket_rect = rotated_rocket.get_rect(center=(x, y))

        
        surface.blit(rotated_rocket, rocket_rect)

        
        target_y = y - self.target_pitch * 2
        pygame.draw.circle(surface, YELLOW, (x, int(target_y)), 8, 2)
        pygame.draw.line(surface, YELLOW, (x - 10, target_y), (x + 10, target_y), 1)
        pygame.draw.line(surface, YELLOW, (x, target_y - 10), (x, target_y + 10), 1)

        
        font = pygame.font.SysFont('Arial', 14, bold=True)
        text = font.render("САМОЛЕТ-НОСИТЕЛЬ", True, CYAN)
        surface.blit(text, (x - 70, y - 120))

        
        if self.on_runway:
            font_small = pygame.font.SysFont('Arial', 12)
            runway_text = font_small.render(f"РАЗБЕГ: {np.linalg.norm(self.vel):.0f}/{self.takeoff_velocity:.0f} м/с",
                                            True, GREEN if np.linalg.norm(self.vel) >= self.rotation_speed else YELLOW)
            surface.blit(runway_text, (x - 60, y + 100))

    def draw_graphs(self, surface):
        
        graph_width = 400
        graph_height = 280

        
        graph_positions = {
            'trajectory': (WIDTH // 2 - 420, 40), 
            'dynamics': (WIDTH // 2 + 10, 40),  
            'control': (WIDTH // 2 - 420, 340),  
            'aerodynamics': (WIDTH // 2 + 10, 340)  
        }

        
        graph_titles = {
            'trajectory': 'ТРАЕКТОРИЯ И ОРИЕНТАЦИЯ',
            'dynamics': 'ДИНАМИКА ПОЛЕТА',
            'control': 'СИСТЕМА УПРАВЛЕНИЯ',
            'aerodynamics': 'АЭРОДИНАМИКА'
        }

        for name, pos in graph_positions.items():
            if name in self.graph_images:
                
                graph_rect = pygame.Rect(pos[0] - 5, pos[1] - 5,
                                         graph_width + 10,
                                         graph_height + 10)
                pygame.draw.rect(surface, PANEL_GRAY, graph_rect, border_radius=6)
                pygame.draw.rect(surface, PANEL_BORDER, graph_rect, 2, border_radius=6)

               
                font = pygame.font.SysFont('Arial', 12, bold=True)
                title = font.render(graph_titles[name], True, CYAN)
                surface.blit(title, (pos[0] + 10, pos[1] - 20))

                
                surface.blit(self.graph_images[name], pos)

    def draw_status_panel(self, surface):
        
        panel_x = 20
        panel_y = 40 
        panel_width = 450
        panel_height = 250  

        
        pygame.draw.rect(surface, PANEL_GRAY,
                         (panel_x, panel_y, panel_width, panel_height),
                         border_radius=12)
        pygame.draw.rect(surface, PANEL_BORDER,
                         (panel_x, panel_y, panel_width, panel_height),
                         2, border_radius=12)

       
        font_title = pygame.font.SysFont('Arial', 18, bold=True)
        title = font_title.render("СТАТУС СИСТЕМЫ", True, BLUE)
        surface.blit(title, (panel_x + 20, panel_y + 15))

        
        font = pygame.font.SysFont('Consolas', 12)
        status_lines = self.get_status_text()

        for i, line in enumerate(status_lines):
            color = WHITE
            if 'РЕЖИМ:' in line:
                color = GREEN if self.mode == "LAUNCH" else YELLOW
            elif 'ЦЕЛЬ' in line:
                color = CYAN
            elif 'РУЛЬ' in line:
                color = PURPLE
            elif 'НА ВПП:' in line:
                color = GREEN if self.on_runway else YELLOW
            elif 'ОТРЫВ:' in line:
                velocity = np.linalg.norm(self.vel)
                color = GREEN if velocity >= self.takeoff_velocity else YELLOW

            text = font.render(line, True, color)
            surface.blit(text, (panel_x + 20, panel_y + 50 + i * 22))

    def draw_events_panel(self, surface):
        
        panel_x = WIDTH - 250
        panel_y = 40  
        panel_width = 230
        panel_height = 150

       
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

                event_text = font.render(f"{time_str} - {event}", True, WHITE)
                surface.blit(event_text, (panel_x + 15, panel_y + 35 + i * 18))
        else:
            no_events = font.render("Событий пока нет", True, (150, 150, 150))
            surface.blit(no_events, (panel_x + 15, panel_y + 50))

    def draw_trajectory_and_runway(self, surface):
        
        traj_x = 20
        traj_y = HEIGHT - 180
        traj_width = 450
        traj_height = 120

        
        pygame.draw.rect(surface, PANEL_GRAY,
                         (traj_x, traj_y, traj_width, traj_height),
                         border_radius=8)
        pygame.draw.rect(surface, PANEL_BORDER,
                         (traj_x, traj_y, traj_width, traj_height),
                         2, border_radius=8)

        
        font = pygame.font.SysFont('Arial', 12, bold=True)
        title = font.render("ТРАЕКТОРИЯ И ВПП", True, CYAN)
        surface.blit(title, (traj_x + 10, traj_y - 18))

        
        runway_length_px = traj_width * 0.8
        runway_x = traj_x + (traj_width - runway_length_px) / 2
        runway_y = traj_y + traj_height - 20
        runway_width = 15

        
        pygame.draw.rect(surface, (100, 100, 100),
                         (runway_x, runway_y, runway_length_px, runway_width))
        pygame.draw.rect(surface, (150, 150, 150),
                         (runway_x, runway_y, runway_length_px, runway_width), 2)

        
        for i in range(0, int(runway_length_px), 30):
            mark_x = runway_x + i
            mark_width = 15
            pygame.draw.rect(surface, WHITE,
                             (mark_x, runway_y + runway_width // 2 - 2, mark_width, 4))

        
        if self.on_runway:
            
            distance_traveled = min(self.pos[0], self.runway_length)
            runway_progress = distance_traveled / self.runway_length

            rocket_runway_x = runway_x + runway_length_px * runway_progress
            rocket_runway_y = runway_y + runway_width // 2

           
            pygame.draw.circle(surface, RED, (int(rocket_runway_x), int(rocket_runway_y)), 6)

            
            font_small = pygame.font.SysFont('Arial', 10)
            speed_text = font_small.render(f"{np.linalg.norm(self.vel):.0f} м/с", True, YELLOW)
            surface.blit(speed_text, (rocket_runway_x - 20, rocket_runway_y - 20))

        
        if self.trajectory_points:
            min_time = min(p['time'] for p in self.trajectory_points)
            max_time = max(p['time'] for p in self.trajectory_points)
            max_alt = max(p['y'] for p in self.trajectory_points) / 1000

            
            scale_x = traj_width / max(max_time - min_time, 1)
            scale_y = (traj_height - 40) / max(max_alt, 1) 
        else:
            scale_x = traj_width / 600
            scale_y = (traj_height - 40) / 200

        
        points = []
        for point in self.trajectory_points:
            screen_x = traj_x + (point['time'] - min_time) * scale_x
            screen_y = traj_y + (traj_height - 20) - point['y'] / 1000 * scale_y

           
            if (traj_x <= screen_x <= traj_x + traj_width and
                    traj_y <= screen_y <= traj_y + traj_height):
                points.append((screen_x, screen_y))

        if len(points) >= 2:
            
            for i in range(len(points) - 1):
                alpha = 150 + 105 * (i / len(points))
                color = (*CYAN, int(alpha))

                pygame.draw.line(surface, color, points[i], points[i + 1], 2)

            
            if points:
                pygame.draw.circle(surface, RED, (int(points[-1][0]), int(points[-1][1])), 4)

        
        if self.show_target_path and self.target_trajectory:
            target_points = []
            for point in self.target_trajectory:
                screen_x = traj_x + (point['time'] - min_time) * scale_x
                screen_y = traj_y + (traj_height - 20) - point['altitude'] / 1000 * scale_y
                target_points.append((screen_x, screen_y))

            if len(target_points) >= 2:
                pygame.draw.lines(surface, (*YELLOW, 100), False, target_points, 1)


class PID:
    

    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, max_output=1.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_output = max_output

        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = 0.0

        
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0

    def calculate(self, error, dt):
        
        proportional = self.Kp * error
        self.last_p = proportional

        
        self.integral += error * dt
        integral_term = self.Ki * self.integral
        self.last_i = integral_term

        
        derivative = 0.0
        if dt > 0:
            derivative = (error - self.previous_error) / dt
        derivative_term = self.Kd * derivative
        self.last_d = derivative_term

        
        output = proportional + integral_term + derivative_term

        
        output = max(-self.max_output, min(self.max_output, output))

        
        self.previous_error = error

        return output

    def reset(self):
        
        self.integral = 0.0
        self.previous_error = 0.0


class ControlPanel:
    

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
                'color': color,
                'active': True
            })

    def start_mission(self):
        
        if self.rocket.mode == "PRELAUNCH":
            self.rocket.mode = "LAUNCH"
            self.rocket.mission_time = 0.0
            self.rocket.throttle = 1.0  
            

    def toggle_pause(self):
        
        if self.rocket.simulation_speed > 0:
            self.rocket.simulation_speed = 0.0
            
        else:
            self.rocket.simulation_speed = 1.0
            

    def reset_simulation(self):
        
        self.rocket.__init__()
        print("Reset")

    def show_analysis(self):
        
        if len(self.rocket.control_history['time']) > 10:
            analysis = PostFlightAnalysis(self.rocket)
            analysis.show_control_analysis()
        else:
            print("low data")

    def draw(self, surface):
        
        
        pygame.draw.rect(surface, PANEL_GRAY, (0, HEIGHT - 100, WIDTH, 100))
        pygame.draw.line(surface, PANEL_BORDER, (0, HEIGHT - 100), (WIDTH, HEIGHT - 100), 2)

        
        font_title = pygame.font.SysFont('Arial', 14, bold=True)
        title = font_title.render("sim control", True, WHITE)
        surface.blit(title, (WIDTH // 2 - 80, HEIGHT - 95))

        
        font = pygame.font.SysFont('Arial', 12, bold=True)

        for button in self.buttons:
            
            color = button['color']
            if not button['active']:
                color = tuple(c // 2 for c in color)  

            
            pygame.draw.rect(surface, color, button['rect'], border_radius=4)
            pygame.draw.rect(surface, PANEL_BORDER, button['rect'], 2, border_radius=4)

            
            text = font.render(button['text'], True, WHITE)
            text_rect = text.get_rect(center=button['rect'].center)
            surface.blit(text, text_rect)

        
        font_small = pygame.font.SysFont('Arial', 10)
        speed_text = font_small.render(f"Скорость: {self.rocket.simulation_speed:.1f}x", True, CYAN)
        surface.blit(speed_text, (500, HEIGHT - 85))

        
        font_status = pygame.font.SysFont('Arial', 12, bold=True)
        if self.rocket.mission_complete:
            mission_text = font_status.render("Mission complite!", True, GREEN)
            surface.blit(mission_text, (WIDTH - 180, HEIGHT - 85))
        elif self.rocket.mode == "LAUNCH":
            if self.rocket.on_runway:
                mission_text = font_status.render("РАЗБЕГ ПО ВПП", True, YELLOW)
            else:
                mission_text = font_status.render("НАБОР ВЫСОТЫ", True, YELLOW)
            surface.blit(mission_text, (WIDTH - 180, HEIGHT - 85))
        elif self.rocket.mode == "PRELAUNCH":
            mission_text = font_status.render("ready to start", True, CYAN)
            surface.blit(mission_text, (WIDTH - 180, HEIGHT - 85))

    def handle_click(self, pos):
        
        for button in self.buttons:
            if button['rect'].collidepoint(pos) and button['active']:
                button['action']()
                return True
        return False


class PostFlightAnalysis:
    

    def __init__(self, rocket):
        self.rocket = rocket

    def show_control_analysis(self):
        
        if len(self.rocket.control_history['time']) < 10:
            print("low data")
            return

        
        print("analiz...")
        

        
        self.create_analysis_figures()

        
        self.print_control_statistics()

        

    def create_analysis_figures(self):
        
        t = self.rocket.control_history['time']

        
        fig1 = plt.figure(figsize=(12, 8), facecolor=DARK_BLUE_NORM)
        fig1.suptitle('АНАЛИЗ СИСТЕМЫ СТАБИЛИЗАЦИИ', fontsize=16, color='white')

        
        ax1 = plt.subplot(2, 2, 1)
        ax1.set_facecolor(DARK_BLUE_NORM)
        ax1.set_title('ОШИБКИ СТАБИЛИЗАЦИИ', fontsize=12, color='white')
        ax1.set_xlabel('Время, с', color='gray')
        ax1.set_ylabel('Ошибка, °', color='gray')
        ax1.grid(True, alpha=0.2, color='gray')
        ax1.tick_params(colors='gray')

        ax1.plot(t, self.rocket.control_history['pitch_error'],
                 color=BLUE_NORM, linewidth=2, label='Тангаж')
        ax1.plot(t, self.rocket.control_history['yaw_error'],
                 color=GREEN_NORM, linewidth=2, label='Рыскание', alpha=0.8)
        ax1.plot(t, self.rocket.control_history['roll_error'],
                 color=PURPLE_NORM, linewidth=2, label='Крен', alpha=0.6)
        ax1.legend(facecolor=DARK_BLUE_NORM, edgecolor='none', labelcolor='white')

        
        ax2 = plt.subplot(2, 2, 2)
        ax2.set_facecolor(DARK_BLUE_NORM)
        ax2.set_title('СИГНАЛЫ УПРАВЛЕНИЯ', fontsize=12, color='white')
        ax2.set_xlabel('Время, с', color='gray')
        ax2.set_ylabel('Выход регулятора', color='gray')
        ax2.set_ylim(-1.1, 1.1)
        ax2.grid(True, alpha=0.2, color='gray')
        ax2.tick_params(colors='gray')

        ax2.plot(t, self.rocket.control_history['pitch_output'],
                 color=BLUE_NORM, linewidth=2, label='Тангаж')
        ax2.plot(t, self.rocket.control_history['yaw_output'],
                 color=GREEN_NORM, linewidth=2, label='Рыскание', alpha=0.8)
        ax2.plot(t, self.rocket.control_history['roll_output'],
                 color=PURPLE_NORM, linewidth=2, label='Крен', alpha=0.6)
        ax2.legend(facecolor=DARK_BLUE_NORM, edgecolor='none', labelcolor='white')

        
        ax3 = plt.subplot(2, 2, 3)
        ax3.set_facecolor(DARK_BLUE_NORM)
        ax3.set_title('use control', fontsize=12, color='white')
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

        
        ax4 = plt.subplot(2, 2, 4)
        ax4.set_facecolor(DARK_BLUE_NORM)
        ax4.set_title('stabing', fontsize=12, color='white')
        ax4.set_xlabel('Максимальная ошибка, °', color='gray')
        ax4.set_ylabel('Средняя ошибка, °', color='gray')
        ax4.grid(True, alpha=0.2, color='gray')
        ax4.tick_params(colors='gray')

        
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

        в
        fig2 = plt.figure(figsize=(12, 8), facecolor=DARK_BLUE_NORM)
        fig2.suptitle('analiz pid', fontsize=16, color='white')

        
        ax5 = plt.subplot(2, 2, 1)
        ax5.set_facecolor(DARK_BLUE_NORM)
        ax5.set_title('pid (ТАНГАЖ)', fontsize=12, color='white')
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

        
        ax6 = plt.subplot(2, 2, 2)
        ax6.set_facecolor(DARK_BLUE_NORM)
        ax6.set_title('analiz', fontsize=12, color='white')
        ax6.set_xlabel('Частота, Гц', color='gray')
        ax6.set_ylabel('Амплитуда', color='gray')
        ax6.grid(True, alpha=0.2, color='gray')
        ax6.tick_params(colors='gray')

        
        pitch_errors = self.rocket.control_history['pitch_error']
        if len(pitch_errors) > 10:
            N = len(pitch_errors)
            T = t[1] - t[0] if len(t) > 1 else 0.02
            yf = np.fft.fft(pitch_errors)
            xf = np.fft.fftfreq(N, T)[:N // 2]

            ax6.plot(xf[1:], 2.0 / N * np.abs(yf[0:N // 2])[1:],
                     color=CYAN_NORM, linewidth=2)

        
        ax7 = plt.subplot(2, 2, 3)
        ax7.set_facecolor(DARK_BLUE_NORM)
        ax7.set_title('Korrel', fontsize=12, color='white')
        ax7.set_xlabel('Канал', color='gray')
        ax7.set_ylabel('Канал', color='gray')
        ax7.grid(False)
        ax7.tick_params(colors='gray')

       
        channels_data = np.array([
            self.rocket.control_history['pitch_error'][:1000],
            self.rocket.control_history['yaw_error'][:1000],
            self.rocket.control_history['roll_error'][:1000]
        ])

        if channels_data.shape[1] > 10:
            corr_matrix = np.corrcoef(channels_data)
            im = ax7.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax7)

            
            channels = ['Танг.', 'Рыск.', 'Крен']
            ax7.set_xticks(range(3))
            ax7.set_yticks(range(3))
            ax7.set_xticklabels(channels)
            ax7.set_yticklabels(channels)

            
            for i in range(3):
                for j in range(3):
                    ax7.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha='center', va='center', color='white')

        
        ax8 = plt.subplot(2, 2, 4)
        ax8.set_facecolor(DARK_BLUE_NORM)
        ax8.set_title('Effectinosy', fontsize=12, color='white')

        
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

        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig1.savefig(f'control_analysis_1_{timestamp}.png', dpi=150, facecolor=DARK_BLUE_NORM)
        fig2.savefig(f'control_analysis_2_{timestamp}.png', dpi=150, facecolor=DARK_BLUE_NORM)

        print(f"\n📁 Графики сохранены как:")
        print(f"   control_analysis_1_{timestamp}.png")
        print(f"   control_analysis_2_{timestamp}.png")

        plt.show()
        plt.close('all')

    def print_control_statistics(self):
        
        print("\nStats control:")
        print("-" * 60)

        
        pitch_errors = self.rocket.control_history['pitch_error']
        yaw_errors = self.rocket.control_history['yaw_error']
        roll_errors = self.rocket.control_history['roll_error']

        if len(pitch_errors) > 0:
            print("\nError stab:")
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

        
        pitch_out = self.rocket.control_history['pitch_output']
        yaw_out = self.rocket.control_history['yaw_output']
        roll_out = self.rocket.control_history['roll_output']

        if len(pitch_out) > 0:
            print("\nsignals:")
            print(f"{'Канал':<12} {'Сред.':<8} {'Макс.':<8} {'Активность':<12}")
            print("-" * 44)

            for name, outputs in zip(['Тангаж', 'Рыскание', 'Крен'],
                                     [pitch_out, yaw_out, roll_out]):
                avg_out = np.mean(np.abs(outputs)) if outputs else 0
                max_out = max(np.abs(outputs)) if outputs else 0
                activity = avg_out / max_out if max_out > 0 else 0

                print(f"{name:<12} {avg_out:>7.3f}  {max_out:>7.3f}  "
                      f"{activity:>10.1%}")

       
        print("\neffectinost:")

        
        tolerance = 2.0  
        if len(pitch_errors) > 0:
            pitch_in_tol = sum(1 for e in pitch_errors if abs(e) <= tolerance) / len(pitch_errors)
            yaw_in_tol = sum(1 for e in yaw_errors if abs(e) <= tolerance) / len(yaw_errors)
            roll_in_tol = sum(1 for e in roll_errors if abs(e) <= tolerance) / len(roll_errors)

            print(f"Время в допуске ±{tolerance}°:")
            print(f"  Тангаж: {pitch_in_tol:>6.1%}")
            print(f"  Рыскание: {yaw_in_tol:>4.1%}")
            print(f"  Крен: {roll_in_tol:>8.1%}")

        
        print("\nSystem is...")

        
        avg_error = np.mean([np.mean(np.abs(pitch_errors)),
                             np.mean(np.abs(yaw_errors)),
                             np.mean(np.abs(roll_errors))])

        if avg_error < 1.0:
            rating = "good+"
            color_code = "🟢"
        elif avg_error < 3.0:
            rating = "good-"
            color_code = "🟡"
        elif avg_error < 5.0:
            rating = "normal"
            color_code = "🟠"
        else:
            rating = "bad"
            color_code = "🔴"

        print(f"{color_code} {rating} (средняя ошибка: {avg_error:.2f}°)")



def main():
    global rocket

    rocket = AdvancedRocket()
    control_panel = ControlPanel(rocket)

    clock = pygame.time.Clock()
    running = True

    
    font = pygame.font.SysFont('Arial', 16)

    print("=" * 80)
    print("🚀 RocetSim")

    print("=" * 80)

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
                elif event.key == pygame.K_1:
                    rocket.simulation_speed = 1.0
                elif event.key == pygame.K_2:
                    rocket.simulation_speed = 2.0
                elif event.key == pygame.K_5:
                    rocket.simulation_speed = 5.0
                elif event.key == pygame.K_0:
                    rocket.simulation_speed = 10.0
                elif event.key == pygame.K_t:
                    rocket.show_trajectory = not rocket.show_trajectory
                elif event.key == pygame.K_p:
                    rocket.show_target_path = not rocket.show_target_path
                elif event.key == pygame.K_c:
                    rocket.show_control_forces = not rocket.show_control_forces

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  
                    control_panel.handle_click(event.pos)

        
        rocket.update()

        
        rocket.draw(screen)
        control_panel.draw(screen)

        
        controls_text = [
            "УПРАВЛЕНИЕ: ПРОБЕЛ - пауза, R - сброс, A - анализ, 1/2/5/0 - скорость",
            "T - траектория, P - целевой путь, C - силы управления, ESC - выход"
        ]

        for i, text in enumerate(controls_text):
            text_surface = font.render(text, True, (200, 200, 200))
            screen.blit(text_surface, (20, HEIGHT - 30 - i * 25))

        
        pygame.display.flip()

        
        clock.tick(60)

    
    if rocket.mission_complete or len(rocket.control_history['time']) > 100:
        
        print("Sim complite - analize")
        

        analysis = PostFlightAnalysis(rocket)
        analysis.show_control_analysis()

    
    pygame.quit()

    print("\n")
    


if __name__ == "__main__":
    main()