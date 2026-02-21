#простой алгоритмический полетный контроллер

import serial
import time
import threading
from collections import deque
import json
import os

class ArduinoInterface:
   
    
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.connected = False
        
        # Данные от Arduino 
        self.telemetry = {
            'pitch': 0.0,        # Тангаж (градусы)
            'roll': 0.0,          # Крен (градусы)
            'yaw': 0.0,           # Рыскание (градусы)
            'pitch_rate': 0.0,    # Скорость тангажа (град/с)
            'roll_rate': 0.0,     # Скорость крена (град/с)
            'yaw_rate': 0.0,      # Скорость рыскания (град/с)
            'target_pitch': 0.0,  # Текущая цель (от нас)
            'throttle': 60.0,     # Текущий газ (%)
            'elevator': 90,       # Положение руля высоты
            'rudder': 90,         # Положение руля направления
            'mode': 0,            # Режим Arduino
            'timestamp': 0        # Время получения данных
        }
        
        
        self.lock = threading.Lock()
        
        
        self.running = False
        self.read_thread = None
        
    def connect(self):
        
        print(f"\nConnecting Arduino on {self.port}...")
        
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1,
                write_timeout=1
            )
            
            
            time.sleep(3)
            
            
            self.serial.write(b"GET\n")
            time.sleep(1)
            
            if self.serial.in_waiting:
                response = self.serial.readline().decode().strip()
                if "DATA:" in response or "READY" in response:
                    self.connected = True
                    print(f"Connected to Arduino on {self.port}")
                    
                    
                    self.running = True
                    self.read_thread = threading.Thread(target=self._read_loop)
                    self.read_thread.daemon = True
                    self.read_thread.start()
                    
                    return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            
        print("Could not connect to Arduino")
        return False
    
    def _read_loop(self):
        
        while self.running:
            try:
                if self.serial and self.serial.in_waiting:
                    line = self.serial.readline().decode().strip()
                    
                    if line.startswith("DATA:"):
                        self._parse_telemetry(line[5:])
                    elif line and not line.startswith(">>>"):
                        print(f"📟 Arduino: {line}")
                        
            except Exception as e:
                
                pass
            
            time.sleep(0.01)
    
    def _parse_telemetry(self, data):
        try:
            parts = data.split(',')
            if len(parts) >= 11:
                with self.lock:
                    self.telemetry['pitch'] = float(parts[0])
                    self.telemetry['roll'] = float(parts[1])
                    self.telemetry['yaw'] = float(parts[2])
                    self.telemetry['pitch_rate'] = float(parts[3])
                    self.telemetry['roll_rate'] = float(parts[4])
                    self.telemetry['yaw_rate'] = float(parts[5])
                    self.telemetry['target_pitch'] = float(parts[6])
                    self.telemetry['throttle'] = float(parts[7])
                    self.telemetry['elevator'] = int(parts[8])
                    self.telemetry['rudder'] = int(parts[9])
                    self.telemetry['mode'] = int(parts[10])
                    self.telemetry['timestamp'] = time.time()
        except Exception as e:
            print(f"Telemetry parse error: {e}")
    
    def send_target(self, pitch, throttle=None):

        if not self.connected:
            return False
        
        if throttle is None:
            throttle = self.telemetry['throttle']
        
        pitch = max(-30, min(30, pitch))
        throttle = max(0, min(100, throttle))
        
        cmd = f"TARGET:{pitch:.1f}:0:{throttle:.0f}\n"
        
        try:
            self.serial.write(cmd.encode())
            return True
        except:
            return False
    
    def send_servo_direct(self, elevator, rudder, throttle):
        
        if not self.connected:
            return False
        
        # Ограничиваем значения
        elevator = max(50, min(130, int(elevator)))
        rudder = max(50, min(130, int(rudder)))
        throttle = max(0, min(180, int(throttle)))
        
        cmd = f"SERVO:{elevator}:{rudder}:{throttle}\n"
        
        try:
            self.serial.write(cmd.encode())
            return True
        except:
            return False
    
    def set_mode(self, mode):
        
        if not self.connected:
            return False
        
        cmd = f"MODE:{mode}\n"
        
        try:
            self.serial.write(cmd.encode())
            print(f"Arduino mode set to: {mode}")
            return True
        except:
            return False
    
    def get_telemetry(self):
        
        with self.lock:
            return self.telemetry.copy()
    
    def close(self):
        
        print("\n Closing connection to Arduino...")
        self.running = False
        
        if self.read_thread:
            self.read_thread.join(timeout=2)
        
        if self.serial and self.serial.is_open:
            
            try:
                self.serial.write(b"MODE:1\n") 
                self.serial.write(b"TARGET:0:0:0\n")
                time.sleep(0.5)
                self.serial.close()
            except:
                pass
            
        print("Connection closed")



class AlgorithmicFlightController:
   
    
    def __init__(self, arduino_interface):
        self.arduino = arduino_interface
        self.name = "ALGORITHMIC"
        
        
        self.Kp = 2.3          
        self.Ki = 0.04         
        self.Kd = 0.35         
        
        
        self.integral = 0
        self.prev_error = 0
        self.prev_output = 0
        
        
        self.error_history = deque(maxlen=10)   
        self.output_history = deque(maxlen=5)   
        
        
        self.alpha_derivative = 0.8     
        self.alpha_measurement = 0.7    
        
        
        self.integral_limit = 50.0
        self.output_limit = 40.0
        
        
        self.zero_crossings = 0
        self.last_error_sign = 0
        
       
        self.control_loop_count = 0
        self.last_calculation_time = time.time()
        
        print(f"\nController initialized")
        print(f"   PID: Kp={self.Kp:.2f}, Ki={self.Ki:.3f}, Kd={self.Kd:.2f}")
    
    def low_pass_filter(self, new_value, old_value):
        
        return self.alpha_measurement * old_value + (1 - self.alpha_measurement) * new_value
    
    def calculate_pid(self, target, current, dt):
        
        
        error = target - current
        
        
        self.error_history.append(error)
        
        
        P = self.Kp * error
        
        
        self.integral += error * dt
        
        
        if abs(P) > self.output_limit * 0.8:
            self.integral -= error * dt * 0.3
        
        
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        I = self.Ki * self.integral
        
        
        if len(self.error_history) >= 2:
            
            raw_derivative = (error - self.prev_error) / max(dt, 0.001)
            
            
            filtered_derivative = self.alpha_derivative * self.prev_output + (1 - self.alpha_derivative) * raw_derivative
            D = self.Kd * filtered_derivative
        else:
            D = 0
        
        
        output = P + I + D
        
        
        output = max(-self.output_limit, min(self.output_limit, output))
        
       
        if error * self.last_error_sign < 0:
            self.zero_crossings += 1
        self.last_error_sign = error if error != 0 else self.last_error_sign
        
       
        self.prev_error = error
        self.prev_output = output
        
        return output
    
    def detect_oscillations(self):
        
        if len(self.error_history) < 5:
            return False
        
        
        if self.zero_crossings > 3:
            self.zero_crossings = 0
            return True
        
        return False
    
    def adapt_parameters(self, error):
   
        if abs(error) > 5.0:
            self.Kp = min(4.0, self.Kp * 1.01)
        
        
        elif abs(error) < 0.5:
            self.Kp = max(1.0, self.Kp * 0.99)
        
        
        if self.detect_oscillations():
            self.Kd = min(1.0, self.Kd * 1.05)
            self.Kp *= 0.98
    
    def calculate_control(self):
        
        
        
        telemetry = self.arduino.get_telemetry()
        
        current_pitch = telemetry['pitch']
        current_time = time.time()
        dt = current_time - self.last_calculation_time
        dt = max(0.01, min(0.1, dt))  
        
       
        target = 0.0
        
        
        control_output = self.calculate_pid(target, current_pitch, dt)
        
        
        error = target - current_pitch
        self.adapt_parameters(error)
        
        
        elevator_pos = 90 + control_output
        elevator_pos = max(50, min(130, int(elevator_pos)))
        
        
        rudder_pos = 90
        
        
        throttle_pos = int(telemetry['throttle'] * 1.8)  # 0-100% -> 0-180
        
        self.control_loop_count += 1
        self.last_calculation_time = current_time
        
        return {
            'elevator': elevator_pos,
            'rudder': rudder_pos,
            'throttle': throttle_pos,
            'control_output': control_output,
            'error': error,
            'Kp': self.Kp,
            'Ki': self.Ki,
            'Kd': self.Kd
        }
    
    def get_status(self):
        
        return {
            'name': self.name,
            'Kp': self.Kp,
            'Ki': self.Ki,
            'Kd': self.Kd,
            'integral': self.integral,
            'zero_crossings': self.zero_crossings,
            'loop_count': self.control_loop_count
        }
    
    def reset(self):
        
        self.integral = 0
        self.prev_error = 0
        self.prev_output = 0
        self.zero_crossings = 0
        self.error_history.clear()
        self.output_history.clear()
        print("reset")



def main():
    
    
    
    print("           FLIGHT CONTROLLER SYSTEM")
    
    
    
    arduino = ArduinoInterface('/dev/ttyUSB0')
    
    if not arduino.connect():
        print("\nFailed to connect to Arduino")
        
        
        return
    
    
    controller = AlgorithmicFlightController(arduino)
    
    
    arduino.set_mode(0)  
    
    print("\nSystem ready!")
    print("   Press Ctrl+C to exit")
    
    
    try:
        
        loop_count = 0
        start_time = time.time()
        
        while True:
            loop_start = time.time()
            
            
            control = controller.calculate_control()
            
            
            arduino.send_servo_direct(
                control['elevator'],
                control['rudder'],
                control['throttle']
            )
            
            
            loop_count += 1
            if loop_count % 50 == 0:  
                telemetry = arduino.get_telemetry()
                
                print(f"\nStatus at {time.strftime('%H:%M:%S')}")
                print(f"   Pitch: {telemetry['pitch']:6.2f}°  |  Error: {control['error']:6.2f}°")
                print(f"   Roll:  {telemetry['roll']:6.2f}°  |  Output: {control['control_output']:6.2f}")
                print(f"   Rate:  {telemetry['pitch_rate']:6.2f}°/s")
                print(f"   PID:   Kp={control['Kp']:.2f}, Ki={control['Ki']:.3f}, Kd={control['Kd']:.2f}")
                print(f"   Throttle: {telemetry['throttle']:.0f}%  |  Elevator: {control['elevator']}")
            
            
            elapsed = time.time() - loop_start
            if elapsed < 0.02:
                time.sleep(0.02 - elapsed)
            
    except KeyboardInterrupt:
        print("\n\nStopping controller...")
    
    finally:
        
        arduino.close()
        print("\nProgram terminated")


if __name__ == "__main__":
    main()