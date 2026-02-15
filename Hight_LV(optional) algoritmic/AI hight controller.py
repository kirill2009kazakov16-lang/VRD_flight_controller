import serial
import time
import numpy as np
import json
import os
from datetime import datetime
from collections import deque

class AdaptivePIDLearner:
    
    
    def __init__(self):
        
        self.Kp = 2.0
        self.Ki = 0.05
        self.Kd = 0.3
        
        
        self.error_history = deque(maxlen=100)  
        self.pid_history = deque(maxlen=50)     
        self.performance_history = deque(maxlen=20)  
        
        
        self.best_Kp = self.Kp
        self.best_Ki = self.Ki
        self.best_Kd = self.Kd
        self.best_performance = float('inf')
        
        
        self.model_file = "adaptive_pid_model.json"
        
        
        self.load_model()
        
        print(f"Adaptive PID Learner initialized")
        print(f"  Initial parameters: Kp={self.Kp:.3f}, Ki={self.Ki:.3f}, Kd={self.Kd:.3f}")
    
    def analyze_performance(self, error, stability):
       
        self.error_history.append(abs(error))
        
        
        if len(self.error_history) >= 10:
            recent_errors = list(self.error_history)[-10:]
            avg_error = np.mean(recent_errors)
            error_std = np.std(recent_errors)
            
            
            performance = avg_error + error_std * 2
            
            self.performance_history.append(performance)
            
            
            self.adapt_pid_parameters(performance, error, stability)
            
            return performance
        
        return None
    
    def adapt_pid_parameters(self, performance, error, stability):
        
        
       
        if len(self.performance_history) >= 3:
            prev_perf = self.performance_history[-2]
            curr_perf = performance
            
            if curr_perf > prev_perf * 1.2: 
                print(f"WARNING: Performance decreased: {prev_perf:.3f} -> {curr_perf:.3f}")
                
                
                if stability < 1.0:  
                    self.Kp *= 1.05  
                    self.Ki *= 0.95  
                else:  
                    self.Kp *= 0.95  
                    self.Kd *= 1.05  
                
                print(f"  New parameters: Kp={self.Kp:.3f}, Ki={self.Ki:.3f}, Kd={self.Kd:.3f}")
            
            elif curr_perf < prev_perf * 0.8:  
                
                if performance < self.best_performance:
                    self.best_performance = performance
                    self.best_Kp = self.Kp
                    self.best_Ki = self.Ki
                    self.best_Kd = self.Kd
                    
                    self.save_model()
                    print(f"NEW RECORD! Performance: {performance:.3f}")
                    print(f"  Best parameters saved")
        
        
        if abs(error) > 3.0:  
            self.Kp = min(self.Kp * 1.02, 5.0)  
        elif abs(error) < 0.3:  
            self.Kp = max(self.Kp * 0.99, 0.5)  
        
        self.pid_history.append((self.Kp, self.Ki, self.Kd))
    
    def get_recommended_pid(self):
       
        return self.Kp, self.Ki, self.Kd
    
    def save_model(self):
        
        model_data = {
            'Kp': float(self.best_Kp),
            'Ki': float(self.best_Ki),
            'Kd': float(self.best_Kd),
            'best_performance': float(self.best_performance),
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'description': 'Adaptive PID for flight controller'
        }
        
        with open(self.model_file, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Model saved to {self.model_file}")
    
    def load_model(self):
        
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'r') as f:
                    model_data = json.load(f)
                
                self.best_Kp = model_data['Kp']
                self.best_Ki = model_data['Ki']
                self.best_Kd = model_data['Kd']
                self.best_performance = model_data['best_performance']
                
                
                self.Kp = self.best_Kp
                self.Ki = self.best_Ki
                self.Kd = self.best_Kd
                
                print(f"Loaded trained model from {model_data['training_date']}")
                print(f"  Best performance: {self.best_performance:.3f}")
                
            except Exception as e:
                print(f"Failed to load model: {e}")
        else:
            print("Model not found, using default parameters")
    
    def reset_learning(self):
        
        self.error_history.clear()
        self.pid_history.clear()
        self.performance_history.clear()
        print("Learning reset")

class IntelligentFlightController:
    
    
    def __init__(self, port='/dev/ttyUSB0'):
        self.serial = None
        self.port = port
        self.connected = False
        
        
        self.pid_learner = AdaptivePIDLearner()
        
        
        self.pitch = 0
        self.target_pitch = 0
        self.mode = 0
        self.is_learning = False
        
        
        self.flight_log = []
        self.start_time = time.time()
        
        
        self.stats = {
            'total_flight_time': 0,
            'best_stability': float('inf'),
            'total_learning_cycles': 0
        }
    
    def connect(self):
        
        print(f"Connecting to {self.port}...")
        
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=115200,
                timeout=2,
                write_timeout=2
            )
            
            time.sleep(3)
            print("Connected to Arduino")
            
            # Проверка связи
            self.send_command("M:0")
            time.sleep(1)
            
            self.connected = True
            return True
            
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def send_command(self, cmd):
        
        if not self.connected or not self.serial:
            return False
        
        try:
            full_cmd = cmd + "\n"
            self.serial.write(full_cmd.encode('utf-8'))
            print(f">>> {cmd}")
            return True
        except Exception as e:
            print(f"Send error: {e}")
            return False
    
    def read_telemetry(self):
        
        if not self.connected or not self.serial:
            return None
        
        try:
            if self.serial.in_waiting:
                line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                
                if line.startswith("TELEM:"):
                    return self.parse_telemetry(line[6:])
                elif line:
                    print(f"Arduino: {line}")
        
        except Exception as e:
            print(f"Read error: {e}")
        
        return None
    
    def parse_telemetry(self, data_str):
        
        try:
            data = {}
            parts = data_str.split(',')
            
            for part in parts:
                if ':' in part:
                    key, value = part.split(':')
                    
                    if key == 'P':
                        data['pitch'] = float(value)
                    elif key == 'TP':
                        data['target_pitch'] = float(value)
                    elif key == 'E':
                        data['error'] = float(value)
                    elif key == 'PID':
                        pid_parts = value.split(':')
                        if len(pid_parts) == 3:
                            data['Kp'] = float(pid_parts[0])
                            data['Ki'] = float(pid_parts[1])
                            data['Kd'] = float(pid_parts[2])
                    elif key == 'M':
                        data['mode'] = int(value)
                    elif key == 'L':
                        data['learning'] = int(value)
            
            return data
            
        except:
            return None
    
    def update_pid_on_arduino(self):
        "
        Kp, Ki, Kd = self.pid_learner.get_recommended_pid()
        
        
        pid_cmd = f"PID:{Kp:.3f}:{Ki:.3f}:{Kd:.3f}"
        self.send_command(pid_cmd)
        
        print(f"PID updated on Arduino: Kp={Kp:.3f}, Ki={Ki:.3f}, Kd={Kd:.3f}")
    
    def learning_mode(self):
        
        print("\n")
        print("LEARNING MODE")
        
        
        
        
        
        self.send_command("M:1")
        self.is_learning = True
        
        
        self.pid_learner.reset_learning()
        
        try:
            learning_cycle = 0
            last_pid_update = time.time()
            last_target_change = time.time()
            
            while True:
                
                data = self.read_telemetry()
                
                if data:
                    self.pitch = data.get('pitch', 0)
                    error = data.get('error', 0)
                    
                    
                    performance = self.pid_learner.analyze_performance(error, abs(error))
                    
                    
                    self.flight_log.append({
                        'time': time.time() - self.start_time,
                        'pitch': self.pitch,
                        'error': error,
                        'Kp': data.get('Kp', 0),
                        'Ki': data.get('Ki', 0),
                        'Kd': data.get('Kd', 0),
                        'performance': performance
                    })
                    
                    
                    if time.time() - last_target_change > 5:
                        
                        new_target = np.random.uniform(-8, 8)
                        self.send_command(f"P:{new_target:.1f}")
                        last_target_change = time.time()
                        print(f"New learning target: {new_target:.1f} deg")
                    
                    
                    if time.time() - last_pid_update > 10:
                        self.update_pid_on_arduino()
                        last_pid_update = time.time()
                        learning_cycle += 1
                
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nLearning completed")
            self.is_learning = False
            
            
            self.pid_learner.save_model()
            
            
            self.send_command("M:0")
            self.send_command("P:0")
            
            
            self.show_learning_stats()
    
    def autopilot_mode(self):
        
        
        print("AUTO MODE")
        
        
        
        
        
        self.pid_learner.load_model()
        
        
        self.update_pid_on_arduino()
        
        
        self.send_command("M:0")
        self.send_command("P:0")
        self.send_command("T:60")
        
        try:
            stability_data = []
            
            while True:
                
                data = self.read_telemetry()
                
                if data:
                    self.pitch = data.get('pitch', 0)
                    error = data.get('error', 0)
                    
                    
                    stability_data.append(abs(error))
                    if len(stability_data) > 20:
                        stability_data.pop(0)
                    
                    
                    if len(stability_data) >= 10:
                        avg_stability = np.mean(stability_data)
                        
                        if avg_stability < self.stats['best_stability']:
                            self.stats['best_stability'] = avg_stability
                            print(f"New stability record: {avg_stability:.3f} deg")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nAutopilot disabled")
    
    def show_learning_stats(self):
        
        if not self.flight_log:
            print("No data for statistics")
            return
        
        print("\n" + "="*60)
        print("LEARNING STATISTICS")
        print("="*60)
        
        errors = [entry['error'] for entry in self.flight_log if 'error' in entry]
        performances = [entry['performance'] for entry in self.flight_log if entry['performance']]
        
        if errors:
            print(f"Total records: {len(self.flight_log)}")
            print(f"Average error: {np.mean(np.abs(errors)):.3f} deg")
            print(f"Max error: {np.max(np.abs(errors)):.3f} deg")
            print(f"Std deviation: {np.std(errors):.3f} deg")
        
        if performances:
            print(f"Best performance: {np.min(performances):.3f}")
            print(f"Worst performance: {np.max(performances):.3f}")
        
        Kp, Ki, Kd = self.pid_learner.get_recommended_pid()
        
        
    
    
    def show_menu(self):
        
        os.system('clear')
        

        print("ML FLIGHT CONTROLLER")

        
        print(f"\n[SYSTEM STATUS]")
        print(f"  Port: {self.port}")
        print(f"  Connection: {'OK' if self.connected else 'FAILED'}")
        print(f"  Mode: {'Learning' if self.is_learning else 'Normal'}")
        
        Kp, Ki, Kd = self.pid_learner.get_recommended_pid()
        print(f"\n[CURRENT PID PARAMETERS]")
        print(f"  Kp: {Kp:.3f}")
        print(f"  Ki: {Ki:.3f}")
        print(f"  Kd: {Kd:.3f}")
        
        print(f"\n[STATISTICS]")
        print(f"  Best stability: {self.stats['best_stability']:.3f} deg")
        print(f"  Learning cycles: {self.stats['total_learning_cycles']}")
        
        print("\n" + "="*60)
        print("[MAIN MENU]")
        print("  1 - LEARNING MODE (system learns to tune PID)")
        print("  2 - AUTOPILOT (uses trained parameters)")
        print("  3 - MANUAL CONTROL")
        print("  4 - CALIBRATE SENSORS")
        print("  5 - TEST SERVOS")
        print("  6 - LEARNING STATISTICS")
        print("  7 - SAVE MODEL")
        print("  8 - LOAD MODEL")
        print("  q - EXIT")
        print("="*60)
    
    def run(self):
        
        print("Starting Intelligent Flight Controller...")
        
        if not self.connect():
            print("Failed to connect to Arduino")
            return
        
        print("\nSystem ready!")
        print("Adaptive PID Learner activated")
        
        try:
            while True:
                self.show_menu()
                
                # чтение телеметрии
                data = self.read_telemetry()
                if data:
                    self.pitch = data.get('pitch', 0)
                    self.target_pitch = data.get('target_pitch', 0)
                    self.mode = data.get('mode', 0)
                
                # Получение команды
                try:
                    choice = input("\nSelect action: ").strip().lower()
                    
                    if choice == '1':
                        self.learning_mode()
                    elif choice == '2':
                        self.autopilot_mode()
                    elif choice == '3':
                        self.manual_control()
                    elif choice == '4':
                        self.send_command("C")
                        print("Sensor calibration started...")
                        time.sleep(6)
                    elif choice == '5':
                        self.send_command("TEST")
                    elif choice == '6':
                        self.show_learning_stats()
                        input("\nPress Enter to continue...")
                    elif choice == '7':
                        self.pid_learner.save_model()
                        input("\nPress Enter to continue...")
                    elif choice == '8':
                        self.pid_learner.load_model()
                        self.update_pid_on_arduino()
                        input("\nPress Enter to continue...")
                    elif choice == 'q':
                        print("Exiting...")
                        self.send_command("M:3")  # Emergency mode
                        self.send_command("T:0")   # Throttle 0
                        break
                
                except KeyboardInterrupt:
                    print("\nInterrupted")
                    break
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            if self.serial and self.serial.is_open:
                self.serial.close()
            print("Program completed")
    
    def manual_control(self):
        print("\nManual control")
        print("Commands: pX (pitch), tX (throttle), mX (mode), q (exit)")
        
        while True:
            try:
                cmd = input("\nCommand: ").strip().lower()
                
                if cmd.startswith('p'):
                    pitch = cmd[1:]
                    self.send_command(f"P:{pitch}")
                elif cmd.startswith('t'):
                    throttle = cmd[1:]
                    self.send_command(f"T:{throttle}")
                elif cmd.startswith('m'):
                    mode = cmd[1:]
                    self.send_command(f"M:{mode}")
                elif cmd == 'q':
                    break
                else:
                    print("Use: p10, t60, m0, q")
            
            except KeyboardInterrupt:
                break

def main():
    
    print("Starting")
    
    controller = IntelligentFlightController()
    controller.run()

if __name__ == "__main__":
    main()