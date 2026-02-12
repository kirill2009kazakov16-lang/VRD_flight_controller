// airplane_cascade_pid_uart_rpi.ino
#include <Wire.h>
#include <Servo.h>
#include <MPU6050_tockn.h>

// ===== НАСТРОЙКИ ПИНОВ =====
#define ELEVATOR_PIN 3     // Руль высоты
#define RUDDER_PIN 2       // Руль направления
#define THROTTLE_PIN 9     // ESC мотора
#define STATUS_LED 13
#define BUZZER_PIN 10

// ===== ОГРАНИЧЕНИЯ =====
#define ELEVATOR_MIN 60
#define ELEVATOR_MAX 120
#define ELEVATOR_NEUTRAL 90

#define RUDDER_MIN 60
#define RUDDER_MAX 120
#define RUDDER_NEUTRAL 90

#define SERVO_SMOOTH_FACTOR 0.2

// ===== UART ПРОТОКОЛ С RPI =====
#define UART_BAUD_RATE 500000  // Высокая скорость для быстрой передачи
#define TELEMETRY_PACKET_SIZE 64
#define COMMAND_PACKET_SIZE 32

// Команды от RPi
enum RPiCommand {
  CMD_NONE = 0,
  CMD_SET_PID_PARAMS,      // Установка PID параметров
  CMD_SET_TARGETS,         // Установка целевых значений
  CMD_SET_MODE,           // Смена режима
  CMD_CALIBRATE,          // Калибровка
  CMD_REQUEST_TELEMETRY,  // Запрос телеметрии
  CMD_ENABLE_CASCADE,     // Включить каскадный PID
  CMD_DISABLE_CASCADE,    // Выключить каскадный PID
  CMD_UPLOAD_ML_MODEL,    // Загрузка ML модели (параметров)
  CMD_EMERGENCY_STOP      // Аварийная остановка
};

// Структура пакета данных для RPi
struct TelemetryPacket {
  uint32_t timestamp;
  float pitch_angle;
  float roll_angle;
  float yaw_rate;
  float pitch_rate;
  float roll_rate;
  float pitch_inner_error;
  float pitch_outer_error;
  float yaw_error;
  float elevator_output;
  float rudder_output;
  float throttle;
  uint8_t flight_mode;
  uint8_t pid_mode;  // 0=обычный, 1=каскадный
  uint16_t checksum;
};

// Структура команды от RPi
struct CommandPacket {
  uint8_t command;
  float param1;
  float param2;
  float param3;
  float param4;
  uint16_t checksum;
};

// ===== КАСКАДНЫЙ ПИД =====
struct CascadePID {
  // Внешний контур (угол)
  float outer_Kp, outer_Ki, outer_Kd;
  float outer_integral;
  float outer_prev_error;
  float outer_max_output;
  
  // Внутренний контур (скорость)
  float inner_Kp, inner_Ki, inner_Kd;
  float inner_integral;
  float inner_prev_error;
  float inner_max_output;
  
  // Состояние
  float outer_output;  // Целевая скорость для внутреннего контура
  float inner_output;  // Выход на сервопривод
  bool enabled;
};

// ===== СЕРВОПРИВОДЫ =====
Servo elevatorServo;
Servo rudderServo;
Servo throttleServo;

// ===== ДАТЧИК =====
MPU6050 mpu6050(Wire);

// ===== ПЕРЕМЕННЫЕ =====
float pitchAngle = 0, rollAngle = 0;
float pitchRate = 0, rollRate = 0, yawRate = 0;
float prevPitchAngle = 0, prevRollAngle = 0;

// Позиции серв
float currentElevatorPos = ELEVATOR_NEUTRAL;
float currentRudderPos = RUDDER_NEUTRAL;

// ===== ПИД РЕГУЛЯТОРЫ =====

// Каскадный ПИД для тангажа
CascadePID pitchCascadePID = {
  // Внешний контур (угол)
  2.0, 0.05, 0.3, 0, 0, 20,
  // Внутренний контур (скорость)
  1.5, 0.1, 0.2, 0, 0, 30,
  // Состояние
  0, 0, true
};

// Простой ПИД для рыскания
struct SimplePID {
  float Kp, Ki, Kd;
  float integral;
  float prev_error;
  float max_output;
};

SimplePID yawPID = {1.2, 0.03, 0.15, 0, 0, 25};

// Прямой ПИД для тангажа (как запасной)
SimplePID pitchSimplePID = {2.5, 0.08, 0.4, 0, 0, 30};

// ===== ЦЕЛЕВЫЕ ЗНАЧЕНИЯ =====
float targetPitch = 0;
float targetYawRate = 0;
float targetThrottle = 0;
float targetPitchRate = 0;  // Для каскадного ПИД

// ===== РЕЖИМЫ =====
enum FlightMode {
  MODE_MANUAL,           // Ручное управление по UART
  MODE_STABILIZE,        // Стабилизация с простым PID
  MODE_CASCADE_STABILIZE,// Стабилизация с каскадным PID
  MODE_AUTO_TAKEOFF,     // Автоматический взлет
  MODE_AUTO_LEVEL,       // Автоуровень
  MODE_AUTO_TURN,        // Автоповороты
  MODE_CALIBRATION,      // Калибровка
  MODE_EMERGENCY         // Аварийный
};

FlightMode currentMode = MODE_CALIBRATION;
bool useCascadePID = true;  // Использовать каскадный ПИД

// ===== ФИЛЬТРЫ =====
class LowPassFilter {
private:
  float value;
  float alpha;
  
public:
  LowPassFilter(float alpha = 0.9) : value(0), alpha(alpha) {}
  
  float update(float newValue) {
    value = alpha * value + (1 - alpha) * newValue;
    return value;
  }
  
  float getValue() { return value; }
  void reset(float newValue = 0) { value = newValue; }
};

LowPassFilter pitchRateFilter(0.85);
LowPassFilter rollRateFilter(0.85);
LowPassFilter yawRateFilter(0.9);

// ===== UART БУФЕРЫ =====
uint8_t telemetryBuffer[TELEMETRY_PACKET_SIZE];
uint8_t commandBuffer[COMMAND_PACKET_SIZE];
uint8_t commandIndex = 0;

// ===== ФУНКЦИИ =====

uint16_t calculateChecksum(const uint8_t* data, size_t length) {
  uint16_t checksum = 0;
  for (size_t i = 0; i < length - 2; i++) {  // Исключаем поле checksum
    checksum += data[i];
  }
  return checksum;
}

void sendTelemetryToRPi() {
  static uint32_t lastTelemetryTime = 0;
  uint32_t currentTime = micros();
  
  // Отправляем телеметрию с частотой 100 Гц
  if (currentTime - lastTelemetryTime < 10000) return;
  lastTelemetryTime = currentTime;
  
  TelemetryPacket packet;
  packet.timestamp = currentTime;
  packet.pitch_angle = pitchAngle;
  packet.roll_angle = rollAngle;
  packet.yaw_rate = yawRate;
  packet.pitch_rate = pitchRate;
  packet.roll_rate = rollRate;
  packet.pitch_inner_error = targetPitchRate - pitchRate;
  packet.pitch_outer_error = targetPitch - pitchAngle;
  packet.yaw_error = targetYawRate - yawRate;
  packet.elevator_output = currentElevatorPos - ELEVATOR_NEUTRAL;
  packet.rudder_output = currentRudderPos - RUDDER_NEUTRAL;
  packet.throttle = targetThrottle;
  packet.flight_mode = currentMode;
  packet.pid_mode = useCascadePID ? 1 : 0;
  
  // Рассчитываем checksum
  packet.checksum = calculateChecksum((uint8_t*)&packet, sizeof(packet));
  
  // Отправляем пакет
  Serial.write((uint8_t*)&packet, sizeof(packet));
}

void parseRPiCommand() {
  while (Serial.available()) {
    uint8_t byte = Serial.read();
    
    // Ищем начало пакета (0xFF 0xAA)
    static bool gotFirstByte = false;
    if (!gotFirstByte && byte == 0xFF) {
      gotFirstByte = true;
      commandIndex = 0;
    } else if (gotFirstByte && byte == 0xAA) {
      // Начало пакета найдено
      commandBuffer[commandIndex++] = 0xFF;
      commandBuffer[commandIndex++] = 0xAA;
      gotFirstByte = false;
    } else if (gotFirstByte) {
      gotFirstByte = false;
    } else if (commandIndex > 0) {
      // Заполняем буфер
      commandBuffer[commandIndex++] = byte;
      
      // Полный пакет получен
      if (commandIndex >= sizeof(CommandPacket)) {
        CommandPacket* cmd = (CommandPacket*)commandBuffer;
        
        // Проверяем checksum
        uint16_t checksum = calculateChecksum(commandBuffer, sizeof(CommandPacket));
        if (checksum == cmd->checksum) {
          processCommand(cmd);
        }
        
        commandIndex = 0;
      }
    }
  }
}

void processCommand(CommandPacket* cmd) {
  switch (cmd->command) {
    case CMD_SET_PID_PARAMS:
      if (cmd->param1 == 0) {  // Pitch cascade
        pitchCascadePID.outer_Kp = cmd->param2;
        pitchCascadePID.outer_Ki = cmd->param3;
        pitchCascadePID.outer_Kd = cmd->param4;
      } else if (cmd->param1 == 1) {  // Pitch cascade inner
        pitchCascadePID.inner_Kp = cmd->param2;
        pitchCascadePID.inner_Ki = cmd->param3;
        pitchCascadePID.inner_Kd = cmd->param4;
      } else if (cmd->param1 == 2) {  // Yaw PID
        yawPID.Kp = cmd->param2;
        yawPID.Ki = cmd->param3;
        yawPID.Kd = cmd->param4;
      }
      break;
      
    case CMD_SET_TARGETS:
      targetPitch = cmd->param1;
      targetYawRate = cmd->param2;
      targetThrottle = cmd->param3;
      break;
      
    case CMD_SET_MODE:
      currentMode = (FlightMode)((int)cmd->param1);
      break;
      
    case CMD_ENABLE_CASCADE:
      useCascadePID = true;
      break;
      
    case CMD_DISABLE_CASCADE:
      useCascadePID = false;
      break;
      
    case CMD_CALIBRATE:
      calibrateSensors();
      break;
      
    case CMD_EMERGENCY_STOP:
      currentMode = MODE_EMERGENCY;
      break;
      
    case CMD_UPLOAD_ML_MODEL:
      // Здесь можно загрузить параметры ML модели
      // Например, для адаптивного PID
      break;
  }
}

void calibrateSensors() {
  Serial.println("Калибровка датчиков...");
  digitalWrite(STATUS_LED, HIGH);
  
  mpu6050.calcGyroOffsets(true);
  
  // Сброс фильтров
  pitchRateFilter.reset(0);
  rollRateFilter.reset(0);
  yawRateFilter.reset(0);
  
  // Сброс PID
  pitchCascadePID.outer_integral = 0;
  pitchCascadePID.outer_prev_error = 0;
  pitchCascadePID.inner_integral = 0;
  pitchCascadePID.inner_prev_error = 0;
  yawPID.integral = 0;
  yawPID.prev_error = 0;
  pitchSimplePID.integral = 0;
  pitchSimplePID.prev_error = 0;
  
  digitalWrite(STATUS_LED, LOW);
  Serial.println("Калибровка завершена");
}

float calculateSimplePID(SimplePID* pid, float error, float dt) {
  // Пропорциональная
  float p = pid->Kp * error;
  
  // Интегральная
  pid->integral += error * dt;
  
  // Антивиндовинг
  if (abs(p) > pid->max_output * 0.8) {
    pid->integral -= error * dt;
  }
  
  pid->integral = constrain(pid->integral, -100, 100);
  float i = pid->Ki * pid->integral;
  
  // Дифференциальная
  float d = pid->Kd * (error - pid->prev_error) / dt;
  pid->prev_error = error;
  
  float output = p + i + d;
  output = constrain(output, -pid->max_output, pid->max_output);
  
  return output;
}

float calculateCascadePID(CascadePID* pid, float angle_error, float rate, float dt) {
  if (!pid->enabled) return 0;
  
  // ВНЕШНИЙ КОНТУР: угол -> целевая скорость
  float outer_p = pid->outer_Kp * angle_error;
  
  pid->outer_integral += angle_error * dt;
  pid->outer_integral = constrain(pid->outer_integral, -50, 50);
  float outer_i = pid->outer_Ki * pid->outer_integral;
  
  float outer_d = pid->outer_Kd * (angle_error - pid->outer_prev_error) / dt;
  pid->outer_prev_error = angle_error;
  
  pid->outer_output = outer_p + outer_i + outer_d;
  pid->outer_output = constrain(pid->outer_output, -pid->outer_max_output, pid->outer_max_output);
  
  // ВНУТРЕННИЙ КОНТУР: скорость -> выход
  float target_rate = pid->outer_output;
  float rate_error = target_rate - rate;
  
  float inner_p = pid->inner_Kp * rate_error;
  
  pid->inner_integral += rate_error * dt;
  
  // Антивиндовинг для внутреннего контура
  if (abs(inner_p) > pid->inner_max_output * 0.9) {
    pid->inner_integral -= rate_error * dt;
  }
  
  pid->inner_integral = constrain(pid->inner_integral, -30, 30);
  float inner_i = pid->inner_Ki * pid->inner_integral;
  
  float inner_d = pid->inner_Kd * (rate_error - pid->inner_prev_error) / dt;
  pid->inner_prev_error = rate_error;
  
  pid->inner_output = inner_p + inner_i + inner_d;
  pid->inner_output = constrain(pid->inner_output, -pid->inner_max_output, pid->inner_max_output);
  
  return pid->inner_output;
}

void readSensors() {
  static uint32_t lastReadTime = micros();
  uint32_t currentTime = micros();
  float dt = (currentTime - lastReadTime) / 1000000.0;
  if (dt > 0.1) dt = 0.02;
  
  mpu6050.update();
  
  // Сырые данные
  float gyroX = mpu6050.getGyroX();
  float gyroY = mpu6050.getGyroY();
  float gyroZ = mpu6050.getGyroZ();
  
  float accX = mpu6050.getAccX();
  float accY = mpu6050.getAccY();
  float accZ = mpu6050.getAccZ();
  
  // Фильтрация акселерометра
  static float accX_filt = 0, accY_filt = 0, accZ_filt = 0;
  accX_filt = 0.8 * accX_filt + 0.2 * accX;
  accY_filt = 0.8 * accY_filt + 0.2 * accY;
  accZ_filt = 0.8 * accZ_filt + 0.2 * accZ;
  
  // Углы из акселерометра
  float accelPitch = atan2(accY_filt, sqrt(accX_filt * accX_filt + accZ_filt * accZ_filt)) * 180.0 / PI;
  float accelRoll = atan2(-accX_filt, accZ_filt) * 180.0 / PI;
  
  // Комплементарный фильтр для углов
  static float filteredPitch = 0, filteredRoll = 0;
  float alpha = 0.96;
  
  filteredPitch = alpha * (filteredPitch + gyroY * dt) + (1 - alpha) * accelPitch;
  filteredRoll = alpha * (filteredRoll + gyroX * dt) + (1 - alpha) * accelRoll;
  
  pitchAngle = filteredPitch;
  rollAngle = filteredRoll;
  
  // Фильтрация скоростей
  pitchRate = pitchRateFilter.update(gyroY);
  rollRate = rollRateFilter.update(gyroX);
  yawRate = yawRateFilter.update(gyroZ);
  
  lastReadTime = currentTime;
}

void updateServos(float elevatorOutput, float rudderOutput) {
  // Плавное обновление позиций
  float targetElevator = ELEVATOR_NEUTRAL + elevatorOutput;
  float targetRudder = RUDDER_NEUTRAL + rudderOutput;
  
  targetElevator = constrain(targetElevator, ELEVATOR_MIN, ELEVATOR_MAX);
  targetRudder = constrain(targetRudder, RUDDER_MIN, RUDDER_MAX);
  
  currentElevatorPos = currentElevatorPos * (1 - SERVO_SMOOTH_FACTOR) + 
                      targetElevator * SERVO_SMOOTH_FACTOR;
  
  currentRudderPos = currentRudderPos * (1 - SERVO_SMOOTH_FACTOR) + 
                    targetRudder * SERVO_SMOOTH_FACTOR;
  
  // Управление сервами
  elevatorServo.write((int)currentElevatorPos);
  rudderServo.write((int)currentRudderPos);
  
  // Мотор
  int throttlePos = map((int)targetThrottle, 0, 100, 0, 180);
  throttleServo.write(throttlePos);
}

void flightControl() {
  static uint32_t lastControlTime = micros();
  uint32_t currentTime = micros();
  float dt = (currentTime - lastControlTime) / 1000000.0;
  if (dt > 0.1) dt = 0.02;
  
  // Автоматические режимы
  switch (currentMode) {
    case MODE_AUTO_TAKEOFF:
      targetThrottle = 80;
      targetPitch = 10;
      targetYawRate = 0;
      break;
      
    case MODE_AUTO_LEVEL:
      targetThrottle = 60;
      targetPitch = 0;
      targetYawRate = 0;
      break;
      
    case MODE_AUTO_TURN:
      targetThrottle = 65;
      targetPitch = 2;
      // Периодические повороты
      if ((millis() / 5000) % 2 == 0) {
        targetYawRate = 15;
      } else {
        targetYawRate = -15;
      }
      break;
      
    case MODE_EMERGENCY:
      targetThrottle = 0;
      targetPitch = 0;
      targetYawRate = 0;
      // Сброс интегралов
      pitchCascadePID.outer_integral = 0;
      pitchCascadePID.inner_integral = 0;
      yawPID.integral = 0;
      pitchSimplePID.integral = 0;
      break;
      
    case MODE_CALIBRATION:
      targetThrottle = 0;
      targetPitch = 0;
      targetYawRate = 0;
      break;
      
    // MODE_MANUAL и MODE_STABILIZE используют целевые значения от RPi
  }
  
  // Расчет ПИД
  float elevatorOutput = 0;
  float rudderOutput = 0;
  
  // Тангаж: каскадный или простой ПИД
  if (useCascadePID && currentMode != MODE_EMERGENCY) {
    float pitchError = targetPitch - pitchAngle;
    elevatorOutput = calculateCascadePID(&pitchCascadePID, pitchError, pitchRate, dt);
  } else {
    float pitchError = targetPitch - pitchAngle;
    elevatorOutput = calculateSimplePID(&pitchSimplePID, pitchError, dt);
  }
  
  // Рыскание: всегда простой ПИД
  float yawError = targetYawRate - yawRate;
  rudderOutput = calculateSimplePID(&yawPID, yawError, dt);
  
  // Компенсация крена рулем направления
  rudderOutput += rollAngle * 0.4;
  
  // Обновление сервоприводов
  updateServos(elevatorOutput, rudderOutput);
  
  lastControlTime = currentTime;
}

void debugOutput() {
  static uint32_t lastDebugTime = 0;
  if (millis() - lastDebugTime < 200) return;
  lastDebugTime = millis();
  
  Serial.print("Mode:");
  switch(currentMode) {
    case MODE_MANUAL: Serial.print("MAN"); break;
    case MODE_STABILIZE: Serial.print("STAB"); break;
    case MODE_CASCADE_STABILIZE: Serial.print("CASCADE"); break;
    case MODE_AUTO_TAKEOFF: Serial.print("TAKEOFF"); break;
    case MODE_AUTO_LEVEL: Serial.print("LEVEL"); break;
    case MODE_AUTO_TURN: Serial.print("TURN"); break;
    case MODE_CALIBRATION: Serial.print("CALIB"); break;
    case MODE_EMERGENCY: Serial.print("EMER"); break;
  }
  
  Serial.print(",P:");
  Serial.print(pitchAngle, 2);
  
  Serial.print(",PR:");
  Serial.print(pitchRate, 1);
  
  Serial.print(",Elev:");
  Serial.print(currentElevatorPos - ELEVATOR_NEUTRAL, 1);
  
  if (useCascadePID) {
    Serial.print(",CascadeOut:");
    Serial.print(pitchCascadePID.inner_output, 2);
  }
  
  Serial.println();
}

void setup() {
  // Высокоскоростной UART для RPi
  Serial.begin(UART_BAUD_RATE);
  
  // Настройка пинов
  pinMode(STATUS_LED, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  
  // Сервоприводы
  elevatorServo.attach(ELEVATOR_PIN);
  rudderServo.attach(RUDDER_PIN);
  throttleServo.attach(THROTTLE_PIN);
  
  // Нейтральное положение
  elevatorServo.write(ELEVATOR_NEUTRAL);
  rudderServo.write(RUDDER_NEUTRAL);
  throttleServo.write(0);
  
  delay(1000);
  
  // Звуковой сигнал
  for(int i = 0; i < 3; i++) {
    tone(BUZZER_PIN, 1000 + i * 500, 100);
    delay(150);
  }
  
  // Инициализация I2C
  Wire.begin();
  Wire.setClock(400000);
  
  // Инициализация MPU6050
  mpu6050.begin();
  delay(100);
  
  // Калибровка
  calibrateSensors();
  
  // Инициализация позиций
  currentElevatorPos = ELEVATOR_NEUTRAL;
  currentRudderPos = RUDDER_NEUTRAL;
  
  Serial.println("СИСТЕМА С КАСКАДНЫМ ПИД ГОТОВА");
  Serial.println("Ожидание команд от Raspberry Pi...");
}

void loop() {
  // Основной цикл 100 Гц
  static uint32_t lastLoopTime = micros();
  uint32_t currentTime = micros();
  
  if (currentTime - lastLoopTime >= 10000) {  // 100 Гц
    lastLoopTime = currentTime;
    
    // 1. Чтение датчиков
    readSensors();
    
    // 2. Обработка команд от RPi
    parseRPiCommand();
    
    // 3. Управление полетом
    flightControl();
    
    // 4. Отправка телеметрии на RPi
    sendTelemetryToRPi();
    
    // 5. Отладочный вывод (если нужно)
    debugOutput();
  }
  
  // Медленные задачи (10 Гц)
  static uint32_t lastSlowTime = 0;
  if (millis() - lastSlowTime >= 100) {
    lastSlowTime = millis();
    
    // Индикация состояния
    if (currentMode == MODE_EMERGENCY) {
      digitalWrite(STATUS_LED, (millis() % 200 < 100));
    } else if (useCascadePID) {
      digitalWrite(STATUS_LED, (millis() % 1000 < 100));
    } else {
      digitalWrite(STATUS_LED, (millis() % 1000 < 50));
    }
  }
}