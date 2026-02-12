// airplane_elevator_rudder_stabilizer_smooth.ino
#include <Wire.h>
#include <Servo.h>
#include <MPU6050_tockn.h>

// ===== НАСТРОЙКИ ПИНОВ =====
#define ELEVATOR_PIN 3     // Руль высоты
#define RUDDER_PIN 2       // Руль направления
#define THROTTLE_PIN 9     // ESC мотора
#define STATUS_LED 13
#define BUZZER_PIN 10

// ===== ОГРАНИЧЕНИЯ СЕРВОПРИВОДОВ =====
#define ELEVATOR_MIN 60    // Минимальный угол руля высоты
#define ELEVATOR_MAX 120   // Максимальный угол руля высоты
#define ELEVATOR_NEUTRAL 90

#define RUDDER_MIN 60      // Минимальный угол руля направления
#define RUDDER_MAX 120     // Максимальный угол руля направления
#define RUDDER_NEUTRAL 90

#define SERVO_SMOOTH_FACTOR 0.3  // Коэффициент плавности (0-1)

// ===== СЕРВОПРИВОДЫ =====
Servo elevatorServo;
Servo rudderServo;
Servo throttleServo;

// ===== ДАТЧИК =====
MPU6050 mpu6050(Wire);

// ===== ПЕРЕМЕННЫЕ =====
float pitchAngle = 0;
float rollAngle = 0;
float yawRate = 0;

// Текущие положения серв (для плавности)
float currentElevatorPos = ELEVATOR_NEUTRAL;
float currentRudderPos = RUDDER_NEUTRAL;

// ===== ПИД НАСТРОЙКИ (ОЧЕНЬ ВАЖНО!) =====
struct PID {
  float Kp, Ki, Kd;
  float integral;
  float prevError;
  float maxOutput;
  float maxIntegral;
};

// МЯГКИЕ НАСТРОЙКИ ДЛЯ НАЧАЛА
PID pitchPID = {1.5, 0.05, 0.2, 0, 0, 25, 50};    // Начинаем с маленьких значений!
PID yawPID = {1.0, 0.02, 0.1, 0, 0, 20, 30};

// ===== ЦЕЛЕВЫЕ ЗНАЧЕНИЯ =====
float targetPitch = 0;
float targetYawRate = 0;
float targetThrottle = 0;

// ===== РЕЖИМЫ =====
enum FlightState {
  STATE_PREFLIGHT,
  STATE_STABILIZATION_TEST,  // Тест стабилизации на земле
  STATE_TAKEOFF,
  STATE_CLIMB,
  STATE_LEVEL_FLIGHT,
  STATE_TURN_LEFT,
  STATE_TURN_RIGHT,
  STATE_EMERGENCY
};

FlightState currentState = STATE_PREFLIGHT;
unsigned long stateStartTime = 0;

// ===== ПАРАМЕТРЫ =====
const float TAKEOFF_PITCH = 10.0;     // Меньше угол для начала
const float CLIMB_PITCH = 6.0;        // Меньше угол
const float TURN_YAW_RATE = 15.0;     // Меньше скорость поворота
const float MAX_PITCH_ANGLE = 20.0;
const float MAX_ROLL_ANGLE = 25.0;

// ===== ТАЙМЕРЫ =====
const unsigned long STAB_TEST_TIME = 5000;   // 5 сек теста стабилизации
const unsigned long TAKEOFF_TIME = 3000;
const unsigned long CLIMB_TIME = 4000;
const unsigned long LEVEL_TIME = 8000;
const unsigned long TURN_TIME = 4000;

// ===== ФУНКЦИИ =====

void initSensors() {
  Wire.begin();
  mpu6050.begin();
  delay(100);
  
  Serial.println("КАЛИБРОВКА... НЕ ДВИГАТЬ 3 СЕК!");
  
  // Мигание во время калибровки
  for(int i = 0; i < 6; i++) {
    digitalWrite(STATUS_LED, !digitalRead(STATUS_LED));
    delay(500);
  }
  
  mpu6050.calcGyroOffsets();
  delay(500);
  
  Serial.println("КАЛИБРОВКА ЗАВЕРШЕНА");
  beep(2000, 300);
}

void readSensors() {
  static unsigned long lastTime = 0;
  float dt = (millis() - lastTime) / 1000.0;
  if (dt <= 0 || dt > 0.1) dt = 0.02;
  
  mpu6050.update();
  
  // Берем данные
  float gyroY = mpu6050.getGyroY();
  float gyroZ = mpu6050.getGyroZ();
  float gyroX = mpu6050.getGyroX();
  
  float accX = mpu6050.getAccX();
  float accY = mpu6050.getAccY();
  float accZ = mpu6050.getAccZ();
  
  // Фильтрация акселерометра (убрать шум)
  static float accX_filt = 0, accY_filt = 0, accZ_filt = 0;
  float filterAlpha = 0.5;
  accX_filt = filterAlpha * accX_filt + (1 - filterAlpha) * accX;
  accY_filt = filterAlpha * accY_filt + (1 - filterAlpha) * accY;
  accZ_filt = filterAlpha * accZ_filt + (1 - filterAlpha) * accZ;
  
  // Углы из акселерометра
  float accelPitch = atan2(accY_filt, sqrt(accX_filt * accX_filt + accZ_filt * accZ_filt)) * 180.0 / PI;
  float accelRoll = atan2(-accX_filt, accZ_filt) * 180.0 / PI;
  
  // Комплементарный фильтр
  static float filteredPitch = 0, filteredRoll = 0;
  float alpha = 0.96;
  
  filteredPitch = alpha * (filteredPitch + gyroY * dt) + (1 - alpha) * accelPitch;
  filteredRoll = alpha * (filteredRoll + gyroX * dt) + (1 - alpha) * accelRoll;
  
  pitchAngle = filteredPitch;
  rollAngle = filteredRoll;
  
  // Фильтр для рыскания
  static float filteredYawRate = 0;
  filteredYawRate = 0.8 * filteredYawRate + 0.2 * gyroZ;
  yawRate = filteredYawRate;
  
  lastTime = millis();
}

float calculatePID(PID* pid, float error, float dt) {
  // Пропорциональная
  float p = pid->Kp * error;
  
  // Интегральная с ограничением
  pid->integral += error * dt;
  
  // Антивиндовинг - если выход близок к максимуму, не накапливаем интеграл
  if (abs(p) > pid->maxOutput * 0.8) {
    pid->integral -= error * dt;
  }
  
  // Ограничение интеграла
  pid->integral = constrain(pid->integral, -pid->maxIntegral, pid->maxIntegral);
  float i = pid->Ki * pid->integral;
  
  // Дифференциальная с фильтром
  static float prevDerivative = 0;
  float derivative = (error - pid->prevError) / dt;
  
  // Фильтр производной (убирает шум)
  float dAlpha = 0.7;
  derivative = dAlpha * prevDerivative + (1 - dAlpha) * derivative;
  prevDerivative = derivative;
  
  float d = pid->Kd * derivative;
  pid->prevError = error;
  
  float output = p + i + d;
  
  // Плавное ограничение выхода
  output = constrain(output, -pid->maxOutput, pid->maxOutput);
  
  return output;
}

void updateServosSmoothly(float pitchOutput, float yawOutput) {
  // Вычисляем целевые положения
  float targetElevator = ELEVATOR_NEUTRAL + pitchOutput;
  float targetRudder = RUDDER_NEUTRAL + yawOutput;
  
  // Добавляем компенсацию крена рулем направления
  targetRudder += rollAngle * 0.3;  // Коэффициент компенсации
  
  // Ограничиваем углы
  targetElevator = constrain(targetElevator, ELEVATOR_MIN, ELEVATOR_MAX);
  targetRudder = constrain(targetRudder, RUDDER_MIN, RUDDER_MAX);
  
  // Плавное движение сервоприводов
  currentElevatorPos = currentElevatorPos * (1 - SERVO_SMOOTH_FACTOR) + 
                      targetElevator * SERVO_SMOOTH_FACTOR;
  
  currentRudderPos = currentRudderPos * (1 - SERVO_SMOOTH_FACTOR) + 
                    targetRudder * SERVO_SMOOTH_FACTOR;
  
  // Управляем сервоприводами
  elevatorServo.write((int)currentElevatorPos);
  rudderServo.write((int)currentRudderPos);
  
  // Мотор
  int throttlePos = map((int)targetThrottle, 0, 100, 0, 180);
  throttleServo.write(throttlePos);
  
  // Небольшая задержка для сервоприводов
  delayMicroseconds(100);
}

bool safetyCheck() {
  if (abs(pitchAngle) > MAX_PITCH_ANGLE) {
    Serial.print("ОПАСНОСТЬ ТАНГАЖ: ");
    Serial.println(pitchAngle);
    return false;
  }
  
  if (abs(rollAngle) > MAX_ROLL_ANGLE) {
    Serial.print("ОПАСНОСТЬ КРЕН: ");
    Serial.println(rollAngle);
    return false;
  }
  
  return true;
}

void beep(int freq, int duration) {
  tone(BUZZER_PIN, freq, duration);
  delay(duration);
  noTone(BUZZER_PIN);
}

void preflightCheck() {
  Serial.println("=== ПРЕДПОЛЕТНАЯ ПРОВЕРКА ===");
  
  // Медленный тест сервоприводов
  Serial.println("Плавный тест сервоприводов...");
  
  // Плавное движение руля высоты
  for (int pos = 90; pos <= 120; pos += 2) {
    elevatorServo.write(pos);
    delay(30);
  }
  beep(1000, 100);
  
  for (int pos = 120; pos >= 60; pos -= 2) {
    elevatorServo.write(pos);
    delay(30);
  }
  beep(1200, 100);
  
  for (int pos = 60; pos <= 90; pos += 2) {
    elevatorServo.write(pos);
    delay(30);
  }
  
  // Плавное движение руля направления
  for (int pos = 90; pos <= 120; pos += 2) {
    rudderServo.write(pos);
    delay(30);
  }
  beep(1400, 100);
  
  for (int pos = 120; pos >= 60; pos -= 2) {
    rudderServo.write(pos);
    delay(30);
  }
  beep(1600, 100);
  
  for (int pos = 60; pos <= 90; pos += 2) {
    rudderServo.write(pos);
    delay(30);
  }
  
  // Короткий тест мотора
  Serial.println("Короткий тест мотора...");
  throttleServo.write(40);
  beep(2000, 200);
  delay(300);
  throttleServo.write(0);
  
  Serial.println("Проверка завершена!");
  delay(1000);
  
  // Обратный отсчет
  for (int i = 3; i > 0; i--) {
    Serial.print(i); Serial.println("...");
    beep(2000, 100);
    delay(900);
  }
  
  beep(3000, 500);
  Serial.println("ТЕСТ СТАБИЛИЗАЦИИ...");
}

void flightControl() {
  float dt = 0.02;
  
  switch (currentState) {
    case STATE_PREFLIGHT:
      targetThrottle = 0;
      targetPitch = 0;
      targetYawRate = 0;
      
      if (millis() - stateStartTime > 3000) {
        currentState = STATE_STABILIZATION_TEST;
        stateStartTime = millis();
        Serial.println("РЕЖИМ: ТЕСТ СТАБИЛИЗАЦИИ");
        Serial.println("Наклоняйте самолет - сервоприводы должны пытаться вернуть его");
      }
      break;
      
    case STATE_STABILIZATION_TEST:
      // Тест стабилизации на земле
      targetThrottle = 0;           // Мотор выключен
      targetPitch = 0;              // Цель - горизонтальное положение
      targetYawRate = 0;            // Без поворотов
      
      // Мигание LED
      digitalWrite(STATUS_LED, (millis() % 500 < 250));
      
      if (millis() - stateStartTime > STAB_TEST_TIME) {
        currentState = STATE_TAKEOFF;
        stateStartTime = millis();
        Serial.println("РЕЖИМ: ВЗЛЕТ");
      }
      break;
      
    case STATE_TAKEOFF:
      targetThrottle = 80;           // Средний газ
      targetPitch = TAKEOFF_PITCH;
      targetYawRate = 0;
      
      digitalWrite(STATUS_LED, (millis() % 300 < 150));
      
      if (millis() - stateStartTime > TAKEOFF_TIME) {
        currentState = STATE_CLIMB;
        stateStartTime = millis();
        Serial.println("РЕЖИМ: НАБОР ВЫСОТЫ");
      }
      break;
      
    case STATE_CLIMB:
      targetThrottle = 70;
      targetPitch = CLIMB_PITCH;
      targetYawRate = 0;
      
      digitalWrite(STATUS_LED, (millis() % 600 < 150));
      
      if (millis() - stateStartTime > CLIMB_TIME) {
        currentState = STATE_LEVEL_FLIGHT;
        stateStartTime = millis();
        Serial.println("РЕЖИМ: ГОРИЗОНТАЛЬНЫЙ ПОЛЕТ");
      }
      break;
      
    case STATE_LEVEL_FLIGHT:
      targetThrottle = 60;
      targetPitch = 0;
      targetYawRate = 0;
      
      digitalWrite(STATUS_LED, (millis() % 1000 < 200));
      
      if (millis() - stateStartTime > LEVEL_TIME) {
        if (random(0, 2) == 0) {
          currentState = STATE_TURN_LEFT;
          Serial.println("РЕЖИМ: ПОВОРОТ НАЛЕВО");
        } else {
          currentState = STATE_TURN_RIGHT;
          Serial.println("РЕЖИМ: ПОВОРОТ НАПРАВО");
        }
        stateStartTime = millis();
      }
      break;
      
    case STATE_TURN_LEFT:
      targetThrottle = 65;
      targetPitch = 1;              // Очень маленький подъем
      targetYawRate = -TURN_YAW_RATE;
      
      digitalWrite(STATUS_LED, (millis() % 400 < 200));
      
      if (millis() - stateStartTime > TURN_TIME) {
        currentState = STATE_LEVEL_FLIGHT;
        stateStartTime = millis();
        Serial.println("РЕЖИМ: ГОРИЗОНТАЛЬНЫЙ ПОЛЕТ");
      }
      break;
      
    case STATE_TURN_RIGHT:
      targetThrottle = 65;
      targetPitch = 1;
      targetYawRate = TURN_YAW_RATE;
      
      digitalWrite(STATUS_LED, (millis() % 400 < 200));
      
      if (millis() - stateStartTime > TURN_TIME) {
        currentState = STATE_LEVEL_FLIGHT;
        stateStartTime = millis();
        Serial.println("РЕЖИМ: ГОРИЗОНТАЛЬНЫЙ ПОЛЕТ");
      }
      break;
      
    case STATE_EMERGENCY:
      targetThrottle = 0;
      targetPitch = 0;
      targetYawRate = 0;
      
      // Сброс интегралов
      pitchPID.integral = 0;
      yawPID.integral = 0;
      
      digitalWrite(STATUS_LED, (millis() % 200 < 100));
      
      // Прерывистый звук
      if (millis() % 1000 < 500) {
        tone(BUZZER_PIN, 800);
      } else {
        noTone(BUZZER_PIN);
      }
      break;
  }
  
  // Проверка безопасности
  if (!safetyCheck() && currentState != STATE_EMERGENCY) {
    currentState = STATE_EMERGENCY;
    Serial.println("АВАРИЯ! Активирован аварийный режим!");
  }
  
  // ПИД расчет
  float pitchError = targetPitch - pitchAngle;
  float pitchOutput = calculatePID(&pitchPID, pitchError, dt);
  
  float yawError = targetYawRate - yawRate;
  float yawOutput = calculatePID(&yawPID, yawError, dt);
  
  // Обновление сервоприводов с плавностью
  updateServosSmoothly(pitchOutput, yawOutput);
}

void sendTelemetry() {
  static unsigned long lastTelemetry = 0;
  if (millis() - lastTelemetry < 100) return;  // 10 Гц
  
  Serial.print("СОСТ:");
  switch(currentState) {
    case STATE_PREFLIGHT: Serial.print("PRE"); break;
    case STATE_STABILIZATION_TEST: Serial.print("TEST"); break;
    case STATE_TAKEOFF: Serial.print("TOFF"); break;
    case STATE_CLIMB: Serial.print("CLIMB"); break;
    case STATE_LEVEL_FLIGHT: Serial.print("LEVEL"); break;
    case STATE_TURN_LEFT: Serial.print("TL"); break;
    case STATE_TURN_RIGHT: Serial.print("TR"); break;
    case STATE_EMERGENCY: Serial.print("EMER"); break;
  }
  
  Serial.print(",P:");
  Serial.print(pitchAngle, 1);
  
  Serial.print(",R:");
  Serial.print(rollAngle, 1);
  
  Serial.print(",YR:");
  Serial.print(yawRate, 1);
  
  Serial.print(",Thr:");
  Serial.print(targetThrottle);
  
  Serial.print(",Elev:");
  Serial.print((int)currentElevatorPos);
  
  Serial.print(",Rud:");
  Serial.print((int)currentRudderPos);
  
  Serial.print(",TgtP:");
  Serial.print(targetPitch, 1);
  
  Serial.println();
  
  lastTelemetry = millis();
}

// ===== SETUP =====
void setup() {
  Serial.begin(115200);
  Serial.println("СТАБИЛИЗАТОР САМОЛЕТА С ПЛАВНЫМ УПРАВЛЕНИЕМ");
  
  // Настройка пинов
  pinMode(STATUS_LED, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  
  // Настройка сервоприводов
  elevatorServo.attach(ELEVATOR_PIN);
  rudderServo.attach(RUDDER_PIN);
  throttleServo.attach(THROTTLE_PIN);
  
  // Установка в нейтральное положение
  elevatorServo.write(ELEVATOR_NEUTRAL);
  rudderServo.write(RUDDER_NEUTRAL);
  throttleServo.write(0);
  
  delay(1000);
  
  // Предполетная проверка
  preflightCheck();
  
  // Инициализация датчиков
  initSensors();
  
  // Сброс ПИД
  pitchPID.integral = 0;
  pitchPID.prevError = 0;
  yawPID.integral = 0;
  yawPID.prevError = 0;
  
  // Инициализация текущих положений
  currentElevatorPos = ELEVATOR_NEUTRAL;
  currentRudderPos = RUDDER_NEUTRAL;
  
  stateStartTime = millis();
  randomSeed(analogRead(0));
  
  Serial.println("СИСТЕМА ГОТОВА!");
  Serial.println("=====================================");
}

// ===== LOOP =====
void loop() {
  static unsigned long lastControlTime = 0;
  
  // Главный цикл 50 Гц
  if (millis() - lastControlTime >= 20) {
    lastControlTime = millis();
    
    readSensors();
    flightControl();
    sendTelemetry();
  }
  
  // Команды отладки
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 'e' || cmd == 'E') {
      currentState = STATE_EMERGENCY;
      Serial.println("АВАРИЙНЫЙ РЕЖИМ АКТИВИРОВАН");
    }
    else if (cmd == 't' || cmd == 'T') {
      currentState = STATE_STABILIZATION_TEST;
      stateStartTime = millis();
      Serial.println("ТЕСТ СТАБИЛИЗАЦИИ");
    }
    else if (cmd == 'f' || cmd == 'F') {
      currentState = STATE_LEVEL_FLIGHT;
      stateStartTime = millis();
      Serial.println("ГОРИЗОНТАЛЬНЫЙ ПОЛЕТ");
    }
    else if (cmd == 'p') {
      // Настройка PID: p,Kp,Ki,Kd
      String input = Serial.readStringUntil('\n');
      input.trim();
      
      if (input.startsWith("pitch,")) {
        sscanf(input.c_str(), "pitch,%f,%f,%f", 
               &pitchPID.Kp, &pitchPID.Ki, &pitchPID.Kd);
        Serial.print("Pitch PID: ");
        Serial.print(pitchPID.Kp); Serial.print(", ");
        Serial.print(pitchPID.Ki); Serial.print(", ");
        Serial.println(pitchPID.Kd);
      }
      else if (input.startsWith("yaw,")) {
        sscanf(input.c_str(), "yaw,%f,%f,%f", 
               &yawPID.Kp, &yawPID.Ki, &yawPID.Kd);
        Serial.print("Yaw PID: ");
        Serial.print(yawPID.Kp); Serial.print(", ");
        Serial.print(yawPID.Ki); Serial.print(", ");
        Serial.println(yawPID.Kd);
      }
    }
  }
}