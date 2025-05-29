#include <WiFi.h>
#include <Wire.h>
#include <MPU9250.h>
// WiFi настройки
const char* ssid = "Xiaomi_44EC"; // Замените на имя вашей WiFi сети
const char* password = "aidabitch"; // Замените на пароль вашей WiFi сети
// Настройки сервера
const char* host = "192.168.123.241"; // IP адрес вашего компьютера
const int port = 8080; // Порт для подключения
// IMU объект
MPU9250 IMU;
// WiFi клиент
WiFiClient client;
// Переменные для данных IMU
float ax, ay, az; // Акселерометр
float gx, gy, gz; // Гироскоп
float mx, my, mz; // Магнитометр
float temperature; // Температура
// Таймер для отправки данных
unsigned long previousMillis = 0;
const long interval = 50; // Интервал отправки данных в мс (20 Гц)
void setup() {
Serial.begin(115200);
 // Инициализация I2C
Wire.begin(25, 26); // SDA = GPIO25, SCL = GPIO26
 // Инициализация IMU
Serial.println("Инициализация IMU...");
IMU.setup(0x68); // Инициализация с адресом 0x68
 // Калибровка (опционально)
Serial.println("Калибровка акселерометра и гироскопа...");
IMU.calibrateAccelGyro();
Serial.println("Калибровка магнитометра...");
IMU.calibrateMag();
 // Подключение к WiFi
Serial.print("Подключение к WiFi: ");
Serial.println(ssid);
WiFi.begin(ssid, password);
while (WiFi.status() != WL_CONNECTED) {
delay(500);
Serial.print(".");
 }
Serial.println("");
Serial.println("WiFi подключен!");
Serial.print("IP адрес: ");
Serial.println(WiFi.localIP());
 // Подключение к серверу
connectToServer();
}
void loop() {
unsigned long currentMillis = millis();
 // Проверка подключения к серверу
if (!client.connected()) {
Serial.println("Подключение к серверу потеряно. Переподключение...");
connectToServer();
 }
 // Отправка данных с заданным интервалом
if (currentMillis - previousMillis >= interval) {
 previousMillis = currentMillis;
 // Обновление данных с IMU
IMU.update();
 // Получение данных акселерометра
 ax = IMU.getAccX();
 ay = IMU.getAccY();
 az = IMU.getAccZ();
 // Получение данных гироскопа
 gx = IMU.getGyroX();
 gy = IMU.getGyroY();
 gz = IMU.getGyroZ();
 // Получение данных магнитометра
 mx = IMU.getMagX();
 my = IMU.getMagY();
 mz = IMU.getMagZ();
 // Получение температуры
 temperature = IMU.getTemperature();
 // Формирование JSON строки с данными
 String jsonData = "{";
 jsonData += "\"timestamp\":" + String(millis()) + ",";
 jsonData += "\"accel\":{\"x\":" + String(ax, 4) + ",\"y\":" + String(ay, 4) + ",\"z\":" + String(az, 4) + "},";
 jsonData += "\"gyro\":{\"x\":" + String(gx, 4) + ",\"y\":" + String(gy, 4) + ",\"z\":" + String(gz, 4) + "},";
 jsonData += "\"mag\":{\"x\":" + String(mx, 4) + ",\"y\":" + String(my, 4) + ",\"z\":" + String(mz, 4) + "},";
 jsonData += "\"temp\":" + String(temperature, 2);
 jsonData += "}\n";
 // Отправка данных
if (client.connected()) {
client.print(jsonData);
 // Вывод в Serial для отладки
Serial.print("Отправлено: ");
Serial.print(jsonData);
 }
 }
 // Проверка входящих команд от сервера (для будущего использования)
if (client.available()) {
 String command = client.readStringUntil('\n');
Serial.print("Получена команда: ");
Serial.println(command);
 // Здесь можно обрабатывать команды для динамика
 }
}
void connectToServer() {
Serial.print("Подключение к серверу ");
Serial.print(host);
Serial.print(":");
Serial.println(port);
int attempts = 0;
while (!client.connect(host, port) && attempts < 5) {
Serial.print(".");
delay(1000);
 attempts++;
 }
if (client.connected()) {
Serial.println("\nПодключено к серверу!");
 // Отправка приветственного сообщения
client.println("{\"type\":\"connection\",\"device\":\"ESP32_IMU\"}");
 } else {
Serial.println("\nНе удалось подключиться к серверу!");
 }
}