import socket
import threading
import json
import time
from datetime import datetime

# Настройки сервера
HOST = '0.0.0.0'  # Слушать на всех интерфейсах
PORT = 8080       # Порт должен совпадать с портом в скетче ESP32

# Хранилище последних данных
latest_data = None
clients = []

def handle_client(client_socket, address):
    """Обработка подключенного клиента"""
    print(f"[ПОДКЛЮЧЕНИЕ] Новое подключение от {address}")
    clients.append(client_socket)
    
    buffer = ""
    try:
        while True:
            # Получение данных
            data = client_socket.recv(1024).decode('utf-8')
            if not data:
                break
            
            buffer += data
            
            # Обработка полных JSON сообщений (разделенных \n)
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                if line.strip():
                    try:
                        json_data = json.loads(line)
                        process_imu_data(json_data, address)
                    except json.JSONDecodeError as e:
                        print(f"[ОШИБКА] Ошибка парсинга JSON: {e}")
                        print(f"[ДАННЫЕ] {line}")
    
    except ConnectionResetError:
        print(f"[ОТКЛЮЧЕНИЕ] Клиент {address} отключился")
    except Exception as e:
        print(f"[ОШИБКА] {e}")
    
    finally:
        clients.remove(client_socket)
        client_socket.close()

def process_imu_data(data, address):
    """Обработка данных IMU"""
    global latest_data
    latest_data = data
    
    # Добавление метки времени сервера
    data['server_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    
    # Вывод данных
    if 'type' in data and data['type'] == 'connection':
        print(f"[ИНФО] Устройство {data.get('device', 'Unknown')} подключено")
    else:
        # Форматированный вывод данных IMU
        print(f"\n[ДАННЫЕ IMU от {address[0]}] Время: {data.get('server_time')}")
        
        if 'accel' in data:
            acc = data['accel']
            print(f"  Акселерометр: X={acc['x']:7.3f} Y={acc['y']:7.3f} Z={acc['z']:7.3f} м/с²")
        
        if 'gyro' in data:
            gyr = data['gyro']
            print(f"  Гироскоп:     X={gyr['x']:7.3f} Y={gyr['y']:7.3f} Z={gyr['z']:7.3f} рад/с")
        
        # if 'mag' in data:
        #     mag = data['mag']
        #     print(f"  Магнитометр:  X={mag['x']:7.3f} Y={mag['y']:7.3f} Z={mag['z']:7.3f} мкТл")
        
        # if 'temp' in data:
        #     print(f"  Температура:  {data['temp']:.2f} °C")
    
    # Здесь можно добавить логику для анализа данных и отправки команд обратно

def send_command_to_esp(command):
    """Отправка команды всем подключенным ESP32"""
    message = json.dumps(command) + '\n'
    for client in clients:
        try:
            client.send(message.encode('utf-8'))
            print(f"[КОМАНДА] Отправлена команда: {command}")
        except:
            print("[ОШИБКА] Не удалось отправить команду")

def console_input():
    """Обработка ввода команд с консоли"""
    while True:
        try:
            cmd = input("\nВведите команду (или 'help' для справки): ")
            if cmd.lower() == 'help':
                print("Доступные команды:")
                print("  status - показать последние данные")
                print("  beep - отправить команду звукового сигнала (для будущего)")
                print("  exit - выйти из программы")
            elif cmd.lower() == 'status':
                if latest_data:
                    print(f"Последние данные: {json.dumps(latest_data, indent=2)}")
                else:
                    print("Данные еще не получены")
            elif cmd.lower() == 'beep':
                send_command_to_esp({"command": "beep", "duration": 100})
            elif cmd.lower() == 'exit':
                break
        except KeyboardInterrupt:
            break

def main():
    """Главная функция сервера"""
    # Создание сокета
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        # Привязка и прослушивание
        server_socket.bind((HOST, PORT))
        server_socket.listen(5)
        print(f"[СЕРВЕР] Сервер запущен на {HOST}:{PORT}")
        print(f"[СЕРВЕР] Ожидание подключений...")
        
        # Запуск потока для консольных команд
        console_thread = threading.Thread(target=console_input)
        console_thread.daemon = True
        console_thread.start()
        
        # Принятие подключений
        while True:
            client_socket, address = server_socket.accept()
            client_thread = threading.Thread(
                target=handle_client, 
                args=(client_socket, address)
            )
            client_thread.daemon = True
            client_thread.start()
    
    except KeyboardInterrupt:
        print("\n[СЕРВЕР] Остановка сервера...")
    finally:
        server_socket.close()

if __name__ == "__main__":
    main()