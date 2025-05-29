import socket
import threading
import json
import time
import csv
import os
from datetime import datetime
from collections import deque
import numpy as np
import sys

# Настройки сервера
HOST = '0.0.0.0'
PORT = 8080

# Параметры записи данных
WINDOW_SIZE = 50  # Размер окна для признаков (50 * 50ms = 2.5 секунды)
RECORDING = False
CURRENT_LABEL = None
DATA_BUFFER = deque(maxlen=WINDOW_SIZE)

# Настройки вывода
SHOW_DATA = False  # Показывать ли данные IMU в реальном времени
SHOW_STATS = True  # Показывать ли статистику

# Метки поведения
BEHAVIORS = {
    '1': 'rest',      # покой
    '2': 'walk',      # ходьба
    '3': 'run',       # бег
    '4': 'crazy'      # сумасшествие (крутится)
}

# Создание директории для данных
os.makedirs('dataset', exist_ok=True)

# Файл для записи сырых данных
raw_data_file = None
raw_data_writer = None

# Файл для записи признаков
features_file = open('dataset/features.csv', 'w', newline='')
features_writer = None
features_header_written = False

# Хранилище последних данных и статистики
latest_data = None
clients = []
data_count = 0
last_status_time = time.time()

def extract_features(window_data):
    """Извлечение признаков из окна данных"""
    if len(window_data) < WINDOW_SIZE:
        return None
    
    # Преобразование в numpy массивы
    accel_data = np.array([[d['accel']['x'], d['accel']['y'], d['accel']['z']] for d in window_data])
    gyro_data = np.array([[d['gyro']['x'], d['gyro']['y'], d['gyro']['z']] for d in window_data])
    
    features = {}
    
    # Статистические признаки для акселерометра
    for i, axis in enumerate(['x', 'y', 'z']):
        features[f'accel_{axis}_mean'] = np.mean(accel_data[:, i])
        features[f'accel_{axis}_std'] = np.std(accel_data[:, i])
        features[f'accel_{axis}_max'] = np.max(accel_data[:, i])
        features[f'accel_{axis}_min'] = np.min(accel_data[:, i])
        features[f'accel_{axis}_range'] = features[f'accel_{axis}_max'] - features[f'accel_{axis}_min']
    
    # Статистические признаки для гироскопа
    for i, axis in enumerate(['x', 'y', 'z']):
        features[f'gyro_{axis}_mean'] = np.mean(gyro_data[:, i])
        features[f'gyro_{axis}_std'] = np.std(gyro_data[:, i])
        features[f'gyro_{axis}_max'] = np.max(gyro_data[:, i])
        features[f'gyro_{axis}_min'] = np.min(gyro_data[:, i])
        features[f'gyro_{axis}_range'] = features[f'gyro_{axis}_max'] - features[f'gyro_{axis}_min']
    
    # Магнитуда ускорения
    accel_magnitude = np.sqrt(np.sum(accel_data**2, axis=1))
    features['accel_magnitude_mean'] = np.mean(accel_magnitude)
    features['accel_magnitude_std'] = np.std(accel_magnitude)
    
    # Магнитуда угловой скорости
    gyro_magnitude = np.sqrt(np.sum(gyro_data**2, axis=1))
    features['gyro_magnitude_mean'] = np.mean(gyro_magnitude)
    features['gyro_magnitude_std'] = np.std(gyro_magnitude)
    
    # Частотные признаки (простой анализ)
    # Количество пересечений нуля
    for i, axis in enumerate(['x', 'y', 'z']):
        zero_crossings = np.sum(np.diff(np.sign(accel_data[:, i])) != 0)
        features[f'accel_{axis}_zero_crossings'] = zero_crossings
    
    # Энергия сигнала
    features['accel_energy'] = np.sum(accel_magnitude**2)
    features['gyro_energy'] = np.sum(gyro_magnitude**2)
    
    return features

def handle_client(client_socket, address):
    """Обработка подключенного клиента"""
    print(f"[ПОДКЛЮЧЕНИЕ] Новое подключение от {address}")
    clients.append(client_socket)
    
    buffer = ""
    try:
        while True:
            data = client_socket.recv(1024).decode('utf-8')
            if not data:
                break
            
            buffer += data
            
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                if line.strip():
                    try:
                        json_data = json.loads(line)
                        process_imu_data(json_data, address)
                    except json.JSONDecodeError as e:
                        if SHOW_DATA:
                            print(f"[ОШИБКА] Ошибка парсинга JSON: {e}")
    
    except ConnectionResetError:
        print(f"[ОТКЛЮЧЕНИЕ] Клиент {address} отключился")
    except Exception as e:
        if SHOW_DATA:
            print(f"[ОШИБКА] {e}")
    
    finally:
        if client_socket in clients:
            clients.remove(client_socket)
        client_socket.close()

def process_imu_data(data, address):
    """Обработка данных IMU"""
    global latest_data, features_writer, features_header_written, data_count, last_status_time
    
    latest_data = data
    data_count += 1
    
    data['server_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    
    if 'type' in data and data['type'] == 'connection':
        print(f"[ИНФО] Устройство {data.get('device', 'Unknown')} подключено")
        return
    
    # Показывать данные только если это включено
    if SHOW_DATA:
        print(f"\n[ДАННЫЕ IMU от {address[0]}] Время: {data.get('server_time')}")
        if 'accel' in data:
            acc = data['accel']
            print(f"  Акселерометр: X={acc['x']:7.3f} Y={acc['y']:7.3f} Z={acc['z']:7.3f} м/с²")
        if 'gyro' in data:
            gyr = data['gyro']
            print(f"  Гироскоп:     X={gyr['x']:7.3f} Y={gyr['y']:7.3f} Z={gyr['z']:7.3f} рад/с")
    
    # Показывать статистику каждые 5 секунд
    current_time = time.time()
    if SHOW_STATS and current_time - last_status_time > 5:
        print(f"\r[СТАТУС] Получено пакетов: {data_count} | Подключений: {len(clients)} | Запись: {'ВКЛ' if RECORDING else 'ВЫКЛ'} {f'({CURRENT_LABEL})' if CURRENT_LABEL else ''}", end='')
        last_status_time = current_time
    
    # Запись данных если включена
    if RECORDING and CURRENT_LABEL and raw_data_writer:
        # Запись сырых данных
        raw_data_writer.writerow([
            data['server_time'],
            data['timestamp'],
            CURRENT_LABEL,
            data['accel']['x'], data['accel']['y'], data['accel']['z'],
            data['gyro']['x'], data['gyro']['y'], data['gyro']['z']
        ])
        
        # Добавление в буфер для извлечения признаков
        DATA_BUFFER.append(data)
        
        # Извлечение признаков если буфер полный
        if len(DATA_BUFFER) == WINDOW_SIZE:
            features = extract_features(list(DATA_BUFFER))
            if features:
                # Запись заголовка если еще не записан
                if not features_header_written:
                    headers = list(features.keys()) + ['label']
                    features_writer = csv.DictWriter(features_file, fieldnames=headers)
                    features_writer.writeheader()
                    features_header_written = True
                
                # Добавление метки и запись
                features['label'] = CURRENT_LABEL
                features_writer.writerow(features)
                features_file.flush()
                
                print(f"\n[ЗАПИСЬ] Записаны признаки для поведения: {CURRENT_LABEL}")

def show_help():
    """Показать справку по командам"""
    print("\n=== КОМАНДЫ УПРАВЛЕНИЯ ===")
    print("ЗАПИСЬ ДАННЫХ:")
    print("  1 - начать запись ПОКОЙ (сидит/лежит/стоит)")
    print("  2 - начать запись ХОДЬБА")
    print("  3 - начать запись БЕГ") 
    print("  4 - начать запись СУМАСШЕСТВИЕ (крутится)")
    print("  s - остановить запись")
    print("")
    print("ПРОСМОТР:")
    print("  show - показать/скрыть данные IMU в реальном времени")
    print("  stats - показать/скрыть статистику")
    print("  status - показать текущий статус")
    print("  data - показать последние данные")
    print("")
    print("УПРАВЛЕНИЕ:")
    print("  help - показать эту справку")
    print("  clear - очистить экран")
    print("  exit - выйти")

def console_input():
    """Обработка ввода команд с консоли"""
    global RECORDING, CURRENT_LABEL, raw_data_file, raw_data_writer, DATA_BUFFER, SHOW_DATA, SHOW_STATS
    
    print("\n=== СБОР ДАННЫХ ДЛЯ ОБУЧЕНИЯ ===")
    print("Введите 'help' для списка команд")
    
    while True:
        try:
            print("\n" + "="*50)
            cmd = input("Команда: ").strip().lower()
            
            if cmd in ['1', '2', '3', '4']:
                # Закрыть предыдущий файл если был
                if raw_data_file:
                    raw_data_file.close()
                
                # Новая запись
                CURRENT_LABEL = BEHAVIORS[cmd]
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'dataset/raw_data_{CURRENT_LABEL}_{timestamp}.csv'
                
                raw_data_file = open(filename, 'w', newline='')
                raw_data_writer = csv.writer(raw_data_file)
                raw_data_writer.writerow([
                    'server_time', 'timestamp', 'label',
                    'accel_x', 'accel_y', 'accel_z',
                    'gyro_x', 'gyro_y', 'gyro_z'
                ])
                
                DATA_BUFFER.clear()
                RECORDING = True
                
                print(f"\n✅ [ЗАПИСЬ НАЧАТА]")
                print(f"   Поведение: {CURRENT_LABEL.upper()}")
                print(f"   Файл: {filename}")
                print(f"   ВАЖНО: Убедитесь, что собака демонстрирует поведение '{CURRENT_LABEL}'!")
                print(f"   Введите 's' для остановки записи")
                
            elif cmd == 's':
                if RECORDING:
                    RECORDING = False
                    CURRENT_LABEL = None
                    if raw_data_file:
                        raw_data_file.close()
                        raw_data_file = None
                    print("\n⏹️  [ЗАПИСЬ ОСТАНОВЛЕНА]")
                else:
                    print("❌ Запись не активна")
                    
            elif cmd == 'status':
                print(f"\n📊 [ТЕКУЩИЙ СТАТУС]")
                print(f"   Запись: {'🔴 АКТИВНА' if RECORDING else '⚪ НЕ АКТИВНА'}")
                if RECORDING:
                    print(f"   Поведение: {CURRENT_LABEL}")
                    print(f"   Буфер: {len(DATA_BUFFER)}/{WINDOW_SIZE}")
                
                print(f"   Подключений: {len(clients)}")
                print(f"   Получено пакетов: {data_count}")
                
                # Подсчет записанных данных
                try:
                    with open('dataset/features.csv', 'r') as f:
                        feature_count = sum(1 for line in f) - 1
                    print(f"   Записано образцов: {feature_count}")
                except FileNotFoundError:
                    print(f"   Записано образцов: 0")
            
            elif cmd == 'show':
                SHOW_DATA = not SHOW_DATA
                print(f"📺 Показ данных IMU: {'ВКЛ' if SHOW_DATA else 'ВЫКЛ'}")
                
            elif cmd == 'stats':
                SHOW_STATS = not SHOW_STATS
                print(f"📈 Показ статистики: {'ВКЛ' if SHOW_STATS else 'ВЫКЛ'}")
                
            elif cmd == 'data':
                if latest_data:
                    print(f"\n📡 [ПОСЛЕДНИЕ ДАННЫЕ]")
                    if 'accel' in latest_data:
                        acc = latest_data['accel']
                        print(f"   Акселерометр: X={acc['x']:7.3f} Y={acc['y']:7.3f} Z={acc['z']:7.3f} м/с²")
                    if 'gyro' in latest_data:
                        gyr = latest_data['gyro']
                        print(f"   Гироскоп:     X={gyr['x']:7.3f} Y={gyr['y']:7.3f} Z={gyr['z']:7.3f} рад/с")
                else:
                    print("❌ Данные еще не получены")
                    
            elif cmd == 'help':
                show_help()
                
            elif cmd == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                print("=== СБОР ДАННЫХ ДЛЯ ОБУЧЕНИЯ ===")
                
            elif cmd == 'exit':
                if RECORDING:
                    RECORDING = False
                    if raw_data_file:
                        raw_data_file.close()
                print("\n👋 Выход из программы...")
                break
                
            else:
                print("❌ Неизвестная команда. Введите 'help' для справки")
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Программа прервана пользователем")
            break
        except EOFError:
            print("\n\n👋 Выход из программы...")
            break

def main():
    """Главная функция сервера"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(5)
        print(f"🚀 [СЕРВЕР] Запущен на {HOST}:{PORT}")
        print(f"⏳ [СЕРВЕР] Ожидание подключений...")
        
        console_thread = threading.Thread(target=console_input)
        console_thread.daemon = True
        console_thread.start()
        
        while True:
            client_socket, address = server_socket.accept()
            client_thread = threading.Thread(
                target=handle_client, 
                args=(client_socket, address)
            )
            client_thread.daemon = True
            client_thread.start()
    
    except KeyboardInterrupt:
        print("\n⏹️  [СЕРВЕР] Остановка сервера...")
    finally:
        server_socket.close()
        if features_file:
            features_file.close()

if __name__ == "__main__":
    main()