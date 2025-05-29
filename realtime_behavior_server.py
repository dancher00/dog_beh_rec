import socket
import threading
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
import joblib
import warnings
warnings.filterwarnings('ignore')

# Настройки сервера
HOST = '0.0.0.0'
PORT = 8080

# Загрузка модели
print("🔄 Загрузка модели...")
try:
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    print("✅ Модель загружена успешно")
    print(f"   Классы: {label_encoder.classes_}")
    print(f"   Тип модели: {type(model).__name__}")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    exit(1)

# Параметры для классификации
WINDOW_SIZE = 50  # Такой же как при обучении
DATA_BUFFER = deque(maxlen=WINDOW_SIZE)

# Хранилище данных
latest_data = None
clients = []
behavior_history = deque(maxlen=20)  # История последних 20 предсказаний

# Статистика поведения
behavior_stats = {
    'rest': {'count': 0, 'total_time': 0, 'start_time': None},
    'walk': {'count': 0, 'total_time': 0, 'start_time': None},
    'run': {'count': 0, 'total_time': 0, 'start_time': None},
    'crazy': {'count': 0, 'total_time': 0, 'start_time': None}
}
current_behavior = None
last_behavior_change = datetime.now()

def extract_features_from_window(window_data):
    """Извлечение признаков из окна данных (ИДЕНТИЧНО обучению)"""
    if len(window_data) < WINDOW_SIZE:
        return None
    
    # Создаем DataFrame точно в том же формате, что при обучении
    data_for_df = []
    for d in window_data:
        row = {
            'accel_x': d['accel']['x'],
            'accel_y': d['accel']['y'], 
            'accel_z': d['accel']['z'],
            'gyro_x': d['gyro']['x'],
            'gyro_y': d['gyro']['y'],
            'gyro_z': d['gyro']['z']
        }
        data_for_df.append(row)
    
    df = pd.DataFrame(data_for_df)
    
    # Извлечение признаков (копия из extract_features.py)
    accel_data = np.array([[row['accel_x'], row['accel_y'], row['accel_z']] for _, row in df.iterrows()])
    gyro_data = np.array([[row['gyro_x'], row['gyro_y'], row['gyro_z']] for _, row in df.iterrows()])
    
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
    
    # Количество пересечений нуля
    for i, axis in enumerate(['x', 'y', 'z']):
        if len(accel_data) > 1:
            zero_crossings = np.sum(np.diff(np.sign(accel_data[:, i])) != 0)
            features[f'accel_{axis}_zero_crossings'] = zero_crossings
        else:
            features[f'accel_{axis}_zero_crossings'] = 0
    
    # Энергия сигнала
    features['accel_energy'] = np.sum(accel_magnitude**2)
    features['gyro_energy'] = np.sum(gyro_magnitude**2)
    
    # Преобразуем в массив в правильном порядке
    feature_names = [
        'accel_x_mean', 'accel_x_std', 'accel_x_max', 'accel_x_min', 'accel_x_range',
        'accel_y_mean', 'accel_y_std', 'accel_y_max', 'accel_y_min', 'accel_y_range',
        'accel_z_mean', 'accel_z_std', 'accel_z_max', 'accel_z_min', 'accel_z_range',
        'gyro_x_mean', 'gyro_x_std', 'gyro_x_max', 'gyro_x_min', 'gyro_x_range',
        'gyro_y_mean', 'gyro_y_std', 'gyro_y_max', 'gyro_y_min', 'gyro_y_range',
        'gyro_z_mean', 'gyro_z_std', 'gyro_z_max', 'gyro_z_min', 'gyro_z_range',
        'accel_magnitude_mean', 'accel_magnitude_std',
        'gyro_magnitude_mean', 'gyro_magnitude_std',
        'accel_x_zero_crossings', 'accel_y_zero_crossings', 'accel_z_zero_crossings',
        'accel_energy', 'gyro_energy'
    ]
    
    feature_vector = np.array([features[name] for name in feature_names]).reshape(1, -1)
    return feature_vector

def predict_behavior(features):
    """Предсказание поведения"""
    try:
        # Нормализация признаков
        features_scaled = scaler.transform(features)
        
        # Предсказание
        prediction = model.predict(features_scaled)[0]
        
        # Декодирование метки
        behavior = label_encoder.inverse_transform([prediction])[0]
        
        # Попытка получить вероятности (может не работать для всех моделей)
        probabilities = None
        try:
            if hasattr(model, 'predict_proba'):
                proba_array = model.predict_proba(features_scaled)[0]
                probabilities = {}
                for i, label in enumerate(label_encoder.classes_):
                    probabilities[label] = float(proba_array[i])
            elif hasattr(model, 'decision_function'):
                # Для SVM без вероятностей
                decision_scores = model.decision_function(features_scaled)[0]
                # Преобразуем в простые "уверенности"
                if len(label_encoder.classes_) == 2:
                    probabilities = {
                        label_encoder.classes_[0]: 1.0 if decision_scores < 0 else 0.0,
                        label_encoder.classes_[1]: 1.0 if decision_scores >= 0 else 0.0
                    }
                else:
                    # Для многоклассовой SVM - используем расстояния
                    probabilities = {}
                    for i, label in enumerate(label_encoder.classes_):
                        probabilities[label] = 1.0 if i == prediction else 0.0
        except:
            # Если не можем получить вероятности
            probabilities = {label: 1.0 if label == behavior else 0.0 
                           for label in label_encoder.classes_}
        
        return behavior, probabilities
        
    except Exception as e:
        print(f"❌ Ошибка предсказания: {e}")
        return None, None

def update_behavior_stats(new_behavior):
    """Обновление статистики поведения"""
    global current_behavior, last_behavior_change
    
    now = datetime.now()
    
    if current_behavior != new_behavior:
        # Завершение предыдущего поведения
        if current_behavior and behavior_stats[current_behavior]['start_time']:
            duration = (now - behavior_stats[current_behavior]['start_time']).total_seconds()
            behavior_stats[current_behavior]['total_time'] += duration
            behavior_stats[current_behavior]['start_time'] = None
        
        # Начало нового поведения
        current_behavior = new_behavior
        behavior_stats[new_behavior]['count'] += 1
        behavior_stats[new_behavior]['start_time'] = now
        last_behavior_change = now

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
                        print(f"[ОШИБКА] Ошибка парсинга JSON: {e}")
    
    except ConnectionResetError:
        print(f"[ОТКЛЮЧЕНИЕ] Клиент {address} отключился")
    except Exception as e:
        print(f"[ОШИБКА] {e}")
    
    finally:
        if client_socket in clients:
            clients.remove(client_socket)
        client_socket.close()

def process_imu_data(data, address):
    """Обработка данных IMU с классификацией"""
    global latest_data
    latest_data = data
    
    data['server_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    
    if 'type' in data and data['type'] == 'connection':
        print(f"[ИНФО] Устройство {data.get('device', 'Unknown')} подключено")
        return
    
    # Добавление в буфер
    DATA_BUFFER.append(data)
    
    # Показываем прогресс заполнения буфера
    if len(DATA_BUFFER) % 10 == 0:
        print(f"\r[БУФЕР] Заполнено: {len(DATA_BUFFER)}/{WINDOW_SIZE}", end='')
    
    # Классификация если буфер полный
    if len(DATA_BUFFER) == WINDOW_SIZE:
        features = extract_features_from_window(list(DATA_BUFFER))
        
        if features is not None:
            behavior, probabilities = predict_behavior(features)
            
            if behavior and probabilities:
                # Обновление статистики
                update_behavior_stats(behavior)
                behavior_history.append(behavior)
                
                # Эмодзи для поведений
                behavior_emoji = {
                    'rest': '😴',
                    'walk': '🚶',
                    'run': '🏃',
                    'crazy': '🌀'
                }
                
                # Вывод результата
                print(f"\n" + "="*50)
                print(f"[КЛАССИФИКАЦИЯ] {datetime.now().strftime('%H:%M:%S')}")
                print(f"  {behavior_emoji.get(behavior, '🐕')} Поведение: {behavior.upper()}")
                
                if probabilities and any(p > 0 for p in probabilities.values()):
                    print(f"  📊 Вероятности:")
                    for b, p in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                        if p > 0:
                            bar = '█' * int(p * 20)
                            emoji = behavior_emoji.get(b, '🐕')
                            print(f"    {emoji} {b:8s}: {bar:20s} {p:.1%}")
                
                # Отправка результата клиентам
                result = {
                    'type': 'behavior',
                    'behavior': behavior,
                    'probabilities': probabilities,
                    'timestamp': data['server_time']
                }
                send_to_all_clients(result)
            else:
                print(f"\n❌ Ошибка предсказания")

def send_to_all_clients(data):
    """Отправка данных всем подключенным клиентам"""
    message = json.dumps(data) + '\n'
    for client in clients[:]:  # Копия списка
        try:
            client.send(message.encode('utf-8'))
        except:
            if client in clients:
                clients.remove(client)

def console_input():
    """Обработка консольных команд"""
    print("\n" + "="*50)
    print("🐕 КЛАССИФИКАЦИЯ ПОВЕДЕНИЯ СОБАКИ В РЕАЛЬНОМ ВРЕМЕНИ")
    print("="*50)
    print("Команды:")
    print("  status   - текущий статус")
    print("  stats    - статистика поведения")
    print("  history  - последние 20 предсказаний")
    print("  reset    - сбросить статистику")
    print("  data     - показать последние данные IMU")
    print("  exit     - выйти")
    print("="*50)
    
    while True:
        try:
            cmd = input("\n🐕 Команда: ").strip().lower()
            
            if cmd == 'status':
                print(f"\n📊 [СТАТУС СИСТЕМЫ]")
                print(f"   Текущее поведение: {current_behavior or 'Определяется...'}")
                print(f"   Буфер данных: {len(DATA_BUFFER)}/{WINDOW_SIZE}")
                print(f"   Подключено клиентов: {len(clients)}")
                
                if current_behavior:
                    duration = (datetime.now() - last_behavior_change).total_seconds()
                    print(f"   Длительность: {duration:.1f} сек")
            
            elif cmd == 'stats':
                print(f"\n📈 [СТАТИСТИКА ПОВЕДЕНИЯ]")
                total_time = sum(s['total_time'] for s in behavior_stats.values())
                
                behavior_emoji = {'rest': '😴', 'walk': '🚶', 'run': '🏃', 'crazy': '🌀'}
                
                for behavior, stats in behavior_stats.items():
                    count = stats['count']
                    time_spent = stats['total_time']
                    
                    # Добавить текущее время если это активное поведение
                    if behavior == current_behavior and stats['start_time']:
                        time_spent += (datetime.now() - stats['start_time']).total_seconds()
                    
                    percent = (time_spent / total_time * 100) if total_time > 0 else 0
                    emoji = behavior_emoji.get(behavior, '🐕')
                    
                    print(f"\n   {emoji} {behavior.upper()}:")
                    print(f"      Количество раз: {count}")
                    print(f"      Общее время: {time_spent:.1f} сек ({percent:.1f}%)")
            
            elif cmd == 'history':
                print(f"\n📜 [ИСТОРИЯ] Последние {len(behavior_history)} предсказаний:")
                if behavior_history:
                    print("   " + " → ".join(behavior_history))
                else:
                    print("   Пока нет данных")
            
            elif cmd == 'data':
                if latest_data:
                    print(f"\n📡 [ПОСЛЕДНИЕ ДАННЫЕ IMU]")
                    if 'accel' in latest_data:
                        acc = latest_data['accel']
                        print(f"   Акселерометр: X={acc['x']:7.3f} Y={acc['y']:7.3f} Z={acc['z']:7.3f}")
                    if 'gyro' in latest_data:
                        gyr = latest_data['gyro']
                        print(f"   Гироскоп:     X={gyr['x']:7.3f} Y={gyr['y']:7.3f} Z={gyr['z']:7.3f}")
                else:
                    print("❌ Данные еще не получены")
            
            elif cmd == 'reset':
                for stats in behavior_stats.values():
                    stats['count'] = 0
                    stats['total_time'] = 0
                    stats['start_time'] = None
                behavior_history.clear()
                print("🔄 Статистика сброшена")
            
            elif cmd == 'exit':
                print("👋 Выход из программы...")
                break
            
            else:
                print("❌ Неизвестная команда")
                
        except KeyboardInterrupt:
            print("\n👋 Программа прервана")
            break

def main():
    """Главная функция"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(5)
        print(f"🚀 [СЕРВЕР] Запущен на {HOST}:{PORT}")
        print(f"⏳ [СЕРВЕР] Ожидание подключений...")
        print(f"🤖 [МОДЕЛЬ] Готова к классификации!")
        
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

if __name__ == "__main__":
    main()