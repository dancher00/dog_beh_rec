import pandas as pd
import numpy as np
import glob
import os
from collections import deque

def extract_features_from_window(window_data):
    """Извлечение статистических признаков из окна данных"""
    if len(window_data) < 10:  # Минимум 10 образцов
        return None
    
    # Преобразование в numpy массивы
    accel_data = np.array([[row['accel_x'], row['accel_y'], row['accel_z']] for _, row in window_data.iterrows()])
    gyro_data = np.array([[row['gyro_x'], row['gyro_y'], row['gyro_z']] for _, row in window_data.iterrows()])
    
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
    
    return features

def process_raw_data_files():
    """Обработка всех файлов с сырыми данными и извлечение признаков"""
    
    # Найти все файлы с сырыми данными
    raw_files = glob.glob('dataset/raw_data_*.csv')
    
    if not raw_files:
        print("❌ Не найдено файлов с сырыми данными в dataset/")
        return
    
    print(f"📁 Найдено {len(raw_files)} файлов с сырыми данными:")
    for file in raw_files:
        print(f"   {file}")
    
    all_features = []
    window_size = 50  # Размер окна для признаков
    
    for file_path in raw_files:
        print(f"\n🔄 Обработка {file_path}...")
        
        try:
            # Загрузка данных
            df = pd.read_csv(file_path)
            print(f"   Загружено {len(df)} строк")
            
            if len(df) < window_size:
                print(f"   ⚠️  Данных мало ({len(df)} < {window_size}), используем меньшее окно")
                window_size_current = max(10, len(df) // 2)
            else:
                window_size_current = window_size
            
            # Получение метки из имени файла или из данных
            if 'label' in df.columns:
                label = df['label'].iloc[0]
            else:
                # Извлечение метки из имени файла
                filename = os.path.basename(file_path)
                if '_rest_' in filename:
                    label = 'rest'
                elif '_walk_' in filename:
                    label = 'walk'
                elif '_run_' in filename:
                    label = 'run'
                elif '_crazy_' in filename:
                    label = 'crazy'
                else:
                    label = 'unknown'
            
            print(f"   Метка: {label}")
            
            # Скользящее окно для извлечения признаков
            features_count = 0
            step = window_size_current // 4  # Перекрытие окон на 75%
            
            for i in range(0, len(df) - window_size_current + 1, step):
                window = df.iloc[i:i + window_size_current]
                features = extract_features_from_window(window)
                
                if features:
                    features['label'] = label
                    all_features.append(features)
                    features_count += 1
            
            print(f"   ✅ Извлечено {features_count} наборов признаков")
            
        except Exception as e:
            print(f"   ❌ Ошибка обработки файла: {e}")
    
    if not all_features:
        print("\n❌ Не удалось извлечь признаки из файлов")
        return
    
    # Создание DataFrame с признаками
    features_df = pd.DataFrame(all_features)
    
    print(f"\n📊 Общая статистика:")
    print(f"   Всего наборов признаков: {len(features_df)}")
    print("\n   Распределение по классам:")
    print(features_df['label'].value_counts())
    
    # Сохранение в файл
    output_file = 'dataset/features.csv'
    features_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Признаки сохранены в: {output_file}")
    print(f"📈 Размер файла: {features_df.shape}")
    print(f"🔤 Количество признаков: {len(features_df.columns) - 1}")
    
    # Показать первые несколько строк
    print(f"\n📋 Первые 3 строки:")
    print(features_df.head(3))
    
    return features_df

if __name__ == "__main__":
    print("🚀 Извлечение признаков из сырых данных...")
    print("=" * 50)
    
    # Проверка существования директории
    if not os.path.exists('dataset'):
        print("❌ Директория dataset/ не найдена")
        exit(1)
    
    # Обработка файлов
    features_df = process_raw_data_files()
    
    if features_df is not None:
        print(f"\n🎉 Готово! Теперь можно запускать обучение модели:")
        print(f"   python3 train_dog_behavior_model.py")
    else:
        print("\n❌ Не удалось извлечь признаки. Проверьте сырые данные.")