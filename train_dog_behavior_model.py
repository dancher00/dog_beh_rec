import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data():
    """Загрузка и подготовка данных"""
    print("Загрузка данных...")
    df = pd.read_csv('dataset/features.csv')
    
    print(f"Загружено {len(df)} образцов")
    print("\nРаспределение классов:")
    print(df['label'].value_counts())
    
    # Разделение на признаки и метки
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Кодирование меток
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Сохранение кодировщика меток
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    
    return X, y_encoded, label_encoder

def train_models(X_train, X_test, y_train, y_test, scaler):
    """Обучение различных моделей"""
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'SVM': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=42
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- Обучение {name} ---")
        
        # Обучение
        model.fit(X_train, y_train)
        
        # Предсказания
        y_pred = model.predict(X_test)
        
        # Оценка
        accuracy = model.score(X_test, y_test)
        print(f"Точность на тестовой выборке: {accuracy:.3f}")
        
        # Кросс-валидация
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"Кросс-валидация (5-fold): {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_score': cv_scores.mean(),
            'predictions': y_pred
        }
    
    return results

def visualize_results(results, X_test, y_test, label_encoder):
    """Визуализация результатов"""
    # Создание директории для графиков
    import os
    os.makedirs('plots', exist_ok=True)
    
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]
    
    # Confusion Matrix для лучшей модели
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, best_model['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('Истинные метки')
    plt.xlabel('Предсказанные метки')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png')
    plt.close()
    
    # Сравнение моделей
    plt.figure(figsize=(10, 6))
    models_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in models_names]
    cv_scores = [results[name]['cv_score'] for name in models_names]
    
    x = np.arange(len(models_names))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Test Accuracy')
    plt.bar(x + width/2, cv_scores, width, label='CV Score')
    
    plt.xlabel('Модели')
    plt.ylabel('Точность')
    plt.title('Сравнение моделей')
    plt.xticks(x, models_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png')
    plt.close()
    
    # Feature Importance для Random Forest
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel('Важность')
        plt.title('Top 15 важных признаков (Random Forest)')
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png')
        plt.close()

def main():
    """Основная функция обучения"""
    # Создание директорий
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Загрузка данных
    X, y, label_encoder = load_and_prepare_data()
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nРазмер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки: {len(X_test)}")
    
    # Нормализация данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Сохранение scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Обучение моделей
    results = train_models(X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    
    # Выбор лучшей модели
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    
    print(f"\n\nЛучшая модель: {best_model_name}")
    print(f"Точность: {results[best_model_name]['accuracy']:.3f}")
    
    # Детальный отчет для лучшей модели
    print("\nДетальный отчет классификации:")
    print(classification_report(
        y_test, 
        results[best_model_name]['predictions'],
        target_names=label_encoder.classes_
    ))
    
    # Сохранение лучшей модели
    joblib.dump(best_model, 'models/best_model.pkl')
    with open('models/model_info.txt', 'w') as f:
        f.write(f"Model: {best_model_name}\n")
        f.write(f"Accuracy: {results[best_model_name]['accuracy']:.3f}\n")
        f.write(f"Features: {list(X.columns)}\n")
    
    # Визуализация результатов
    visualize_results(results, X_test, y_test, label_encoder)
    
    print("\n✅ Обучение завершено!")
    print(f"Модель сохранена в: models/best_model.pkl")
    print("Графики сохранены в: plots/")

if __name__ == "__main__":
    main()