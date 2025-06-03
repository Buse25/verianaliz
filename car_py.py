import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# Veri setini yükleme
# car.data dosyasının sütun isimlerini belirleme
column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
data = pd.read_csv("car.data", names=column_names, header=None)

print("Veri seti başarıyla yüklendi. İlk 5 satır:")
print(data.head())

# Veri seti hakkında genel bilgiler
print("\nVeri seti boyutu:", data.shape)
print("\nVeri seti istatistikleri:")
print(data.describe(include='all'))

# Kategorik verileri sayısal değerlere dönüştürme
# Her bir kategorik sütun için dönüşüm yapma
for column in data.columns:
    data[column] = pd.Categorical(data[column]).codes

print("\nDönüştürülmüş veri seti:")
print(data.head())

# Özellikler ve hedef değişkeni ayırma
X = data.drop('class', axis=1)
y = data['class']

# Veri setini eğitim ve test olarak bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nEğitim seti boyutu: {X_train.shape}")
print(f"Test seti boyutu: {X_test.shape}")

# Karar ağacı modelini oluşturma
dt_classifier = DecisionTreeClassifier(random_state=42)

# Modeli eğitme
dt_classifier.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
y_pred = dt_classifier.predict(X_test)

# Model performansını değerlendirme
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel doğruluğu: {accuracy:.4f}")

print("\nSınıflandırma raporu:")
print(classification_report(y_test, y_pred))

print("\nKarmaşıklık matrisi:")
print(confusion_matrix(y_test, y_pred))

# Özellik önemliliklerini görselleştirme
feature_importance = dt_classifier.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
plt.title('Özellik Önemlilikleri')
plt.xlabel('Önemlilik')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nÖzellik önemlilikleri görselleştirildi ve 'feature_importance.png' olarak kaydedildi.")

# Karar ağacını görselleştirme
plt.figure(figsize=(15, 10))
tree.plot_tree(dt_classifier, feature_names=X.columns, filled=True, rounded=True)
plt.savefig('decision_tree.png', dpi=100)
print("\nKarar ağacı görselleştirildi ve 'decision_tree.png' olarak kaydedildi.")

# Eğitim ve test setinde model performansı karşılaştırma
train_pred = dt_classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nEğitim seti doğruluğu: {train_accuracy:.4f}")
print(f"Test seti doğruluğu: {test_accuracy:.4f}")

# Farklı max_depth değerleri için model doğruluğu karşılaştırma
depths = range(1, 11)
train_scores = []
test_scores = []

for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, dt.predict(X_train)))
    test_scores.append(accuracy_score(y_test, dt.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, marker='o', linestyle='-', label='Eğitim doğruluğu')
plt.plot(depths, test_scores, marker='o', linestyle='-', label='Test doğruluğu')
plt.xlabel('Ağaç Derinliği')
plt.ylabel('Doğruluk')
plt.title('Farklı Ağaç Derinlikleri için Model Performansı')
plt.legend()
plt.grid(True)
plt.savefig('depth_performance.png')
print("\nFarklı ağaç derinlikleri için model performansı görselleştirildi ve 'depth_performance.png' olarak kaydedildi.")

# En iyi derinlik değerini bulma
best_depth = depths[np.argmax(test_scores)]
print(f"\nEn iyi ağaç derinliği: {best_depth}")

# En iyi derinlik değeri ile modeli tekrar eğitme
best_dt = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
best_dt.fit(X_train, y_train)
best_pred = best_dt.predict(X_test)
best_accuracy = accuracy_score(y_test, best_pred)

print(f"\nEn iyi modelin doğruluğu: {best_accuracy:.4f}")
print("\nEn iyi model için sınıflandırma raporu:")
print(classification_report(y_test, best_pred))

print("\nProgram tamamlandı!")
