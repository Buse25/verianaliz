# Araç Değerlendirme Karar Ağacı Modeli

Bu proje, **UCI Car Evaluation** veri seti üzerinde karar ağacı algoritması kullanılarak araçların kabul edilebilirlik durumlarını sınıflandırmayı amaçlamaktadır. Model, `scikit-learn` kütüphanesi ile geliştirilmiştir.

## 📊 Kullanılan Veri Seti

- Kaynak: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/car+evaluation)
- Toplam 1728 örnek
- Özellikler:
  - buying: Satın alma maliyeti
  - maint: Bakım maliyeti
  - doors: Kapı sayısı
  - persons: Yolcu kapasitesi
  - lug_boot: Bagaj büyüklüğü
  - safety: Güvenlik seviyesi
- Hedef (class): unacc, acc, good, v-good

## 🧠 Kullanılan Yöntem

- **Algoritma:** Decision Tree Classifier (CART)
- **Kodlama Dili:** Python
- **Kütüphaneler:** pandas, scikit-learn, matplotlib, seaborn

## 📈 Model Performansı

- Doğruluk (Accuracy): **%97,29**
- Optimize Edilmiş Doğruluk (depth=10): **%98,26**
- Sınıflandırma raporu, özellikle "unacc" ve "acc" sınıflarında yüksek başarı göstermiştir.

## 🗂️ Dosya Yapısı

