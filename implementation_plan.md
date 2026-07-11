# Implementation Plan — EEGFlow

### EEG Sinyal İşleme ve Makine Öğrenmesi Sınıflandırma Kont Paneli

> **Proje Adı:** EEGFlow
> **Geliştirici:** İsmet Can Sezgin
> **Danışman / Hoca:** Berkay Çaltı & Ümit Reva
> **Süre:** 20 iş günü (4 hafta)
> **Platform:** Web Tabanlı Kontrol Paneli (Web SPA)
> **Arka Plan:** Python (FastAPI) + Sinyal İşleme / ML Kütüphaneleri
> **Ön Yüz:** Vanilla HTML5, CSS3 (Glassmorphism & Dark Theme), JS (ES6) + Chart.js / Canvas
> **Doküman Sürümü:** v2.0 (Danışman geri bildirimleri doğrultusunda fazlara bölünerek güncellendi)
> **Doküman Türü:** Implementation Plan / Tek Doğruluk Kaynağı (Single Source of Truth)

---

## İçindekiler

1. Yönetici Özeti
2. Problem Tanımı ve Hedefler
3. Proje Geliştirme Fazları (Danışman Tavsiyeli)
4. Kapsam ve Sınırlar
5. Teknoloji Yığını
6. Proje Klasör Ağacı ve Mimari Yapı
7. Sinyal İşleme ve Model Eğitimi Akış Şeması
8. Sistem Mimarisi & Backend Modülleri
9. Sinyal Filtreleme ve Özellik Çıkarım Matrisi
10. Zaman Dilimleme (Epoching) ve Olay Etiketleri
11. Görsel Tasarım ve Arayüz Teması
12. UI (Kullanıcı Arayüzü) Ekran Envanteri
13. Dağıtım ve Yerel Çalıştırma Süreci
14. Kodlama Standartları ve Optimizasyon
15. Test Stratejisi
16. Dokümantasyon Yapısı (docs/)
17. 20 İş Günlük Yol Haritası (Gün Gün Detaylı Plan)
18. Haftalık Sprint Özeti & Kilometre Taşları (Milestones)
19. Git Workflow & PR Süreci
20. GitHub Projects Board & Issue Yönetimi
21. Daily Standup Standardı
22. Risk Yönetimi
23. Definition of Done (Bitti Kriteri)
24. Teslim Edilecekler & Onay Durumu

---

## 1. Yönetici Özeti

EEGFlow; çok kanallı EEG (Elektroensefalografi) verilerini işleyen, görselleştiren, gürültü filtreleme uygulayan ve makine öğrenmesi modelleriyle (SVM, Random Forest, XGBoost) sınıflandıran kapsamlı bir web uygulamasıdır. Proje, özellikle klinik ve akademik EEG analiz süreçlerini basitleştirmeyi ve makine öğrenmesi modellerinin doğruluğunu test etmeyi amaçlar.

Kullanıcı, sisteme bir CSV dosyası yükleyerek kanalları seçer, ham sinyal üzerinde band-pass ve notch filtrelerini çalıştırır, sinyali epoch'lara böler, zaman/frekans alanlarında özellik çıkarımı yapar ve Group K-Fold doğrulama kullanarak modelleri eğitip karşılaştırır. Çıktımız, yüksek performansla çalışan, etkileşimli grafiklerle desteklenmiş, modern bir karanlık mod web arayüzüdür.

---

## 2. Problem Tanımı ve Hedefler

### 2.1 Çözülen Problem

EEG verileri yüksek oranda gürültü (kas hareketleri, göz kırpma, şebeke gürültüsü) içerir. Ayrıca, makine öğrenmesi modellerinin eğitiminde farklı deneklerin (subjects) verilerinin karıştırılması "data leakage" (veri sızıntısı) problemine yol açar ve test doğruluğunu yapay olarak yüksek gösterir. 

EEGFlow;
* Sinyali filtreleyerek gürültüleri temizler.
* Sinyal özelliklerini görselleştirerek anlaşılır kılar.
* **Group K-Fold** çapraz doğrulama ile modelleri denek bazlı ayırarak gerçekçi ve güvenilir sınıflandırma sonuçları sunar.

### 2.2 Başarı Kriterleri (Definition of Success)

* [ ] Yüklenen CSV dosyalarının doğrulanması ve otomatik frekans ($f_s$) tespiti.
* [ ] Çok kanallı ham ve filtrelenmiş sinyallerin tarayıcıda pürüzsüz grafiklerle gösterilmesi.
* [ ] Butterworth band-pass ve notch filtrelerinin SciPy tabanlı olarak kusursuz çalışması.
* [ ] PSD (Power Spectral Density) grafiklerinin interaktif olarak görselleştirilmesi.
* [ ] SVM, Random Forest ve XGBoost modellerinin eğitilmesi ve karşılaştırılması.
* [ ] Group K-Fold çapraz doğrulaması ile sızıntısız model değerlendirmesi.

---

## 3. Proje Geliştirme Fazları (Danışman Tavsiyeli)

Danışman hocamızın geri bildirimleri doğrultusunda, projenin stabilitesini sağlamak ve iş yükünü optimize etmek amacıyla geliştirme süreci 3 ana faza bölünmüştür:

### 3.1 Faz 1 - Pipeline Tasarımı (Gün 1-10)
* **Odak Noktası:** Ham sinyalin sisteme yüklenmesi, gürültülerin giderilmesi (band-pass, notch filtreleme ve detrending) ve sinyalin web arayüzünde görselleştirilmesi.
* **Kritik Kural:** Sinyalin temizliğinden emin olmadan ve Faz 1 başarıyla tamamlanmadan Faz 2'ye geçilmeyecektir.
* **Milestone 1 Toplantısı:** Faz 1 bittiğinde hocamızla bir araya gelinip "Ham Veri -> Temiz Sinyal -> Görselleştirme" akışının üzerinden geçilecektir.

### 3.2 Faz 2 - Öznitelik Mühendisliği (Gün 11-15)
* **Odak Noktası:** Temizlenmiş sinyalin zaman pencerelerine (epoch) bölünmesi ve güç spektral yoğunluğunun (PSD) hesaplanması.
* **Doğrulama:** Çıkarılan verilerin doğruluğu, örneğin Alpha dalgalarının göz kapama anındaki değişimleri gibi fizyolojik olarak bilinen durumlar üzerinden grafiklerle doğrulanacaktır (Milestone 2).

### 3.3 Faz 3 - Model ve Doğrulama (Gün 16-20)
* **Odak Noktası:** SVM, Random Forest ve XGBoost modellerinin bu özellikler üzerinde eğitilmesi.
* **Doğrulama Metodu:** Modellerin kişiye özel ezberlemesini değil, genel bir desen öğrenmesini sağlamak için kesinlikle **Group K-Fold** yöntemi kullanılacaktır (Milestone 3).

---

## 4. Kapsam ve Sınırlar

### 4.1 Kapsam İçi (20 günde teslim edilecek)
* FastAPI tabanlı veri yükleme, işleme ve model eğitimi API'si.
* Ham ve filtrelenmiş sinyal çizimini destekleyen Canvas/Chart.js arayüzü.
* Butterworth band-pass, notch filtreleri ve detrending modülleri.
* Zaman (Mean, Std, Skew, Kurtosis, Hjorth) ve Frekans (PSD, Bant Güçleri, Spektrogram) özellikleri.
* SVM, RF ve XGBoost modellerini içeren eğitim hattı.
* Group K-Fold çapraz doğrulama ve karmaşıklık matrisi (Confusion Matrix) hesaplamaları.
* Model performanslarını karşılaştıran interaktif dashboard.

### 4.2 Kapsam Dışı (Gelecek faz veya projeden çıkarılanlar)
* Donanımdan (EEG cihazından) gerçek zamanlı veri akışı (LSL/Bluetooth).
* Derin öğrenme (CNN, RNN, Transformer) modelleri (Klasik ML ve özellik mühendisliğine odaklanılacak).
* Çok kullanıcılı oturum yönetimi (Auth) ve bulut veritabanı entegrasyonu.

---

## 5. Teknoloji Yığını

| Katman | Teknoloji | Gerekçe |
| --- | --- | --- |
| **Backend API** | Python (FastAPI + Uvicorn) | Hızlı, otomatik dokümantasyon (Swagger) sunan, Python ekosistemiyle uyumlu framework. |
| **Sinyal İşleme** | SciPy, NumPy, Pandas | Bilimsel hesaplama, matris işlemleri ve dijital filtre tasarımı için endüstri standardı. |
| **Makine Öğrenmesi** | Scikit-Learn, XGBoost | Sınıflandırma algoritmaları, Group K-Fold bölücüler ve metrik hesaplama araçları. |
| **Ön Yüz (UI)** | HTML5, CSS3, ES6 JavaScript | Framework bağımlılığı olmadan temiz, performanslı ve esnek tek sayfa uygulama (SPA). |
| **Grafik Kütüphanesi** | Chart.js / Canvas API | Tarayıcı tarafında yüksek frekanslı zaman serisi verilerini pürüzsüz çizme yeteneği. |
| **Versiyon Kontrol** | Git + GitHub | Main korumalı branch mimarisi ve pull request onay süreçleri. |
| **Proje Yönetimi** | GitHub Projects (Kanban) | İşlerin To Do, In Progress, Review ve Done süreçlerinde takibi. |

---

## 6. Proje Klasör Ağacı ve Mimari Yapı

```
eeg_internship_project/
├── backend/
│   ├── main.py                 # FastAPI Sunucu Giriş Noktası
│   ├── requirements.txt         # Python Kütüphane Bağımlılıkları
│   └── utils/
│       ├── data_loader.py       # CSV yükleme, sütun doğrulama
│       ├── filters.py           # Butterworth, Notch, Detrending filtreleri
│       ├── epoching.py          # Sinyal bölütleme ve zaman pencereleri
│       ├── features.py          # Zaman/Frekans özellikleri, PSD, Spektrogram hesaplama
│       └── models.py            # SVM, RF, XGBoost, GroupKFold ve metrik hesaplama
├── frontend/
│   ├── index.html              # Dashboard ana yapısı (SPA)
│   ├── styles.css              # Glassmorphic koyu tema CSS kuralları
│   └── app.js                  # API entegrasyonu ve grafik çizimleri
├── data/
│   ├── generate_sample_data.py  # Testler için sentetik EEG üretici script
│   └── sample_eeg.csv          # Üretilen sentetik veri
├── tests/
│   ├── test_filters.py         # Filtre doğrulama testleri
│   └── test_features.py        # Matematiksel özellik çıkarım testleri
├── docs/                       # Analiz notları ve ekran görüntüleri
├── .gitignore                  # Python ve IDE dosyaları için gitignore
└── implementation_plan.md      # BU PLAN DOSYASI
```

---

## 7. Sinyal İşleme ve Model Eğitimi Akış Şeması

```
[CSV Yükle] ➔ [Kanal & fs Seç] ➔ [Detrending] ➔ [Notch (50Hz)] ➔ [Band-pass (0.5-40Hz)]
                                                                         │
                                                                         ▼
[Metrikler & Matrix] ↵ [Group K-Fold] ↵ [Modeller (SVM/RF/XGB)] ↵ [Özellik Tablosu] ↵ [Epoching]
```

---

## 8. Sistem Mimarisi & Backend Modülleri

* `main.py`: İstekleri karşılar, asenkron endpoints sunar ve verileri JSON formatında ön yüze döner.
* `data_loader.py`: Yüklenen CSV dosyasının biçimini, kanal sayısını, eksik verileri kontrol eder.
* `filters.py`: Ham veriyi SciPy'ın `filtfilt` fonksiyonu ile sıfır-faz kaymalı olarak filtreler.
* `epoching.py`: Kesintisiz sinyali belirli event kodlarına veya sabit saniyelik pencerelere böler.
* `features.py`: Her kanal ve epoch için özellikleri çıkartıp NumPy dizilerine dönüştürür.
* `models.py`: Eğitim özelliklerini standartlaştırır, modelleri eğitir ve Group K-Fold ile doğrular.

---

## 9. Sinyal Filtreleme ve Özellik Çıkarım Matrisi

| Adım | İşlem Tipi | Parametreler | Çıktı / Gerekçe |
| --- | --- | --- | --- |
| **Detrending** | Doğrusal Arındırma | Linear | Sinyaldeki yavaş kaymaları ve DC kaymasını sıfırlar. |
| **Notch Filter** | Dar Bant Durduran | $50\text{ Hz}$ / $60\text{ Hz}$ (Q=30) | Şebeke hattından sızan gürültüyü yok eder. |
| **Band-pass** | Butterworth (2. Derece) | $0.5 - 45\text{ Hz}$ | EEG sinyalleri dışındaki yüksek/alçak frekansları eler. |
| **Time Features** | İstatiksel & Hjorth | Mean, Std, Skew, Kurtosis, Hjorth | Sinyalin genlik ve karmaşıklık profilini çıkarır. |
| **Frequency Features** | PSD & Welch | Delta, Theta, Alpha, Beta, Gamma | Frekans bantlarındaki güç yoğunluğunu ölçer. |

---

## 10. Zaman Dilimleme (Epoching) ve Olay Etiketleri

* **Sliding Window:** Eğer sinyalde belirli olay etiketleri yoksa, kullanıcı 1, 2 veya 4 saniyelik pencereler belirleyebilir (örn: %50 çakışmalı).
* **Event-based:** Sinyalde bulunan `event` sütununa göre (örn: 0 = Relax, 1 = Task) tetikleyici anından sonraki belirli pencereler epoch olarak kesilir.
* **Denek Bilgisi:** Her veri satırı veya dosyası bir `subject_id` içerir. Model eğitiminde bu ID'ler sızıntıyı engellemek için gruplama anahtarı olarak kullanılır.

---

## 11. Görsel Tasarım ve Arayüz Teması

* **Tema:** Derin mavi ve koyu gri tonlarında modern **Glassmorphism Dark Theme**.
* **Renk Paleti:**
  * Arka Plan: `#0B0F19` (Koyu Gece Mavisi)
  * Kartlar/Paneller: `rgba(23, 31, 50, 0.6)` (Cam Efekti, Backdrop Blur)
  * Sinyal Kanalları: Kanal bazlı dinamik renkler (Neon Yeşil, Turkuaz, Mor, Turuncu vb.)
  * Accent Renk: `#3B82F6` (Neon Mavi)
* **Okunabilirlik:** Yazı tipi olarak Inter veya Outfit kullanılacak, sinyal grafikleri yüksek kontrastlı çizilecektir.

---

## 12. UI (Kullanıcı Arayüzü) Ekran Envanteri

Uygulama tek sayfa (SPA) yapısında olup sol tarafta bir kontrol paneli, sağ tarafta ise seçilen sekmeye göre değişen bir dashboard alanı barındıracaktır:

1. **Dashboard Sekmesi (Veri Yükleme):** CSV sürükle-bırak alanı, dosya analizi, kanal listesi ve örnekleme frekansı tespiti.
2. **Sinyal Analiz Sekmesi:** Ham sinyal ve filtre sonrası sinyali eş zamanlı gösteren çok kanallı etkileşimli zaman serisi grafiği.
3. **Özellik Haritası Sekmesi:** Her kanalın PSD bant güçlerini gösteren bar grafikleri ve spektrogram görselleri.
4. **Model Eğitim Sekmesi:** Model hiperparametre seçimleri, "Eğit" butonu, canlı konsol logları.
5. **Sonuç & Karşılaştırma Sekmesi:** Confusion Matrix görseli, ROC-AUC grafiği, model metrik tabloları ve özellik önem sırası (Feature Importance) grafiği.

---

## 13. Dağıtım ve Yerel Çalıştırma Süreci

* **Gereksinimler:** Python 3.10 veya üzeri, Git, Modern bir Web Tarayıcı.
* **Kurulum:**
  1. `backend` klasöründe sanal ortam oluşturma: `python -m venv venv`
  2. Paketlerin yüklenmesi: `pip install -r backend/requirements.txt`
* **Çalıştırma:**
  1. API Sunucusu: `uvicorn backend.main:app --reload` (Port: 8000)
  2. Ön Yüz: Tarayıcı üzerinden doğrudan `frontend/index.html` dosyasının açılması veya basit bir HTTP sunucusu yardımıyla yayına alınması.

---

## 14. Kodlama Standartları ve Optimizasyon

* **Değişkenler ve Sınıflar:** PEP 8 standartlarına uygun Python kodlaması. JavaScript tarafında ES6 standartları (const/let, arrow functions, async/await).
* **Bellek Yönetimi:** EEG verileri RAM'de yüksek yer kaplayabileceğinden, ham veri üzerinde gereksiz kopya oluşturmaktan kaçınılacaktır. Pandas DataFrame işlemleri yerinde (`inplace=True`) veya NumPy matrisleri üzerinden yapılacaktır.
* **Grafik Çizim Optimizasyonu:** Grafik çizilirken binlerce veri noktasının tarayıcıyı yormaması için veri seyreltme (downsampling / LTTB algoritması) uygulanacaktır.

---

## 15. Test Stratejisi

* **Birim Testleri (Unit Tests):** `pytest` kullanılarak filtrelerin kazanç kayıpları ve özellik çıkarım formüllerinin doğruluğu test edilecektir.
  * Komut: `pytest tests/`
* **Entegrasyon Testi:** Sentetik EEG üreten `generate_sample_data.py` scripti ile elde edilen dosya sisteme yüklenip baştan sona veri işleme hattı manuel olarak kontrol edilecektir.

---

## 16. Dokümantasyon Yapısı (docs/)

```
docs/
├── signals/
│   └── processing_flow.md      # Sinyal filtrelerinin matematiksel arka planı
├── models/
│   └── group_kfold_leakage.md  # Group K-Fold mantığı ve neden kullanıldığı
└── screenshots/
    └── dashboard_preview.png   # Uygulamadan ekran görüntüleri
```

---

## 17. 20 İş Günlük Yol Haritası (Gün Gün Detaylı Plan)

Staj günlüğü kurallarına tam uyum sağlamak için her gün sonunda hedeflenen işleri içeren anlamlı commit'ler atılacaktır.

### 🟦 FAZ 1 — Pipeline Tasarımı (Gün 1-10)

* **Gün 1:** GitHub repository'sinin oluşturulması, Projects Kanban panosu kurulumu ve `implementation_plan.md` commit'i.
* **Gün 2 (Bugün):** Proje dizin yapısının oluşturulması, `.gitignore` ve `backend/requirements.txt` bağımlılık dosyalarının hazırlanarak commit edilmesi.
* **Gün 3:** `data_loader.py` yazımı; CSV yükleme, kolon yapıları, eksik veri ve tip kontrolleri.
* **Gün 4:** `generate_sample_data.py` sentetik EEG üretici scriptinin yazılması ve test verisinin hazırlanması.
* **Gün 5:** `main.py` dosyasında veri yükleme (upload) endpoint'inin ve ön yüz veri yükleme panelinin kodlanması.
* **Gün 6:** `filters.py` altında Butterworth Band-pass filtre fonksiyonunun yazılması.
* **Gün 7:** `filters.py` modülüne Notch filtre (50Hz/60Hz) ve lineer detrending özelliklerinin eklenmesi.
* **Gün 8:** Sinyal filtreleme backend endpoints'lerinin tamamlanması ve birim testlerinin hazırlanması.
* **Gün 9:** Ön yüzde ham ve filtrelenmiş sinyalleri kanallara göre listeleyen interaktif grafik ekranının kodlanması.
* **Gün 10:** Grafik optimizasyonu (LTTB veri seyreltme) ve Faz 1 sonu **Milestone 1 Değerlendirmesi** (Hoca onay toplantısı).

### 🟨 FAZ 2 — Öznitelik Mühendisliği (Gün 11-15)

* **Gün 11:** `epoching.py` modülünün kodlanması; sinyalin zaman pencerelerine veya event kodlarına göre dilimlenmesi.
* **Gün 12:** `features.py` yazımı; her epoch ve kanal için zaman alanı özelliklerinin (Mean, Std, Skewness, Hjorth) çıkarılması.
* **Gün 13:** `features.py` modülüne Welch yöntemi ile PSD (Power Spectral Density) ve frekans band güçlerinin eklenmesi.
* **Gün 14:** API tarafında spektrogram hesaplama işlevinin ve ön yüzde spektrogram görselleştirme panelinin yapılması.
* **Gün 15:** Alpha dalgaları doğrulama grafiklerinin arayüze eklenmesi (göz kapalı/açık anlarındaki değişim) ve **Milestone 2 Değerlendirmesi**.

### 🟧 FAZ 3 — Model ve Doğrulama (Gün 16-20)

* **Gün 16:** `models.py` modülünün yazılması. SVM, Random Forest ve XGBoost sınıflandırıcılarının veri hazırlama hatlarına bağlanması.
* **Gün 17:** **Group K-Fold** çapraz doğrulama mekanizmasının entegrasyonu. Modellerin denek bazlı test edilerek sızıntısız doğruluğunun ölçülmesi.
* **Gün 18:** Karmaşıklık Matrisi (Confusion Matrix), Precision, Recall, F1-Score ve ROC-AUC metriklerinin hesaplanması ve API'ye bağlanması.
* **Gün 19:** Ön yüzde model eğitim kontrol paneli ve karşılaştırma grafiklerinin (Model Comparison Dashboard) entegre edilmesi.
* **Gün 20:** PyTest testlerinin koşturulması, kod temizliği (refactor), `README.md` dosyasının ekran görüntüleriyle süslenmesi ve final staj sunumu/milestone 3 kapanışı.

---

## 18. Haftalık Sprint Özeti & Kilometre Taşları (Milestones)

| Sprint | Günler | Temel Odak Noktası (Hocanın Faz Yapısı) | Çıktı / Kilometre Taşı (Milestone) |
| --- | --- | --- | --- |
| **Sprint 1** | 1–10 | **Faz 1: Pipeline Tasarımı** (Veri Yükleme, Filtreleme ve Grafik Görselleştirme) | **Milestone 1:** Ham Veri ➔ Temiz Sinyal ➔ Görselleştirme hattının çalışır hali (Hoca Toplantısı). |
| **Sprint 2** | 11–15 | **Faz 2: Öznitelik Mühendisliği** (Zaman Pencereleri, PSD ve Fizyolojik Doğrulama) | **Milestone 2:** PSD bant güçleri tablosu ve Alpha dalgası göz kapama doğrulama grafikleri. |
| **Sprint 3** | 16–20 | **Faz 3: Model ve Doğrulama** (SVM/RF/XGB, Group K-Fold ve Metrik Karşılaştırma) | **Milestone 3:** Sızıntısız Group K-Fold ile test edilmiş modeller, karşılaştırma ekranı ve staj teslimi. |

---

## 19. Git Workflow & PR Süreci

* **Ana Dal Koruması:** `master` dalına doğrudan onaylanmamış kod birleştirilmeyecektir. Tüm geliştirmeler özellik dallarında (feature branches) yapılacaktır.
* **Dallanma Kuralı:** Yeni özellikler için `feature/data-loader`, `feature/butterworth-filter` gibi açıklayıcı dal isimleri kullanılacaktır.
* **Commit Mesaj Standartları:**
  * `feat(filter): add butterworth bandpass filter utility`
  * `fix(loader): resolve nan handling in csv reader`
  * `test(math): add unit tests for PSD band power calculation`

---

## 20. GitHub Projects Board & Issue Yönetimi

* Her iş gününün konusu bir **Issue** olarak açılacak ve projenin Kanban panosunda önceliklendirilecektir.
* Kanban panosundaki kart durumları: **To Do**, **In Progress**, **Review**, **Done** şeklinde güncel tutulacaktır.

---

## 21. Daily Standup Standardı

Her iş gününün başında stajyer günlüğü ile entegre şekilde şu üç soru yanıtlanacaktır:
1. **Dün ne yaptım?** (Örn: Proje iskeletini ve ortam bağımlılıklarını kurdum.)
2. **Bugün ne yapacağım?** (Örn: CSV dosya okuma ve veri doğrulama modülünü yazacağım.)
3. **Karşılaştığım bir engel var mı?** (Örn: Yok, plan dahilinde ilerliyorum.)

---

## 22. Risk Yönetimi

| Risk Tanımı | Olasılık | Etki | Çözüm / Önlem Planı |
| --- | --- | --- | --- |
| **Büyük Veri Kümesinde Tarayıcı Kasılması** | Yüksek | Orta | Tarayıcı tarafında veri seyreltme (downsampling) uygulanarak grafik noktası sınırlandırılacak. |
| **Model Eğitiminde Data Leakage (Sızıntı)** | Orta | Yüksek | Group K-Fold yapısı kullanılacak; denek verileri eğitim ve test setlerinde kesinlikle ayrılacak. |
| **Sinyal Gürültüsünün Temizlenememesi** | Düşük | Orta | Detrending ve Notch filtrelerinin doğru sıralamayla uygulandığından emin olunacak. |

---

## 23. Definition of Done (Bitti Kriteri)

Bir iş gününün/görevin bitti kabul edilmesi için gereken şartlar:
* [ ] Yazılan Python kodlarında hiçbir derleme veya sözdizimi hatası olmaması.
* [ ] Sinyal hesaplama modüllerinin matematiksel olarak doğruluğunun test edilmesi.
* [ ] İlgili günün kodlarının belirlenen isimde branch'e commmitlenip test edilmesi.
* [ ] Kanban kartının **Done** durumuna getirilmesi.

---

## 24. Teslim Edilecekler & Onay Durumu

1. **Uygulama Kaynak Kodları:** FastAPI backend ve Vanilla JS frontend kodları.
2. **Sentetik Veri Oluşturucu:** Test için kullanılabilecek `generate_sample_data.py`.
3. **Birim Testleri:** Sinyal filtreleme doğruluğunu ölçen test dosyaları.
4. **README & Dokümanlar:** Kurulum adımları, kullanım rehberi ve örnek ekran görüntülerinin bulunduğu detaylı ana sayfa.

---

### Staj Departmanı Onay Durumu

* **Plan Değerlendirme Durumu:** 🟢 Plan Onaylandı (Geliştirmeye Hazır)
* **Onaylayan Akademisyen / Danışman:** Berkay Çaltı & Ümit Reva
* **Onay Tarihi:** 14 Temmuz 2026
