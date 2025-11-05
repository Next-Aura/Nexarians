# Kursus Mini tentang AI dan Pembelajaran Mesin

## Pelajaran 5: Mengenal Berbagai Model Pembelajaran Mesin

Halo, selamat datang di Pelajaran 5! Kamu sudah berhasil bikin dan menilai model pertama pakai regresi linear, keren! Tapi, regresi linear cuma salah satu alat di kotak peralatan kita. Di pelajaran ini, kita bakal jalan-jalan lihat model lain yang populer, ngerti kapan pakainya, dan bandingin performanya. Kita tetap pakai contoh harga rumah biar gampang dipahami. Yuk, kita mulai petualangan nambah skill modeling!

---

### Kenapa Harus Coba Model Lain?

Bayangin kamu cuma punya satu pisau di dapur. Bisa sih masak, tapi motong sayur, ngupas buah, atau ngocok telur bakal lebih gampang kalau pakai alat yang pas. Nah, sama halnya dengan pembelajaran mesin. Regresi linear cocok buat hubungan data yang lurus, tapi dunia nyata seringkali lebih rumit—ada kurva, kategori, atau pola yang nggak lurus. Pilih model yang tepat, prediksi kamu bakal lebih jitu dan AI kamu makin canggih. Kita cek beberapa model keren dan kapan pakainya!

---

### Mengenal Beberapa Model Pembelajaran Mesin

Berikut beberapa model populer selain regresi linear, dijelasin pake bahasa sederhana plus analogi biar gampang kebayang.

1. **Decision Tree**  
   - **Apa itu?** Model yang bikin keputusan dengan nanya ya/tidak, kayak flowchart.  
   - **Analogi:** Mirip main tebak-tebakan—mulai dari pertanyaan besar (misalnya, "Rumahnya lebih dari 120 m²?") lalu nyempitin ke prediksi.  
   - **Kapan dipakai?** Buat klasifikasi (misalnya, ya/tidak) atau regresi dengan data yang nggak lurus. Cocok kalau mau hasil yang gampang dimengerti.  
   - **Kelebihan:** Gampang dipahami dan bisa divisualisasikan.  
   - **Kekurangan:** Bisa overfit kalau pohonnya terlalu "rimbun".  

2. **Random Forest**  
   - **Apa itu?** Sekelompok pohon keputusan yang kerja bareng, kayak tim yang voting buat keputusan terbaik.  
   - **Analogi:** Seperti nanya ke beberapa temen soal harga rumah, terus ambil rata-rata jawabannya biar lebih akurat.  
   - **Kapan dipakai?** Buat klasifikasi atau regresi, apalagi kalau datanya rumit dan mau hasil yang lebih stabil.  
   - **Kelebihan:** Jago nangani data kompleks dan nggak gampang salah.  
   - **Kekurangan:** Agak lambat latihnya dan susah dimengerti dibanding pohon tunggal.  

3. **Support Vector Machine (SVM)**  
   - **Apa itu?** Model yang nyari "garis" terbaik buat misahin data.  
   - **Analogi:** Bayangin gambar jalan lebar yang misahin dua kelompok rumah (misalnya, mahal vs murah) tanpa nyelonong ke sisi lain.  
   - **Kapan dipakai?** Buat klasifikasi dengan batas jelas atau regresi. Cocok buat data kecil.  
   - **Kelebihan:** Jago di data dengan banyak dimensi.  
   - **Kekurangan:** Bisa lambat di data besar dan butuh nyetel parameter dengan hati-hati.  

4. **K-Nearest Neighbor (KNN)**  
   - **Apa itu?** Model "malas" yang prediksi berdasarkan tetangga terdekat dari data.  
   - **Analogi:** Seperti nanya ke tetangga sebelah berapa harga rumah mereka buat nebak harga rumahmu.  
   - **Kapan dipakai?** Buat klasifikasi atau regresi sederhana, terutama di data kecil.  
   - **Kelebihan:** Nggak perlu latihan, cuma nyimpen data.  
   - **Kekurangan:** Lambat buat prediksi di data besar dan gampang bingung kalau ada fitur yang nggak relevan.  

---

### Praktik Langsung: Nyobain Model Berbeda untuk Harga Rumah

Kita bakal bandingin regresi linear, pohon keputusan, dan hutan acak pake data harga rumah. Kita pakai scikit-learn biar gampang.

#### 1. Siapin Library dan Data

Kita tambah fitur biar perbandingannya lebih seru.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Data rumah yang lebih lengkap
data = {
    'Ukuran': [1000, 1500, 1200, 1800, 900, 1400, 1600, 1100],
    'Kamar': [2, 3, 3, 4, 2, 3, 4, 2],
    'Lokasi': [1, 0, 1, 0, 1, 0, 0, 1],  # 1=Kota, 0=Pinggiran
    'Harga': [200000, 250000, 220000, 300000, 180000, 240000, 280000, 210000]
}
df = pd.DataFrame(data)

X = df[['Ukuran', 'Kamar', 'Lokasi']]
y = df['Harga']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

#### 2. Latih dan Cek LinearRegression (Sekadar Review)

```python
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

print(f"Regresi Linear - MAE: ${lr_mae:,.0f}, R²: {lr_r2:.2f}")
```

#### 3. Latih dan Cek Pohon Keputusan (Decision Tree)

```python
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

dt_mae = mean_absolute_error(y_test, dt_pred)
dt_r2 = r2_score(y_test, dt_pred)

print(f"Pohon Keputusan - MAE: ${dt_mae:,.0f}, R²: {dt_r2:.2f}")
```

#### 4. Latih dan Random Forest

```python
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print(f"Hutan Acak - MAE: ${rf_mae:,.0f}, R²: {rf_r2:.2f}")
```

#### 5. Bandingin Hasilnya

Jalankan kode di atas dan lihat nilai MAE dan R². Model mana yang paling oke? Biasanya Hutan Acak menang karena kekuatan timnya, tapi tergantung data yang kamu punya.

---

### Tips Pilih Model yang Pas

- **Mulai Simpel:** Pakai regresi linear buat nyobain dasarnya.  
- **Mau Akurat?** Coba hutan acak atau SVM buat data yang agak ribet.  
- **Butuh Cepat?** KNN atau pohon keputusan cocok buat data kecil.  
- **Pro Tip:** Jangan takut eksperimen dan bandingin hasilnya—nggak ada model yang cocok buat semua masalah!

---

### Hal yang Sering Bikin Pusing dan Solusinya

- **Overfit:** Model kelewat cerdas, hafal data latihan. Cek pakai cross-validation biar yakin.  
- **Underfit:** Model terlalu polos, nggak nangkap pola. Tambah fitur atau ganti model.  
- **Hyperparameter:** Mainin pengaturan (misalnya, kedalaman pohon) biar hasil lebih mantap.  
- **Tips Keren:** Pakai GridSearchCV dari scikit-learn buat nyetel pengaturan otomatis.

---

### Poin Penting

- Setiap model punya kelebihan, pilih sesuai data dan tujuanmu.  
- Model ensemble kayak hutan acak biasanya lebih jago buat akurasi.  
- Selalu bandingin performa model pake metrik evaluasi.  
- Eksperimen itu kunci buat nemuin alat yang paling pas!

---

### Tantangan: Ayo Coba Sendiri!

- Tambah data atau fitur baru (misalnya, umur rumah) dan ulang lagi modelnya.  
- Coba SVM atau KNN pake scikit-learn untuk data rumah.  
- Visualisasi pohon keputusan pake `plot_tree` dari sklearn.  
- Bandingin performa buat klasifikasi (misalnya, prediksi harga > $250k).  

---

### Apa Selanjutnya?

Di pelajaran berikutnya, kita bakal masuk ke dunia nyata: bikin modelmu hidup di aplikasi! Kamu bakal belajar nyimpen, ngeload, dan pake model di dunia produksi. Terus eksplor dan sampai ketemu lagi!