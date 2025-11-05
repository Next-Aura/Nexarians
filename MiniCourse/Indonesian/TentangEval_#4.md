# Kursus Mini tentang AI dan Pembelajaran Mesin

## Pelajaran 4: Ngecek Akurasi Model dan Nambah Fitur

Selamat datang di Pelajaran 4! Kamu udah ngerti dasar-dasar AI, nyiapin data, dan bikin model pertama pake kode. Sekarang, kita fokus bikin modelmu makin jago: ngecek seberapa bagus performanya dan nambah fitur biar prediksinya makin akurat. Kita lanjut pake contoh harga rumah, yuk mulai!

---

### Kenapa Harus Ngecek Akurasi Model?

Bayangin kamu bikin kue. Resep udah diikutin, tapi apa iya kuenya enak? Ya, harus dicobain dulu! Di pembelajaran mesin, ngecek akurasi itu kayak nyicipin kue—buat tahu seberapa oke modelmu dan apa yang bisa diperbaiki.

Tanpa ngecek, kamu mungkin ngira modelmu udah top, padahal bisa aja salah gede. Evaluasi bantu kamu ukur kesalahan dan bikin model lebih bisa dipercaya.

---

### Metrik Kunci buat Evaluasi

Ini beberapa cara gampang buat cek performa modelmu:

1. **Mean Absolute Error (MAE)**  
   - **Apa itu?** Rata-rata selisih absolut antara prediksi sama nilai asli.  
   - **Contoh:** Kalau model nebak harga rumah $210,000 tapi aslinya $200,000, errornya $10,000. MAE ngitung rata-rata error kayak gini.  
   - **Analogi:** Kayak ngukur seberapa meleset tebakanmu di permainan, rata-rata.

2. **Mean Squared Error (MSE)**  
   - **Apa itu?** Mirip MAE, tapi errornya dikuadratin dulu sebelum dirata-rata (ngasih hukuman lebih besar buat kesalahan gede).  
   - **Contoh:** Untuk error $10,000 tadi, MSE jadi (10,000)^2 = 100,000,000.  
   - **Analogi:** Kayak main game yang kesalahan gede bikin skor anjlok lebih parah.

3. **R-Squared (R²)**  
   - **Apa itu?** Skor dari 0 sampe 1 yang nunjukin seberapa baik modelmu nerangin data (1 artinya sempurna).  
   - **Contoh:** R² 0.8 berarti 80% variasi harga rumah bisa dijelasin modelmu.  
   - **Analogi:** Kayak nilai di rapor—makin tinggi, makin jago!

---

### Langkah Demi Langkah: Ngecek Model Harga Rumah

Kita pake kode dari Pelajaran 3 dan tambahin evaluasi.

#### 1. Ambil Toolkit dan Siapin Data

Kita pake data yang sama, tapi sekarang tambah set buat tes.

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Data kita (diperbanyak)
data = {
    'Size': [1000, 1500, 1200, 1800, 900, 1400],
    'Price': [200000, 250000, 220000, 300000, 180000, 240000]
}
df = pd.DataFrame(data)

# Bagi jadi data latih dan tes
X = df[['Size']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

#### 2. Latih Model

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

#### 3. Bikin Prediksi dan Evaluasi

```python
y_pred = model.predict(X_test)

# Hitung metrik
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: ${mae:,.0f}")
print(f"Mean Squared Error: ${mse:,.0f}")
print(f"R-Squared: {r2:.2f}")
```

#### 4. Baca Hasilnya

- MAE/MSE rendah dan R² tinggi artinya modelmu oke.  
- Kalau errornya gede, model perlu diperbaiki (butuh lebih banyak data atau fitur yang lebih baik).

---

### Nambah Fitur Baru

Sekarang model kita cuma pake ukuran rumah. Ayo tambahin lokasi biar modelnya makin pinter!

#### Kenapa Nambah Fitur?

Fitur tambahan kasih model lebih banyak info, kayak nambah bumbu ke masakan biar rasanya lebih nendang.

#### Data Baru dengan Lokasi

Kita ubah lokasi jadi angka (Kota=1, Pinggiran=0).

```python
# Data baru
data = {
    'Size': [1000, 1500, 1200, 1800, 900, 1400],
    'Location': [1, 0, 1, 0, 1, 0],  # 1=Kota, 0=Pinggiran
    'Price': [200000, 250000, 220000, 300000, 180000, 240000]
}
df = pd.DataFrame(data)

X = df[['Size', 'Location']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE dengan Lokasi: ${mae:,.0f}")
print(f"R² dengan Lokasi: {r2:.2f}")
```

#### Apa yang Berubah?

- Model sekarang mikirin ukuran rumah *dan* lokasi.  
- Prediksinya harusnya lebih jago (MAE lebih kecil, R² lebih besar).

---

### Kesalahan Umum dan Tips

- **Overfitting:** Model terlalu jago di data latih, tapi jelek di data baru. **Solusi:** Pake lebih banyak data atau model yang lebih sederhana.  
- **Underfitting:** Model terlalu simpel dan ga nangkap pola. **Solusi:** Tambah fitur atau pake model yang lebih canggih.  
- **Tips:** Selalu pisahin data jadi latih dan tes biar ga *overfit*.

---

### Kesimpulan Utama

- Ngecek model pake metrik kayak MAE, MSE, dan R² buat tahu seberapa jago modelmu.  
- Nambah fitur (misalnya lokasi) bisa bikin prediksi lebih akurat.  
- Evaluasi adalah kunci buat bikin model AI yang bisa dipercaya.

---

### Tantangan: Coba Sendiri!

- Tambah fitur lain ke data rumah (misalnya, jumlah kamar tidur).  
- Coba ubah ukuran data tes (misalnya, 20% vs 50%).  
- Bandingin MAE sebelum dan sesudah nambah fitur.

---

### Apa Selanjutnya?

Di pelajaran berikutnya, kita bakal jelajahi jenis model lain (selain LinearRegression) dan kapan harus pakai mereka. Tetap semangat ngulik dan sampai ketemu lagi!