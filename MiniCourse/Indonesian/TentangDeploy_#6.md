# Kursus Mini tentang AI dan Pembelajaran Mesin

## Pelajaran 6: Bikin Model AI Kamu Hidup! (Simpan, Muat, dan Pakai di Dunia Nyata)

Halo lagi! Selamat datang di Pelajaran 6 🎉  
Kamu sudah bisa bikin model yang jago nebak harga rumah, bandingin LinearRegression vs RandomForest, dan tahu mana yang paling oke. Tapi… kalau modelnya cuma hidup di file Jupyter Notebook doang, ya sama aja kayak bikin kue enak tapi cuma kamu sendiri yang makan.  

Sekarang waktunya kita bikin modelmu “hidup” dan bisa dipakai orang lain — bahkan bisa dipakai di aplikasi HP, website, atau robot sekalipun! Kita akan belajar cara menyimpan model, memuatnya kembali, dan membuat prediksi super cepat tanpa harus latih ulang setiap kali. Yuk langsung gas!

---

### Kenapa Harus Menyimpan Model?

Bayangin kamu latih Random Forest selama 2 jam dengan ribuan data. Besok temenmu mau pakai model itu… apa iya harus latih ulang dari nol lagi? Rugi waktu banget!  

Menyimpan model = nyimpan “otak” AI yang sudah pintar. Kita cuma simpan sekali, terus bisa dipakai berkali-kali, bahkan di komputer atau server lain.

---

### Tool Andalan: Joblib dan Pickle

Di Python (scikit-learn), ada dua cara paling populer buat nyimpan model:

| Tool      | Kelebihan                              | Cocok buat                          |
|-----------|----------------------------------------|-------------------------------------|
| `joblib`  | Lebih cepat & efisien buat model besar | Direkomendasikan scikit-learn       |
| `pickle`  | Bawaan Python, simpel                  | Model kecil sampai sedang           |

Kita pakai `joblib` karena biasanya lebih ngebut.

---

### Langsung Praktik: Simpan & Muat Model Harga Rumah

Kita lanjut pakai data dan Random Forest dari Pelajaran 5.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib   # <- ini yang baru!

# Data yang sama seperti Pelajaran 5
data = {
    'Ukuran': [1000, 1500, 1200, 1800, 900, 1400, 1600, 1100, 2000, 1300],
    'Kamar': [2, 3, 3, 4, 2, 3, 4, 2, 5, 3],
    'Lokasi': [1, 0, 1, 0, 1, 0, 0, 1, 1, 0],
    'Harga': [200000, 250000, 220000, 300000, 180000, 240000, 280000, 210000, 350000, 230000]
}
df = pd.DataFrame(data)

X = df[['Ukuran', 'Kamar', 'Lokasi']]
y = df['Harga']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Latih model (hanya sekali!)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Cek performa dulu biar yakin
pred = model.predict(X_test)
print(f"MAE: ${mean_absolute_error(y_test, pred):,.0f}")
print(f"R²: {r2_score(y_test, pred):.2f}")
```

#### 1. Simpan Model ke File

```python
# Simpan model jadi file .joblib atau .pkl
joblib.dump(model, 'model_harga_rumah_terbaik.joblib')

print("Model sudah disimpan! 🎉")
```

Sekarang ada file `model_harga_rumah_terbaik.joblib` di folder kamu.

#### 2. Muat Lagi Model (Bisa Besok, Bisa di Komputer Lain!)

```python
# Hapus variabel model dulu biar yakin kita benar-benar load dari file
del model

# Muat kembali
model_loaded = joblib.load('model_harga_rumah_terbaik.joblib')

print("Model berhasil dimuat ulang!")
```

#### 3. Langsung Pakai Buat Prediksi Baru (Tanpa Latih Lagi!)

```python
# Contoh rumah baru
rumah_baru = pd.DataFrame({
    'Ukuran': [1350, 2000],
    'Kamar': [3, 5],
    'Lokasi': [1, 0]   # 1 = Kota, 0 = Pinggiran
})

prediksi = model_loaded.predict(rumah_baru)

print(f"Prediksi harga rumah 1: ${prediksi[0]:,.0f}")
print(f"Prediksi harga rumah 2: ${prediksi[1]:,.0f}")
```

Keren, kan? Modelnya langsung bisa dipakai kapan saja!

---

### Bonus: Bikin Fungsi Prediksi yang Rapi (Siap Dipakai Teman)

```python
def prediksi_harga_rumah(ukuran, kamar, lokasi_kota=True):
    # Load model (bisa cuma sekali kalau di aplikasi besar)
    model = joblib.load('model_harga_rumah_terbaik.joblib')
    
    lokasi = 1 if lokasi_kota else 0
    data_baru = pd.DataFrame({
        'Ukuran': [ukuran],
        'Kamar': [kamar],
        'Lokasi': [lokasi]
    })
    
    harga = model.predict(data_baru)[0]
    return f"Estimasi harga rumah: ${harga:,.0f}"

# Contoh pakai
print(prediksi_harga_rumah(1700, 4, lokasi_kota=True))
print(prediksi_harga_rumah(1100, 2, lokasi_kota=False))
```

Sekarang temenmu tinggal panggil fungsi ini tanpa ngerti machine learning sama sekali!

---

### Mau Lebih Keren Lagi? Deploy ke Web atau HP!

Beberapa cara populer (akan kita bahas mendalam di pelajaran bonus nanti):

- Streamlit → bikin web app dalam 10 menit  
- FastAPI/Flask → buat API yang bisa dipanggil dari mana saja  
- TensorFlow Lite / ONNX → pakai model di Android/iOS  
- HuggingFace atau Vercel → deploy gratis dalam hitungan klik

---

### Kesalahan Umum & Tips Anti Gagal

| Masalah                        | Solusi                                                                 |
|--------------------------------|------------------------------------------------------------------------|
| File model gede banget (>500 MB)| Pakai `joblib` + kompresi, atau kurangi `n_estimators` di RandomForest |
| Error saat load di komputer lain| Pastikan versi scikit-learn sama (tulis di `requirements.txt`)        |
| Lupa simpan scaler/preprocessing| Simpan juga `StandardScaler`, `OneHotEncoder`, dll pakai joblib           |

---

### Poin Penting Pelajaran 6

- Model yang sudah dilatih bisa disimpan pakai `joblib` atau `pickle`.  
- Setelah disimpan, model bisa dipakai berulang-ulang tanpa latih ulang → hemat waktu & listrik!  
- Kamu bisa bikin fungsi prediksi yang rapi atau bahkan aplikasi web/HP.  
- Dari sini, modelmu sudah siap masuk ke dunia nyata.

---

### Tantangan: Coba Sendiri!

1. Simpan model LinearRegression dan RandomForest, lalu bandingin ukuran filenya.  
2. Buat fungsi prediksi yang bisa nerima input dari `input()` di terminal.  
3. (Level atas) Coba bikin aplikasi Streamlit sederhana (saya kasih kode dasar di bawah kalau mau):

```python
# streamlit_app.py
import streamlit as st
import joblib
import pandas as pd

model = joblib.load('model_harga_rumah_terbaik.joblib')

st.title("Prediksi Harga Rumah 🏡")
ukuran = st.slider("Ukuran (m²)", 500, 3000, 1500)
kamar = st.slider("Jumlah kamar", 1, 6, 3)
lokasi = st.selectbox("Lokasi", ["Kota", "Pinggiran"])

lok = 1 if lokasi == "Kota" else 0
pred = model.predict([[ukuran, kamar, lok]])[0]

st.success(f"Estimasi harga: **${pred:,.0f}**")
```

Jalankan dengan: `streamlit run streamlit_app.py`

---

### Apa Selanjutnya?

Pelajaran 7 : “Proyek Mini – Bikin Aplikasi Prediksi Harga Rumah Lengkap dari A-Z”  
+ bonus: cara deploy gratis ke internet supaya temenmu bisa coba langsung lewat link!

Tetap semangat ngoding dan sampai jumpa di Pelajaran 7! 🚀