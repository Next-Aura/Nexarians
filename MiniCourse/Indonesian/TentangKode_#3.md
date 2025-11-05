# Kursus Mini tentang AI dan Pembelajaran Mesin

## Pelajaran 3: Bikin Kode ML Pertamamu – Nebakin Harga Rumah

Selamat datang di Pelajaran 3! Setelah ngerti apa itu AI dan cara nyiapin data, sekarang saatnya praktek langsung: kita bikin model pembelajaran mesin pake kode! Gak usah panik kalau kamu baru mulai ngoding—panduan ini bakal langkah demi langkah dan super gampang buat pemula. Ayo, gas!

---

### Kenapa Pake Python?

Python itu kayak “pisau lipat serbaguna” buat AI dan pembelajaran mesin. Gampang dipahami, kuat, dan punya banyak library yang bikin kerjaan kita lebih gampang. Pokoknya, pilihan nomor satu buat urusan data!

---

### Siapin Peralatanmu

Sebelum mulai ngoding, kamu butuh:
- **Python** (bisa download di [python.org](https://www.python.org/)).
- **Jupyter Notebook** (buat nulis dan jalanin kode langsung di browser).
- **scikit-learn** (library kece buat bikin model ML).

**Cara Setup Cepet (jalanin di terminal atau command prompt):**
```sh
pip install notebook scikit-learn pandas
```
Terus buka Jupyter Notebook pake:
```sh
jupyter notebook
```

---

### Langkah Demi Langkah: Nebakin Harga Rumah

Kita bakal pake contoh harga rumah dari pelajaran sebelumnya dan bikin model sederhana pake **LinearRegression**. Yuk, ikutin langkah-langkahnya:

#### 1. Ambil Toolkit (Import Library)

Bayangin library itu kayak kotak alat. Kita bakal pake:
- `pandas` buat ngatur data (mirip Excel, tapi lebih kece).
- `scikit-learn` buat bikin model ML.

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
```

#### 2. Nyiapin Data

Kita bikin tabel kecil buat data rumah.

```python
# Data rumah kita
data = {
    'Ukuran': [1000, 1500, 1200],
    'Harga': [200000, 250000, 220000]
}
df = pd.DataFrame(data)
```

#### 3. Pisahin Fitur dan Label

- **Fitur:** Yang dipake model buat nebak (misalnya, ukuran rumah).
- **Label:** Yang mau kita tebak (misalnya, harga).

```python
X = df[['Ukuran']]  # Fitur (harus 2D)
y = df['Harga']     # Label
```

#### 4. Latih Model

Sekarang kita ajarin model pake data tadi.

```python
model = LinearRegression()
model.fit(X, y)
```

#### 5. Bikin Prediksi

Coba nebak harga rumah baru, misalnya ukuran 1300 sq ft:

```python
harga_tebakan = model.predict([[1300]])
print(f"Harga tebakan buat rumah 1300 sq ft: ${harga_tebakan[0]:,.0f}")
```

---

### Apa yang Lagi Terjadi?

- **Kita kasih contoh ke model** (ukuran rumah sama harganya).
- **Model belajar pola** (rumah makin gede, harganya makin mahal).
- **Kita minta tebakan** buat harga rumah baru berdasarkan ukuran.

#### Analogi

Bayangin kamu nunjukin beberapa harga rumah ke temen, terus nanya, “Menurutmu, rumah 1300 sq ft harganya berapa?” Model kita ini kayak temen yang nebak pake pola yang dia pelajari tadi.

---

### Kesimpulan Utama

- Python sama scikit-learn bikin bikin model ML jadi gampang banget.
- Cuma pake beberapa baris kode, kamu udah bisa bikin model sederhana.
- Prosesnya simpel: nyiapin data → latih model → bikin prediksi.

---

### Tantangan: Coba Sendiri!

- Ganti ukuran dan harga rumah di data tadi.
- Coba nebak harga buat ukuran rumah lain.
- Apa yang terjadi kalau kamu tambah lebih banyak data?

---

### Apa Selanjutnya?

Di pelajaran berikutnya, kita bakal bahas cara ngecek seberapa jago modelmu (evaluasi akurasi) dan coba tambahin fitur lain (misalnya, lokasi). Tetap semangat ngoding dan have fun!