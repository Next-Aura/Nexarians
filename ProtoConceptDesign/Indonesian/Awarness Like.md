# Awareness Like

## Pendahuluan

Machine learning (ML) adalah momen ketika suatu program mampu menghasilkan hal yang sebelumnya tidak ia 'lihat' dari proses training.
Beberapa unsur utama pada machine learning seperti fungsi kesalahan (loss funstion), derivatif-nya, dan juga besar 'langkah' yang diambil dalam meminimalkan kesalahan (loss).
Besarnya 'langkah' ini bisa dilihat berdasarkan besarnya *learning rate* yang ditentukan, learning rate pada machine learning berperan penting dalam meminimalkan kesalahan model,
learning rate pada model machine learning khususnya model gradien linear bekerja dengan mengkalikan hasil gradien fungsi kesalahan sebelum memengaruhi parameter seperti weight atau coefficient yang secara umum dapat dinotasikan sebagai:

W_t = W_t-1 - lr * grad_w

contoh diatas merupakan notasi paling umum pada machine learning saat model sedang dalam proses training.
Learning rate sendiri sifatnya berbeda-beda tergantung pada 'penjadwal' yang digunakan, umumnya penjadwal learning rate punya sifat seperti:

- Constant
- Decay
- Adaptive

## Pendalaman

Sifat-sifat learning rate selama training biasanya dipilih sesuai dengan kebutuhan pengguna dari pertimbangan keselarasan karakteristik model dengan sifat learning rate, asumsi stabilitas fungsi kesalahan (loss) pada saat training, dan faktor lainnya. Sifat-sifat learning rate yang memiliki masing-masing karakter dapat dijabarkan seperti:

### Constant
Learning rate tidak berubah selama proses training, umumnya penjadwal didefinisikan sebagai 'Constant' yang umumnya digunakan saat interpretasi dan kejelasan performa model diutamakan.

### Decay
Learning rate berkurang seiring berjalannya proses pelatihan, 
contoh penjadwalnya adalah 'Invscaling' yang secara eksponensial mengurangi learning rate, umumnya digunakan secara default oleh model-model dengan karakteristik fungsi kesalahan (loss) konveks.

### Adaptive
Learning rate berubah mengikuti pergerakan suatu parameter, contoh penjadwalnya adalah "LROnPlateau" yang akan menurunkan learning rate saat progress penurunan kesalahan stagnant. Umumnya digunakan saat training panjang dengan banyak epoch dan stagnansi performa menjadi permasalahan utama.

Beberapa penjadwal juga membuat lr 'terprogram' secara rumus saat proses training seperti one-cycle. 

## Pendapat

Sifat-sifat lr yang didasari penjadwal bertujuan untuk masalah-masalah spesifik maupun default umum, namun beberapa penjadwal populer aku rasa terlalu terpaku pada penurunan lr dan kekangan rumus matematis. Aku merasa mungkin seharusnya ada penjadwal yang dapat 'mengerti' situasi training dan menyesuaikan diri.

> CoDe: 1. **RatB (Cached)**, 2. **ConB**
> Paper: -