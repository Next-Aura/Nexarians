# RatB - Ratio Based

## Definisi

Konsep AdaLR atau 'Adaptive Learning Rate' merupakan bagian dari penjadwal learning rate dengan sifat adaptif.
AdaLR sebagai konsep penjadwal memiliki keuntungan yang mencolok, yaitu kemampuannya dalam menyesuaikan learning rate berdasarkan suatu patokan yang ditentukan.

**RatBLR** (Ratio Based Learning Rate) hadir sebagai penjadwal learning rate yang berbasis rasio kesalahan (loss). Konsep ini didasari dari ide pokok seorang individu yang menyatakan bahwa penjadwal learning rate sebaiknya memiliki kemampuan untuk beradaptasi secara real-time berdasarkan rasio kesalahan.

## Dukungan teori

Pendapat pribadi dari hasil pengamatan dan obsevasi ku menyatakan semakin besar loss maka semakin kacau model saat proses training, dan learning rate sebagai pengatur langkah 'pergerakan' model dalam menangkap sinyal gradien punya potensi mengembalikan situasi dengan mengatur loss ke arah yang benar, yaitu arah yang mampu menurunkan loss. Dalam observasi ku, saat loss mengalami kenaikan lr yang lebih besar secara konsisten mampu menurunkan loss, secara matematis hal ini dapat disebabkan karna learning rate yang lebih besar mampu mengambil sinyal gradient lebih banyak yang diharapkan mampu menurukan loss.

## Implementasi

RatBLR bekerja dengan menggantungkan besaran learning rate menggunakan rasio kesalahan pada proses training model, RatBLR dapat dinotasikan secara garis besar sebagai:

$\text{lr_rate}_t = \text{lr_rate}_{t-1} \cdot \left( \frac{\text{loss}_{t-1}}{\text{loss}_{t-2}} \right)$

Learning rate baru berasal dari learning rate indeks sebelumnya yang dikalikan dengan rasio kesalahan pada pembagian $\text{loss}_{t-1}$ dan $\text{loss}_{t-2}$, dengan pembagian yang seperti itu dapat menghasilkan sifat matematis yang jikalau $\text{loss}_{t-1} > \text{loss}_{t-2}$ maka
learning rate yang dikali rasio akan bertambah sesuai dengan yang dihasilkan dari pembagian kesalahan, dan sebaliknya jika $\text{loss}_{t-1} < \text{loss}_{t-2}$, maka perkalian dengan rasio kesalahan akan membuat learning rate menjadi lebih kecil, asumsikan nilai fungsi kesalahan (loss) umum pada machine learning klasik adalah non-negatif.
Dengan definisi yang demikian mampu membuat RatBLR cukup efektif dalam menangani dataset dengan noise yang tinggi dan berpotensi menjadi penjadwal learning rate yang terbukti efektif dalam beberapa kasus.

Dari penjelasan diatas implementasi RatBLR bisa diperluas lagi dengan menambahkan 'window' pada loss guna mendapatkan rasio yang lebih 'luas' dan merata,
perluasan implementasi tersebut dapat dinotasikan sebagai berikut (menggunakan dasar sintaks bahasa pemrograman Python):

$\text{lr_rate}_t = \text{lr_rate}_{t-1} \cdot \frac{\text{mean}(\text{loss}[-window:])}{\text{mean}(\text{loss}[-2window:-window])}$

Dengan implementasi yang demikian membuat RatBLR dapat lebih 'mengevaluasi' langkah dalam menambah dan mengurangi learning rate.
Namun bentuk tersebut masih kurang efisien dikarenakan jika rasio loss hanya berada di sekitar 1 maka kenaikan dan penurunan learning rate akan sangat kecil dan memperlambat proses training, 
mengatasi masalah tersebut bentuk rumus terakhir dapat dikuatkan kembali dengan merubah bentuknya menjadi:

$$
\text{lr} =
\begin{cases}
\text{lr} \cdot (i \cdot t^{p}), & \text{jika } r \leq 1 \\
\text{lr} \cdot r, & \text{jika } r > 1
\end{cases}
$$

dengan

$$
r = \sqrt{ \frac{\text{mean}(\text{loss}[-window:])}{\text{mean}(\text{loss}[-2window:-window])} }
$$

Bentuk seperti ini akan membuat rasio menjadi lebih terjaga dari overflow dan menjaga learning rate agar tetap memberikan efek yang kontributif.
Rasio dibawah 1 akan membuat learning rate diturunkan menggunakan formula invscaling guna menciptakan penurunan learning rate yang mulus dan teratur.
Rasio diatas 1 akan membuat learning rate dinaikan sesuai rasio, rasio yang di-akar dapat diasumsikan sudah ter-skala dengan baik dan aman yang dapat mengurangi kemungkinan learning rate mengalami overflow saat kenaikan.

Sensitivitas alami yang dimiliki RatBLR secara teoritis cocok dengan model linear selayaknya keluarga gradient descent dikarenakan loss yang cenderung mulus dan representasinya yang cenderung mudah terjelaskan jika dibandingkan dengan neural network (NN) dan jenis model non-linear lainnya.
Terlepas dari pemaparan keuntungan konsep diatas, 'non-linearitas' yang dimiliki RatBLR dapat berpotensi menghasilkan hasil yang tidak terduga seperti nilai tidak terbatas (infinity) atau NaN (Not a Number) dikarenakan formula RatBLR tidak memiliki batasan sifat matematis secara langsung,
sehingga dalam praktiknya RatBLR menggunakan pemotongan atau clipping untuk mencegah munculnya nilai yang tidak terduga.

Konsep RatBLR dirancang dan dikhususkan untuk machine learning klasik dimana fluktuasi kesalahan atau loss dapat dijelaskan karna kecendrungan sifat linearnya.

## Pendalaman implementasi

Pendalaman implementasi berfungsi untuk meluaskan kasus penggunaan (use case) dari konsep AdaLR yang dijelaskan barusan. Pendalaman implementasi yang dimaksud adalah membuat konsep AdaLR yang dijabarkan tadi kompatibel dengan jenis machine learning yang lebih canggih seperti neural network (NN).

NN terkenal dengan karakteristik kesalahannya (loss) yang sangat tidak stabil dan sifat non-linearitasnya menambah kerumitan NN dari segi representasi kesalahan (loss).
Karena demikian penjadwal learning rate RatBLR secara teoritis tidak stabil jika diimplementasikan dalam sistem NN, non-linearitas kesalahan (loss) pada NN dapat mengacau rasio loss yang menyebabkan RatBLR mengatur besaran learning rate secara tidak stabil.
Mengatasi hal ini formula RatBLR sedikit berubah demi menyesuaikan karakteristik NN, perubahan yang paling mendasar adalah skala rasio yang sekarang diatasi bukan dengan akar kuadrat namun logaritma dan konversi ke nilai absolut untuk mencegah rasio merubah tanda lr saat dikali karna loss negatif bukanlah hal yang mustahil dan umum pada NN, perubahan yang demikian dapat dinotasikan sebagai:

$$
r = \abs( \log({ \frac{\left(\text{mean}(\text{loss}[-window:])\right)}{\text{mean}(\text{loss}[-2window:-window])} }) )
$$

Dengan formula yang demikian rasio akan menjadi lebih stabil dan terorganisir dengan baik yang berpotensi mengurangi sensitivitas algoritma.

Skala rasio yang terkena efek logaritma sudah cukup untuk menambah stabilitas namun sekali lagi dalam sistem NN loss sangat fluktuatif dan untuk menambah stabilitas algoritma formula ini mampu diberikan peningkatan berupa sisitem 'kesabaran' (patience),
sistem patience bekerja dengan menahan perilaku algoritma dalam mengatur besaran learning rate walaupun sudah mencapai threshold rasio sampai batas patience yang ditentukan sebelum akhirnya algoritma dapat mengatur besaran learning rate.
Dengan tambahan mekanisme patience, notasi algoritma dapat didefinisikan sebagai:

$$
\text{lr} =
\begin{cases}
\text{lr} \cdot (i \cdot t^{p}), & \text{jika } r \leq 1 \text{ dan } \text{wait} \geq \text{patience} \\
\text{lr} \cdot r, & \text{jika } r > 1 \text{ dan } \text{wait} \geq \text{patience}
\end{cases}
$$

dengan

$$
r = \abs( \log({ \frac{\left(\text{mean}(\text{loss}[-window:])\right)}{\text{mean}(\text{loss}[-2window:-window])} }) )
$$

$\text{wait}$ berfungsi sebagai 'hitungan kesabaran' algoritma saat rasio sudah mencapai ambang batasnya.
Dengan tambahan mekanisme dan stabilitas berupa logaritma secara teoritis algoritma barusan mampu digunakan dan optimal dalam pengembangan NN kecil, sedang bahkan besar.
Definisi terakhir dari algoritma dapat disebut sebagai **RobustRat** atau Robust Ratio (Rasio yang kokoh) yang menggambarkan karakteristik dari algoritma.

## Kesimpulan

Penjadwal learning rate merupakan hal yang krusial dalam machine learning karna berperan langsung dengan bagaimana parameter seperti weight atau coefficient diperbarui. Konsep **Ratio Based (RatB)** yang berisi **Ratio Based Learning Rate (RatBLR)** dan **Robust Ratio (RobustRat)** sebagai bentuk yang lebih stabil untuk neural network (NN) menjadi opsi penjadwal learning rate yang mampu menciptakan efek adaptif namun tetap kontributif saat proses training model.