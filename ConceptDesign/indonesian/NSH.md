# NSH - Naive Specialists Hierarchy

## Definisi

NSH atau Naive Specialists Hierarchy adalah konsep yang memperkenalkan arsitektur machine learning yang memungkinkan perluasan fungsi kognitif.

## Dukungan teori

Perluasan fungsi kognitif adalah salah satu cara menuju generalisasi, neural network (NN) secara matematis hanya mempertajam 1 fungsi kognitif saja yang berakhir pada asumsi bahwa perluasan kemampuan kognitif tidak didasari oleh NN yang besar/dalam namun oleh teknik training dan harmonisasi informasi lintas paradigma pada arsitektur.

## Struktur arsitektur

NSH terdiri dari kumpulan model yang setiap modelnya mendapatkan bagian informasi yang berbeda-beda, dan layer yang menaungi seluruh model dalam sistem yang dapat disebut sebagai 'payung'. 

## Training

### Pre-process

Payung mengatur informasi apa saja yang akan diberikan pada seluruh model (data fitur), pemotongan data fitur didasari oleh kekayaan informasi, dan variasi informasi fitur. Hal ini mampu menciptakan efek spesialisasi pada setiap model. Efek spesialisasi ini lah yang membuat NSH memungkinkan adanya pemisahan antara model yang optimal dan tidak optimal pada input user.

### Process dan trust

Payung dapat menyaring model-model lewat mekanisme *trust* yang dapat memengaruhi output setiap model. Setiap model memiliki trust berdasarkan besaran, dan stabilitas loss, skor kemampuan pada data validasi saat proses training, dan keyakinan model akan outputnya. Model dengan trust > 1 dapat disebut sebagai prior (priority), trust ~1 disebut major (majority), dan < 0 disebut (freezed), model freezed tidak ikut berpartisipasi dalam proses prediksi sampai trust > 0 pada input user berikutnya. Trust setiap model dapat berubah-ubah tergantung performanya pada data yang diinput user. Juga, trust dapat berkurang seiring user memanggil NSH untuk melakukan prediksi, trust besar akan berkurang (decaying) lebih cepat dibandingkan model dengan trust kecil.

## Prediksi

Payung menghasilkan prediksi akhir dengan menyimpan data representasi output semua model yang setelahnya dikalikan oleh trust masing-masing model, sebelum akhirnya di jumlahkan sebagai prediksi akhir.

## Evaluasi

Payung mampu mengevaluasi diri menggunakan data prediksinya dengan fungsi yang membuat setiap model memprediksi label dari data yang diberikan lalu performa seperti metrik dan loss jika memiliki perkembangan dari training sebelumnya maka trust dapat naik berdasarkan selisih performa dan sebaliknya, jika hasil evaluasi menyatakan adanya kemunduran pada suatu model maka trust model akan berkurang berdasarkan selisih performa.