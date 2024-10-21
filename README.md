# Laporan Proyek Machine Learning - Farras Nur Haidar Ramadhan

## Domain Proyek

### Latar Belakang

Pada tahun 2021, Chicago mengalami salah satu tahun paling mematikan dalam lebih dari dua dekade, dengan **lebih dari 800 pembunuhan** yang tercatat. Peningkatan ini tidak hanya menjadikan tahun tersebut sebagai salah satu yang paling berbahaya dalam sejarah kota, tetapi juga menunjukkan peningkatan signifikan dalam kekerasan bersenjata. **Chicago Police Department** melaporkan bahwa insiden penembakan juga mencapai **3.561 kasus**, yang menunjukkan kenaikan lebih dari **300 insiden** dibandingkan tahun 2020 dan lebih dari **1.400 insiden** dibandingkan dengan tahun 2019 [^1][^2].

Kekerasan bersenjata ini sebagian besar dipicu oleh konflik antar kelompok atau yang disebut dengan geng, yang telah menjadi masalah utama di kota ini. Sebagai kota terbesar ketiga di Negeri Paman Sam, Chicago mencatat lebih banyak pembunuhan daripada kota-kota besar lainnya, seperti New York dan Los Angeles, yang masing-masing mencatat setidaknya **300 pembunuhan lebih sedikit** pada tahun yang sama [^3][^4]. Lonjakan kekerasan tersebut menyoroti tantangan besar yang dihadapi oleh penegak hukum dalam menjaga keamanan umum, meskipun ada upaya untuk mengurangi senjata ilegal di jalanan dan meningkatkan jumlah petugas investigasi [^5][^6].

Berdasarkan paparan latar belakang diatas, dapat dibuat model machine learning untuk memprediksi pola kejahatan yang ada di kota Chicago, dengan memanfaatkan data historis kriminal, seperti pembunuhan dan penembakan.

Model yang telah dibuat ini diharapkan untuk dapat digunakan sebagai langkah antisipasi dalam pengamanan lebih ketat oleh pihak berwajib, seperti memperkuat patroli di area-area berisiko tinggi dan waktu-waktu tertentu yang rentan terhadap kejadian kekerasan.

[^1]: [WTTW News - 2021 Ends as Chicago’s Deadliest Year](https://news.wttw.com/2022/01/02/2021-ends-chicago-s-deadliest-year-quarter-century)
[^2]: [Chicago CRED 2021 Annual Report](https://www.chicagocred.org/blog/2021-annual-report/)
[^3]: [Chicago Journal - 2021 ends as Chicago's deadliest year](https://www.chicagojournal.com/2021-ends-as-chicagos-deadliest-year-in-a-quarter-century/)
[^4]: [WTTW News](https://news.wttw.com/2022/01/02/2021-ends-chicago-s-deadliest-year-quarter-century)
[^5]: [Chicago CRED](https://www.chicagocred.org/blog/2021-annual-report/)
[^6]: [Chicago Police Department Report](https://home.chicagopolice.org/statistics-data/crime-statistics/)

## Business Understanding

Dari penjelasan latar belakang di atas, dapat dibuat rumusan masalah sebagai berikut

### Problem Statements

- Bagaimana menyiapkan data yang diperlukan untuk membuat model machine learning?
- Bagaimana cara membuat model machine learning untuk kebutuhan klasifikasi jenis kejahatan?
### Goals

Berdasarkan rumusan masalah sebelumnya, dapat dibuatkan tujuan laporan sebagai berikut
- Melakukan tahapan persiapan data, agar data yang sudah disiapkan dapat dimasukkan ke dalam model
- Membuat model machine learning untuk mengklasifikasikan jenis kejahatan
### Solution Statemnent

Berdasarkan dari tujuan, didapatkan beberapa solusi untuk menjawab rumusan masalah sebagai berikut

- Melakukan pembagian pada data menjadi data train data data test dengan rasio sebesar 80:20
- Menggunakan 3 algoritma yaitu blablabla. untuk membandingkan akurasi dari 3 algoritma yang diujikan

    ### Solution statements
    - Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
    - Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Dataset yang digunakan untuk proyek ini adalah "Chicago Crime Dataset 2018 to 2021" yang diambil dari laman Kaggle. Dataset tersebut memiliki 4 file dengan format csv berukuran 226.34 MB

Dikarenakan kriteria dari Dicoding untuk menggunakan data kuantitatif, di proyek ini akan menghapus beberapa fitur yang tidak digunakan dan hanya akan menggunakan fitur dengan tipe data numeric pada dataset tersebut, diantaranya:
- Latitude
  Fitur ini berisi tentang informasi koordinasi lokasi berupa garis lintang
- Longitude
  Fitur ini berisi tentang informasi koordinasi lokasi berupa garis bujur
- Hour
  Fitur ini berisi tentang informasi waktu kejadian, fitur ini bentuknya        dalam format jam (0-23)
- Day of Week
  Fitur ini berisi tentang hari kejadian dalam seminggu, dimulai dari 0 yaitu hari Senin dan berakhir di angka 6 yaitu Ahad

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
