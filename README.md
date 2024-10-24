# Laporan Proyek Machine Learning - Farras Nur Haidar Ramadhan

## Domain Proyek

### Latar Belakang

Pada tahun 2021, Chicago mengalami salah satu tahun paling mematikan dalam lebih dari dua dekade, dengan **lebih dari 800 pembunuhan** yang tercatat. Peningkatan ini tidak hanya menjadikan tahun tersebut sebagai salah satu yang paling berbahaya dalam sejarah kota, tetapi juga menunjukkan peningkatan signifikan dalam kekerasan bersenjata. **Chicago Police Department** melaporkan bahwa insiden penembakan juga mencapai **3.561 kasus**, yang menunjukkan kenaikan lebih dari **300 insiden** dibandingkan tahun 2020 dan lebih dari **1.400 insiden** dibandingkan dengan tahun 2019 [^1][^2].

Kekerasan bersenjata ini sebagian besar dipicu oleh konflik antar kelompok atau yang disebut dengan geng, yang telah menjadi masalah utama di kota ini. Sebagai kota terbesar ketiga di Negeri Paman Sam, Chicago mencatat lebih banyak pembunuhan daripada kota-kota besar lainnya, seperti New York dan Los Angeles, yang masing-masing mencatat setidaknya **300 pembunuhan lebih sedikit** pada tahun yang sama [^3][^4]. Lonjakan kekerasan tersebut menyoroti tantangan besar yang dihadapi oleh penegak hukum dalam menjaga keamanan umum, meskipun ada upaya untuk mengurangi senjata ilegal di jalanan dan meningkatkan jumlah petugas investigasi [^5][^6].

Berdasarkan paparan latar belakang diatas, dapat dibuat model machine learning untuk memprediksi pola kejahatan yang ada di kota Chicago, dengan memanfaatkan data historis kriminal, seperti pembunuhan, penembakan dan lain-lain.

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
- Melakukan tahapan persiapan data, agar data yang sudah disiapkan dapat dimasukkan ke dalam model,
- Membuat model machine learning untuk mengklasifikasikan jenis kejahatan.
  
### Solution Statemnent
Berdasarkan dari tujuan, didapatkan beberapa solusi untuk menjawab rumusan masalah sebagai berikut
- Melakukan pembagian pada data menjadi data train data data test dengan rasio sebesar 80:20
- Menargetkan Primary Type sebagai fitur yang diklasifikasikan
- Menggunakan 3 algoritma yaitu blablabla. untuk membandingkan akurasi dari 3 algoritma yang diujikan

## Data Understanding
Dataset yang digunakan untuk proyek ini adalah [Chicago Crime Dataset 2018 to 2021](https://www.kaggle.com/datasets/mingyuouyang/chicago-crime-dataset-2018-to-2021?select=Crimes_-_2018.csv) yang diambil dari laman Kaggle. Dataset tersebut memiliki 4 file dengan format csv berukuran 226.34 MB

Dataset yang telah diunduh, masih perlu dilakukan penyesuaian sampai dataset dapat digunakan dengan baik, beberapa diantaranya:
- Menggabungkan 4 berkas menjadi 1 berkas
- Melakukan penghapusan pada kolom yang tidak digunakan dalam model, yaitu kolom 
- Mengganti tipe data pada kolom Date menjadi "datetime64" dan melakukan ekstraksi kolom Date untuk mendapatkan hari (Day), dan Jam (Hour) untuk kebutuhan EDA.

Setelah proses penyesuaian pada dataset, dilakukan EDA untuk melihat informasi apa saja yang didapatkan:

### Deskripsi Variabel

| #  | Column               | Non-Null Count | Dtype          |
|----|----------------------|----------------|----------------|
| 0  | Primary Type          | 948424         | object         |
| 1  | Date                 | 948424         | datetime64[ns] |
| 2  | Location Description  | 944105         | object         |
| 3  | Arrest               | 948424         | bool           |
| 4  | Domestic             | 948424         | bool           |
| 5  | Beat                 | 948424         | int64          |
| 6  | District             | 948424         | int64          |
| 7  | Latitude             | 934039         | float64        |
| 8  | Longitude            | 934039         | float64        |
| 9  | Community Area       | 948423         | float64        |
| 10 | Hour                 | 948424         | int32          |
| 11 | Day of Week          | 948424         | int32          |

Dari tabel diatas dapat dilihat bahwa didalam dataset berisi 948424 baris dari 11 kolom. Dari 11 kolom memiliki variasi tipe data yang bervariasi yaitu terdapat tipe data `bool` dua kolom, `datetime64` satu kolom, `float64` 3 kolom, `int32` dua kolom, `int64` dua kolom, `object` dua kolom. kolom berisi informasi yakni
1. `Primary Type` berisi informasi jenis kejahatan yang terjadi di Chicago,
2. `Date` berisi tanggal terjadinya kejahatan di Chicago,
3. `Location Description` berisi informasi terkait tempat kejadian,
4. `Arrest` berisi informasi apakah ada penangkapan pelaku kejadian,
5. `Domestic` berisi tentang apakah kejahatan tersebut berkaitan dengan kekerasan rumah tangga,
6. `Beat` berisi informasi yang menunjukkan wilayah patroli kepolisian tempat kejadian kejahatan itu terjadi,
7. `Distric` berisi tentang distrik kepolisian di mana kejadian dilaporkan,
8. `Latitude` berisi tentang informasi koordinasi lokasi berupa garis lintang,
9. `Longitude` berisi tentang informasi koordinasi lokasi berupa garis bujur,
10. `Community Area` berisi kode wilayah komunitas atau area geografis tempat kejahatan itu terjadi,
11. `Hour` berisi informasi jam di mana kejadian dilaporkan terjadi, dalam format 24 jam (0-23),
12. `Day of Week` berisi informasi hari dalam seminggu (0-6), di mana 0 adalah Ahad, 1 adalah Senin, dan seterusnya.

### Deskripsi Statistik
   
|         | Date                       | Beat           | District       | Latitude       | Longitude      | Community Area | Hour           | Day of Week     |
|---------|----------------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| count   | 948424                      | 948424.000000  | 948424.000000  | 934039.000000  | 934039.000000  | 948423.000000  | 948424.000000  | 948424.000000  |
| mean    | 2019-11-19 18:37:12.548218  | 1143.738499    | 11.208713      | 41.842843      | -87.669994     | 36.872467      | 12.897611      | 3.012267       |
| min     | 2018-01-01 00:00:00         | 111.000000     | 1.000000       | 36.619446      | -91.685656     | 1.000000       | 0.000000       | 0.000000       |
| 25%     | 2018-11-16 18:09:45         | 611.000000     | 6.000000       | 41.767482      | -87.712670     | 23.000000      | 9.000000       | 1.000000       |
| 50%     | 2019-10-10 16:33:30         | 1024.000000    | 10.000000      | 41.861559      | -87.663463     | 32.000000      | 14.000000      | 3.000000       |
| 75%     | 2020-11-05 18:20:00         | 1722.000000    | 17.000000      | 41.904861      | -87.627609     | 54.000000      | 18.000000      | 5.000000       |
| max     | 2021-12-31 23:59:00         | 2535.000000    | 31.000000      | 42.022671      | -87.524529     | 77.000000      | 23.000000      | 6.000000       |
| std     | NaN                         | 696.565705     | 6.959525       | 0.086968       | 0.059382       | 21.507729      | 6.647212       | 1.999065       |

### Penangan Missing Values dan Duplikasi
1. Missing Values
   
| Column               | Missing Values |
|----------------------|----------------|
| Primary Type          | 0              |
| Date                 | 0              |
| Location Description  | 4319           |
| Arrest               | 0              |
| Domestic             | 0              |
| Beat                 | 0              |
| District             | 0              |
| Latitude             | 14385          |
| Longitude            | 14385          |
| Community Area       | 1              |
| Hour                 | 0              |
| Day of Week          | 0              |

Tabel diatas menunjukkan terdapat missing values pada kolom `Location Description`, `Latitude`, `Longitude` dan `Community Area`. Maka dari itu akan dilakukan penghapusan missing values.

2. Duplikasi
   
Untuk melihat berapa baris yang terdapat kesamaan identik sekaligus dengan menghapusnya dan menampilkan kembali jumlah baris setelah dilakukan penghapusan duplikasi dengan menggunakan script berikut:
```python
# Melihat jumlah baris yang terduplikasi
before = df.shape[0]
print(f"Jumlah baris sebelum menghapus duplicates: {before}")

# Menghapus baris yang terduplikasi secara identik
df = df.drop_duplicates()

# Menampilkan jumlah baris setelah menghapus duplikasi
after = df.shape[0]
print(f"Jumlah baris setelah menghapus duplicates: {after}")
```
Dari script diatas terdapat `930769` duplikasi identik, dan setelah penghapusan duplikasi menghasilkan total baris senilai `929122`

### Penanganan Outliers
Tahapan ini untuk melihat informasi berupa outliers pada dataset.

![beforeoutliers](https://github.com/user-attachments/assets/69e37e69-785f-4fe8-8aa0-4f4d0409ff96)

Dari gambar diatas terdapat *outliers* pada `Longitude` dan `Latitude`, maka dari itu akan dilakukan penghapusan *outliers* dengan menggunakan metode IQR, dengan formula IQR sebagai berikut:

   $IQR=Q_3-Q_1$
   
   Kemudian membuat batas bawah dan batas atas untuk mencakup *outliers* dengan menggunakan,
   
   $BatasBawah=Q_1-1.5*IQR$
   
   $BatasAtas=Q_3-1.5*IQR$


Setelah dilakukan penghapusan *outliers*, dilakukan pengecekan kembali untuk melihat *outliers*, dan jumlah baris menurun menjadi `924193` baris.

![afteroutliers](https://github.com/user-attachments/assets/106f6dad-f03c-4589-bb28-4bfa736f40f7)

Gambar diatas menunjukkan tidak ada *outliers* pada `Latitude`, pada `Longitude` terdapat masih terdapat *outliers*, tetapi outliers tersebut masih di batas aman.


### Univariate Analysis
Tahapan ini ditujukan unutk melihat informasi dari fitur numerik pada dataset, berikut informasi dalam bentuk histogram:

![univariate](https://github.com/user-attachments/assets/1ee81328-14de-4070-9f20-7182ae8b0cd0)

Dari hasil histogram diatas dapat diuraikan sebagai berikut:
1. `Date`: 
   - Distribusi kejadian kriminalitas berdasarkan tanggal. Kejahatan cenderung merata sepanjang tahun, meskipun ada beberapa variasi pada periode tertentu. Ada penurunan aktivitas kejahatan pada pertengahan 2020 yang mungkin terkait dengan pembatasan aktivitas akibat pandemi COVID-19.

2. `Beat`: 
   - Kode patroli polisi (Beat) memiliki distribusi yang bervariasi, dengan beberapa beat memiliki jumlah kejadian yang jauh lebih banyak dibandingkan yang lain. Ini menunjukkan bahwa beberapa area patroli memiliki tingkat kejahatan yang lebih tinggi.

3. `District`: 
   - Distribusi kejadian kriminal berdasarkan distrik polisi. Beberapa distrik terlihat memiliki jumlah kejahatan yang jauh lebih tinggi dibandingkan distrik lainnya, menunjukkan variasi tingkat kejahatan berdasarkan lokasi.

4. `Latitude`: 
   - Distribusi kejadian kejahatan berdasarkan garis lintang. Terlihat bahwa ada konsentrasi kejahatan di beberapa rentang lintang tertentu, yang mungkin menunjukkan area geografis dengan tingkat kejahatan yang lebih tinggi.

5. `Longitude`: 
   - Distribusi kejadian kejahatan berdasarkan garis bujur. Pola distribusi ini menunjukkan konsentrasi kejadian kejahatan di wilayah geografis tertentu berdasarkan bujur.

6. `Community Area`: 
   - Distribusi kejadian berdasarkan kode area komunitas. Terlihat ada beberapa area komunitas yang memiliki kejadian kejahatan jauh lebih tinggi, yang mungkin mencerminkan area dengan tingkat aktivitas kejahatan yang lebih padat.

7. `Hour`: 
   - Distribusi kejadian kejahatan berdasarkan jam kejadian. Puncak kejadian kejahatan cenderung terjadi pada waktu sore hingga malam hari (sekitar jam 12 hingga 22), dengan beberapa penurunan aktivitas kejahatan selama jam-jam pagi.

8. `Day of Week`: 
   - Distribusi kejadian kejahatan berdasarkan hari dalam seminggu. Secara umum, tingkat kejahatan terlihat cukup merata sepanjang minggu, tanpa variasi signifikan antara hari-hari yang berbeda.

### Multivariate Analysis

Pada Multivariate Analysis akan menunjukkan informasi berupa 1 variabel berkaitan dengan variabel lain, beberapa informasi yang dapat ditujukan:

1. Fitur Kategori
   Pada fitur kategori ini diberikan informasi sebagai berikut:
   - Distribusi jenis kejahatan berdasarkan jam kejadian
     ![Distribusi Jenis Kejahatan berdasarkan Jam Kejadian](https://github.com/user-attachments/assets/0fa010bb-de85-4e01-91c7-78a8f88be688)
   - Distribusi jenis kejahatan berdasarkan hari dalam seminggu
     ![Distribusi Jenis Kejahatan Berdasarkan Hari dalam Seminggu](https://github.com/user-attachments/assets/b6d4a455-4585-4e96-8183-7fd1aae17d06)
   - Rata-rata jam kejadian per jenis kejahatan
     ![Rata-rata Jam Kejadian per Jenis Kejahatan](https://github.com/user-attachments/assets/e7b4353c-0c0d-4875-9370-27ba6e9e96b6)
   - Jenis kejahatan berdasarkan Hour, Arrest, dan Domestic
     ![Jenis Kejahatan Berdasarkan Jam, Arrest, dan Domestic](https://github.com/user-attachments/assets/c884ca1c-7f1c-41a5-ab23-7e36e743f5c0)
   - Distribusi lokasi kejahatan berdasarkan Community Area
     ![Distribusi Lokasi Kejahatan Berdasarkan Community Area](https://github.com/user-attachments/assets/130b61e4-130c-4c43-a054-6c7345e23f41)
   - Distribusi jenis kejahatan berdasarkan Lokasi
     ![Distribusi Jenis Kejahatan Berdasarkan Lokasi](https://github.com/user-attachments/assets/a5fae387-6685-4a9e-a8f9-69bd1c1fb4be)
   - Distribusi lokasi kejahatan berdasarkan waktu (Hour)
     ![Distribusi Lokasi Kejahatan Berdasarkan Waktu (Hour)](https://github.com/user-attachments/assets/a98526dd-4037-4183-a452-700a4fc9b759)
2. Fitur Numerik
   ![multivariate](https://github.com/user-attachments/assets/be5824fd-0d40-49d9-845a-de36c5907adf)
4. Matriks Korelasi<br>
   Melakukan visualisasi pada fitur numerik untuk mengetahui korelasi antar fitur
   ![Matriks Korelasi untuk Fitur Numerik](https://github.com/user-attachments/assets/172643ee-907c-48cd-b21f-c937eeae0eda)
    Melihat fitur `Hour` dan `Day of Week` korelasinya lemah, maka kedua fitur tersebut akan didrop.

# Data Preperation<br>
Pada Data Preperation ini dilakukan persiapan data untuk dapat dimasukkan ke model, proses ini terdapat dua tahap yaitu:
1. Encoding
2. Splitting Data
3. Scaling
## Encoding
Pada proses ini akan dilakukan perubahan pada fitur kategori diubah menjadi fitu numerik untuk memudahkan proses modeling.
## Splitting Data
Pada proses ini dilakukan pembagian data pada dataset. Rasio pembagian data ini adalah 90:10, 90 untuk data latih, dan 10 untuk data uji. pada tahapan ini juga pada data latih akan membuang fitur 
# Modeling
# Evaluation




















Dikarenakan kriteria dari Dicoding untuk menggunakan data kuantitatif, di proyek ini akan menghapus beberapa fitur yang tidak digunakan dan hanya akan menggunakan fitur dengan tipe data numeric pada dataset tersebut, diantaranya:
- 
  Fitur ini berisi tentang informasi koordinasi lokasi berupa garis lintang
- 
  Fitur ini 
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
