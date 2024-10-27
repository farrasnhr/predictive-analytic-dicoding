# Laporan Proyek Machine Learning - Farras Nur Haidar Ramadhan

## Domain Proyek

### Latar Belakang

Pada tahun 2021, Chicago mengalami salah satu tahun paling mematikan dalam lebih dari dua dekade, dengan **lebih dari 800 pembunuhan** yang tercatat. Peningkatan ini tidak hanya menjadikan tahun tersebut sebagai salah satu yang paling berbahaya dalam sejarah kota, tetapi juga menunjukkan peningkatan signifikan dalam kekerasan bersenjata. **Chicago Police Department** melaporkan bahwa insiden penembakan juga mencapai **3.561 kasus**, yang menunjukkan kenaikan lebih dari **300 insiden** dibandingkan tahun 2020 dan lebih dari **1.400 insiden** dibandingkan dengan tahun 2019 [^1][^2].

Kekerasan bersenjata ini sebagian besar dipicu oleh konflik antar kelompok atau yang disebut dengan geng, yang telah menjadi masalah utama di kota ini. Sebagai kota terbesar ketiga di Negeri Paman Sam, Chicago mencatat lebih banyak pembunuhan daripada kota-kota besar lainnya, seperti New York dan Los Angeles, yang masing-masing mencatat setidaknya **300 pembunuhan lebih sedikit** pada tahun yang sama [^3][^4]. Lonjakan kekerasan tersebut menyoroti tantangan besar yang dihadapi oleh penegak hukum dalam menjaga keamanan umum, meskipun ada upaya untuk mengurangi senjata ilegal di jalanan dan meningkatkan jumlah petugas investigasi [^5][^6].

Berdasarkan paparan latar belakang diatas, dapat dibuat model machine learning untuk memprediksi wilayah dengan sering terjadinya kejahatan yang ada di kota Chicago, dengan memanfaatkan data historis kriminal, seperti kejadian kriminal, lokasi kejadian dan lain-lain.

Model yang telah dibuat ini diharapkan untuk dapat digunakan sebagai langkah antisipasi dalam pengamanan lebih ketat oleh pihak berwajib, seperti memperkuat patroli di area-area berisiko tinggi dan rentan terhadap kejadian kekerasan.

[^1]: [WTTW News - 2021 Ends as Chicago’s Deadliest Year](https://news.wttw.com/2022/01/02/2021-ends-chicago-s-deadliest-year-quarter-century)
[^2]: [Chicago CRED 2021 Annual Report](https://www.chicagocred.org/blog/2021-annual-report/)
[^3]: [Chicago Journal - 2021 ends as Chicago's deadliest year](https://www.chicagojournal.com/2021-ends-as-chicagos-deadliest-year-in-a-quarter-century/)
[^4]: [WTTW News](https://news.wttw.com/2022/01/02/2021-ends-chicago-s-deadliest-year-quarter-century)
[^5]: [Chicago CRED](https://www.chicagocred.org/blog/2021-annual-report/)
[^6]: [Chicago Police Department Report](https://home.chicagopolice.org/statistics-data/crime-statistics/)

## Business Understanding
Dari penjelasan latar belakang di atas, dapat dibuat rumusan masalah sebagai berikut:

### Problem Statements
- Bagaimana menyiapkan data yang diperlukan untuk membuat model machine learning?
- Bagaimana cara membuat model machine learning untuk kebutuhan klasifikasi jenis kejahatan?
  
### Goals
Berdasarkan rumusan masalah sebelumnya, dapat dibuatkan tujuan laporan sebagai berikut
- Melakukan tahapan persiapan data, agar data yang sudah disiapkan dapat dimasukkan ke dalam model,
- Membuat model machine learning untuk mengklasifikasikan wilayah kejahatan dengan tiga tingkatan yaitu, Sering Terjadi, Jarang Terjadi, dan Sangat Jarang Terjadi.
  
### Solution Statemnent
Berdasarkan dari tujuan, didapatkan beberapa solusi untuk menjawab rumusan masalah sebagai berikut
- Melakukan pembagian pada data menjadi data train data data test dengan rasio sebesar 70:30,
- Menargetkan `Location Description` sebagai fitur yang diklasifikasikan,
- Menggunakan dua algoritma model yaitu Naive Bayes dan K-Nearest Neighbor untuk membandingkan akurasi dari dua algoritma yang diujikan.
  
## *Library* Untuk Proyek
```python
# Library dasar untuk manipulasi data dan visualisasi
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Library untuk preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Library untuk pembagian data
from sklearn.model_selection import train_test_split

# Library untuk algoritma machine learning
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Library untuk evaluasi model
from sklearn.metrics import accuracy_score, confusion_matrix
```

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
   - Distribusi kejadian kejahatan berdasarkan hari dalam seminggu. Secara umum, tingkat kejahatan terlihat cukup merata sepanjang minggu.

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
    Melihat fitur `Hour` dan `Day of Week` korelasi kedua fitur tersebut  lemah dan juga fitur `Date` hanya digunakan untuk proses EDA, maka ketiga fitur tersebut akan didrop.

# Data Preperation<br>
Pada Data Preperation ini dilakukan persiapan data untuk dapat dimasukkan ke model, proses ini terdapat tiga tahap yaitu:
1. Encoding
2. Splitting Data
3. Scaling
## Encoding
Pada proses ini akan dilakukan perubahan pada fitur kategori seperti `Primary Type`, dan `Location Description` diubah menjadi fitur numerik, fitur lain seperti `Arrest` dan `Domestic` dari kategori `boolean` berupa teks True / False akan diubah menjadi numerik (0 atau 1). Proses Encoding ini menggunakan LabelEncoder dari *library* SKLearn.
```python
# Menggunakan LabelEncoder untuk fitur 'Primary Type'
df_encoded['Primary Type'] = le.fit_transform(df['Primary Type'])
df_encoded['Location Description'] = le.fit_transform(df['Location Description'])

# Mengubah kolom boolean menjadi 0 atau 1
df_encoded['Arrest'] = df_encoded['Arrest'].astype(int)
df_encoded['Domestic'] = df_encoded['Domestic'].astype(int)
```

## Splitting Data
Sebelum data di bagi, dilakukan pemisahan fitur dan target dari dataset. Semua kolom kecuali `Community Area` dipilih sebagai fitur, sementara kolom `Community Area` dijadikan target yang ingin diprediksi atau klasifikasi. setelah pemisahan fitur dan target, dilakukan pembagian data pada dataset. Rasio pembagian data ini adalah 70:30, 70% untuk data latih, dan 30% untuk data uji. Pada tahapan ini akan memisahkan fitur dan target dalam dataset.

```python
# Memisahkan fitur dan target
X = df_encoded.drop(columns=['Community Area'])  # Semua kolom kecuali 'Community Area' sebagai fitur
y = df_encoded['Community Area']  # Kolom 'Community Area' sebagai target

# Membagi data menjadi 70% untuk training dan 30% untuk testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
## Scaling

Setelah data dibagi, langkah selanjutnya adalah melakukan proses scaling pada fitur-fitur dalam data yang telah dibagi. Scaling ini bertujuan untuk menyelaraskan rentang nilai setiap fitur, sehingga tidak ada fitur yang mendominasi atau mempengaruhi model secara berlebihan. Dengan menggunakan metode MinMax Scaling dari *library* SKLearn, setiap nilai fitur akan dipetakan dalam rentang 0 hingga 1. Scaling dilakukan pada data latih dan data uji.

```python
# Inisialisasi MinMaxScaler
scaler = MinMaxScaler()

# Melakukan scaling pada data latih
X_train_scaled = scaler.fit_transform(X_train)

# Melakukan scaling pada data uji
X_test_scaled = scaler.transform(X_test)
```

# Modeling

Pada tahapan ini setelah data dibagi dan di*scaling*, dilakukan pembangunan model, model yang dipilih ialah KNN (K-Nearest Neighbor) dan AdaBoost untuk dibandingkan performanya, setiap model akan dilatih menggunakan data latih dan setelah model dibangun akan dilakukan pengujian dengan data uji untuk melihat performa kemampuan model dalam klasifikasi.

## K-Nearest Neighbor (KNN)
K-Nearest Neighbor mengklasifikasikan objek berdasarkan “kedekatan” objek tersebut dengan “mayoritas” tetangganya, K-NN pada umumnya berjalan dengan baik untuk diimplementasikan pada dataset yang berukuran sangat besar. K-Nearest Neighbors (KNN) sederhana dan mudah diimplementasikan, serta tidak memerlukan proses pelatihan. KNN bekerja baik pada data teratur, tetapi komputasinya mahal pada dataset besar karena menghitung jarak untuk setiap titik. Selain itu, KNN sensitif terhadap perbedaan skala dan outliers serta rentan terhadap dimensi tinggi [^7][^8]. Berikut perhitungan algoritma KNN:
1. Menetapkan nilai k.
2. Melakukan perhitungan jarak antara data uji dengan data latih dengan menggunakan rumus *Euclidean Distance*.
   
$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

4. Mengurutkan data latih dengan jarak terkecil.
5. Menetapkan kelas, di mana kelas yang ditentukan merupakan kelas dengan jumlah nilai pada k terbanyak pada data uji.<br>

Pada pembangunan model KNN, parameter yang digunakan yaitu `n_neighbors = 5` dan `distance metric = Euclidean`.
```python
# Inisialisasi model KNN dengan `n_neighbors` dan `metric`.
knn_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
```
Setelah inisialisasi model, model akan dilatih dengan data latih.
```python
knn_model.fit(X_train_scaled, y_train)
```
[^7]: [B. Purnama, S, Si., MT, Pengantar Machine Learning. Bandung: Informatika, 2019.]()
[^8]: [J. Suntoro, Data Mining: Algoritma dan Implementasi dengan Pemrograman PHP. Jakarta: Elex Media Komputindo, 2019.]()

## Naive Bayes
Naïve Bayes merupakan teknik klasifikasi yang mengandalkan Teorema Bayes. Metodologi kategorisasi ini menggunakan teknik probabilitas dan statistik yang awalnya diusulkan oleh seorang ilmuwan Inggris bernama Thomas Bayes. Naïve Bayes adalah metode klasifikasi yang menggunakan konsep probabilitas untuk membangun model prediksi klasifikasi. Dengan memanfaatkan data historis, model dapat menghasilkan perkiraan kejadian di masa depan. Pendekatan ini menghitung kemungkinan terjadinya suatu kejadian, dan dapat diubah jika tersedia lebih banyak data yang menguatkan[^9][^10].<br>
Pada algoritma Naive Bayes, model ini akan menggunakan parameter `nb_model = GaussianNB`.

```python
# Inisialisasi model Naive Bayes dengan parameter nb_model.
nb_model = GaussianNB()

# Melatih model pada data training
nb_model.fit(X_train_scaled, y_train)
```
[^9]: [D. M. S. Kurniawan, Pengenalan Machine Learning dengan Python. Elex Media Komputindo, 2022.]()
[^10]: [C. Davidson and Pilon, Bayesian Methods for Hackers. Boston: Addison 56 Wesley, 2015.]()

# Evaluation
Setelah model dilatih dengan data latih, dilakukan evaluasi menggunakan data uji guna melihat performa model dalam klasifikasi data uji. Evaluasi pada proyek ini menggunakan *Confusion Matrix* sebagai gambaran dari model memprediksi data uji. *Confusion Matrix* ini membandingkan data target prediksi 
dengan data target aktual. Nilai prediksi adalah hasil pemodelan machine learning, 
sedangkan data aktual adalah nilai sebenarnya yang dimiliki[^11].

|               | Prediksi Positif | Prediksi Negatif |
|---------------|------------------|------------------|
| **Aktual Positif** | True Positive (TP) | False Negative (FN) |
| **Aktual Negatif** | False Positive (FP) | True Negative (TN) |

Dari tabel diatas merupakan gambaran dari *Confusion Matrix*, di mana TP adalah nilai prediksi benar sesuai dengan nilai aktual benar, FP adalah nilai prediksi benar sesuai dengan nilai aktual salah, FN adalah nilai prediksi salah sesuai dengan nilai aktual benar, FP adalah nilai prediksi salah sesuai dengan nilai aktual salah. Dari gambaran *Confusion Matrix* dapat dilakukan pengukuran performa model yaitu akurasi[31]. Akurasi dapat digambarkan seberapa akurat machine learning dapat memprediksi nilai[^11]. 

$$Akurasi = \frac{TP + TN}{TP + TN + FP + FN}\$$<br>
di mana:
- **TP** = True Positive (prediksi benar untuk kelas positif)
- **TN** = True Negative (prediksi benar untuk kelas negatif)
- **FP** = False Positive (prediksi salah untuk kelas positif)
- **FN** = False Negative (prediksi salah untuk kelas negatif)

Dari pengukuran performa model sebelumnya dapat diuraikan untuk performa model yang telah dilatih:

```python
# Prediksi dan akurasi untuk Naive Bayes
nb_train_pred = nb_model.predict(X_train_scaled)
nb_test_pred = nb_model.predict(X_test_scaled)
nb_train_accuracy = accuracy_score(y_train, nb_train_pred)
nb_test_accuracy = accuracy_score(y_test, nb_test_pred)

# Prediksi dan akurasi untuk KNN
knn_train_pred = knn_model.predict(X_train_scaled)
knn_test_pred = knn_model.predict(X_test_scaled)
knn_train_accuracy = accuracy_score(y_train, knn_train_pred)
knn_test_accuracy = accuracy_score(y_test, knn_test_pred)
```

### Confusion Matrix Naive Bayes - Data Latih

|           | Predicted 0 | Predicted 1 | Predicted 2 |
|-----------|-------------|-------------|-------------|
| **Actual 0** | 25319       | 123844      | 44          |
| **Actual 1** | 9225        | 43363       | 903         |
| **Actual 2** | 1042        | 43804       | 4665        |

### Confusion Matrix Naive Bayes - Data Uji

|           | Predicted 0 | Predicted 1 | Predicted 2 |
|-----------|-------------|-------------|-------------|
| **Actual 0** | 11000       | 55206       | 0           |
| **Actual 1** | 3933        | 185481      | 360         |
| **Actual 2** | 481         | 18881       | 1996        |

### Confusion Matrix KNN - Data Latih

|           | Predicted 0 | Predicted 1 | Predicted 2 |
|-----------|-------------|-------------|-------------|
| **Actual 0** | 151481      | 2084        | 98          |
| **Actual 1** | 2461        | 440008      | 1292        |
| **Actual 2** | 163         | 2621        | 46727       |

### Confusion Matrix KNN - Data Uji

|           | Predicted 0 | Predicted 1 | Predicted 2 |
|-----------|-------------|-------------|-------------|
| **Actual 0** | 64630       | 1484        | 92          |
| **Actual 1** | 1774        | 186885      | 1035        |
| **Actual 2** | 121         | 1771        | 19466       |

Dari hasil tersebut dapat menggunakan formula untuk mendapatkan nilai akurasi.

```python
# Membuat DataFrame dengan akurasi model
model_accuracies = {
    'Model': ['Naive Bayes', 'KNN'],
    'Data Latih': [nb_train_accuracy, nb_test_accuracy],
    'Data Uji': [knn_train_accuracy, knn_test_accuracy]
}
```

| Model        | Data Latih | Data Uji |
|--------------|------------|----------|
| Naive Bayes  | 0.716636   | 0.986523 |
| KNN          | 0.715568   | 0.977360 |

Dan dapat divisualisasikan sebagai berikut:

![akurasi](https://github.com/user-attachments/assets/edbf92f8-8f51-45cf-a46e-c47e1fba5690)<br>

Dari hasil perhitungan akurasi model, terlihat bahwa model Naive Bayes dan KNN menunjukkan performa yang berbeda pada data latih dan data uji. Model Naive Bayes memiliki akurasi sebesar 0.7166 pada data latih dan 0.9865 pada data uji. Sementara itu, model KNN menunjukkan akurasi 0.7156 pada data latih dan 0.9774 pada data uji.

Model Naive Bayes dan KNN sama-sama menunjukkan akurasi tinggi pada data uji, yang mengindikasikan bahwa keduanya berhasil menangkap pola-pola penting dalam data, meskipun terdapat sedikit perbedaan pada akurasi data latih. Akurasi yang tinggi pada data uji ini menunjukkan bahwa model mampu membedakan area dengan kategori yang berbeda misalnya, area dengan kategori "Jarang Terjadi," "Sering Terjadi," dan "Sangat Jarang Terjadi" – dengan baik, berdasarkan pola kejadian di area tersebut.

Dengan hasil ini dapat menyimpulkan bahwa baik Naive Bayes maupun KNN adalah model yang cocok untuk membantu prediksi area dengan tingkat kejadian tertentu dalam area yang diklasifikasikan.


[^11]: [I. Saputra and D. A. Kristiyanti, Machine Learning Untuk Pemula. Bandung: Informatika, 2022.]()

