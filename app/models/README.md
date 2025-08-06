# 📁 `models/`

Folder ini berisi **model-model terbaik** dari setiap tahapan modular dalam sistem klasifikasi tipe tokoh cerita rakyat.  
Karena ukuran file yang besar, seluruh model tidak disertakan di GitHub dan disediakan dalam bentuk file `.zip` yang bisa diunduh melalui link berikut:

🔗 [Unduh model ZIP via OneDrive](https://1drv.ms/f/c/2ee4829be2b80f3f/ErLrXj229mhNlHnNhnSzGVEBv4RwPoXU1vq8UCU0BUMs4Q?e=YOWC2l)

---

### 📦 Langkah Instalasi Model

1. **Unduh** file `.zip` dari link OneDrive di atas.
2. **Ekstrak** isi file ZIP ke folder `models/` di root project agar struktur folder sesuai.
    #### 🖥️ Command Line (Linux/Mac/WSL)
    ```bash
    unzip model_TA_Rayssa_Ravelia.zip -d models/
    ```

    #### 🖥️ Command Line (Windows PowerShell)

    ```powershell
    Expand-Archive -Path .\model_TA_Rayssa_Ravelia.zip -DestinationPath .\models\
    ```

    > 💡 Pastikan nama file `.zip` sesuai. Jika berbeda, sesuaikan pada perintah di atas.

---

### 📁 Struktur Folder yang Diharapkan

Setelah diekstrak, struktur folder `models/` akan menjadi seperti berikut:

```
models/
├── ner_model/
│   ├── config.json
│   └── ... (file model NER)
├── random_forest_normalized/
│   ├── best_model.pkl
│   ├── best_params.txt
│   └── ... (file model random forest normalized)
├── V4_CahyaBERT/
│   ├── best_fold_1
│   ├── best_fold_2
│   └── ... (file model V4 CahyaBERT)
```

> ⚠️ **Jangan ubah nama folder atau file** agar aplikasi Streamlit dapat berjalan tanpa error.

---

Setelah itu, kamu bisa langsung menjalankan aplikasi demo menggunakan:

```bash
streamlit run app/demo_app/app.py
```