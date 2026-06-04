# Menggunakan image Python yang ringan
FROM python:3.12-slim

# Menetapkan direktori kerja di dalam container
WORKDIR /app

# Menyalin file requirements ke dalam container
COPY requirements.txt .

# Menginstall dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh isi proyek ke dalam container
COPY . .

# Menentukan command untuk menjalankan model (opsional, tergantung kebutuhan)
# CMD ["python", "Membangun_model/modelling.py"]
