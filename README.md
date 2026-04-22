# Görüntü İşleme Projesi

Bu proje, NumPy kullanarak manuel piksel seviyesinde görüntü işleme işlemleri gerçekleştiren bir FastAPI web uygulamasıdır.

## Kurulum ve Çalıştırma

Aşağıdaki adımları sırasıyla terminale (PowerShell veya CMD) yazarak projeyi hazır hale getirebilirsiniz.

### 1. Kütüphaneleri Yükleme
Öncelikle gerekli tüm araçları yüklemek için şu komutu çalıştırın:
```bash
pip install -r requirements.txt
```

### 2. Sanal Ortamı (venv) Aktifleştirme
Windows kullanıyorsanız, projeyi çalıştırmadan önce sanal ortamı aktif etmelisiniz:
```powershell
.\venv\Scripts\activate
```

### 3. Web Uygulamasını Başlatma
Her şey hazır olduğunda sunucuyu şu komutla başlatın:
```bash
uvicorn main:app --reload
```