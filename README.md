# Teknofest2023
### Türkçe Doğal Dil İşleme Yarışması

### Takım: Data Hackers

<img src='data/img/img.jpg' width='450'>

---
### Gereksinimler
Bilgisayarınızda python3 yüklü olmalıdır. Python3'ü yüklemek için [linkteki](https://www.python.org/downloads/) adımları takip edebilirsiniz.

### Ortam Kurulumu
Projeyi bir problem yaşamadan çalıştırabilmek için isterseniz sanal ortam kullanabilir, isterseniz de globalde yüklediğiniz python ile çalışabilirsiniz.
Sanal ortam sağlayıcılarına örnek olarak conda verilebilir. Conda indirmek için [bu linkteki](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) adımları uygulayabilirsiniz. 
Conda ile virtual environment oluşturmak için aşağıdaki bash komutlarını çalıştırabilirsiniz.
```bash
conda conda create --name teknofest python=3.9
conda activate teknofest
```

Sanal ortamınızı kurduktan sonra requierements.txt dosyasını kullanarak kütüphanelerin uygun versiyonlarını kurabilirsiniz.
```bash
pip install -r requirements.txt
```
### Uygulamanın Çalıştırılması
Gradio servisinin çalıştırılması için run.py dosyasını çalıştırmanız gerekmektedir. Dosyayı çalıştırdıktan sonra uygulama infrence haline gelecektir. Tahminlerinizi yapabilirsiniz.
```bash
python run.py
```

### Analiz ve model denemesi dosyaları
`/notebook` klasörü altındaki notebook dosyalarını:
```bash
jupyter notebook
```
şeklinde çalıştırarak kullanabilirsiniz.
