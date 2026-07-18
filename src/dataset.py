from pathlib import Path

import wfdb


class ECGDataset:

    def __init__(self, data_dir):

        self.data_dir = Path(data_dir)

    def load_record(self, record_name):

        record_path = self.data_dir / record_name

        return wfdb.rdrecord(str(record_path))

    def load_annotation(self, record_name):

        record_path = self.data_dir / record_name

        return wfdb.rdann(str(record_path), "atr")
    
# در مراحل بعد قابلیت‌هایی مانند 
# فهرست کردن رکوردها، اعتبارسنجی فایل‌ها
#  بارگذاری دسته‌ای و تبدیل داده‌ها به
#  قالب مناسب مدل به آن اضافه خواهیم شد    