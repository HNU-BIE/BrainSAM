from tqdm import tqdm as std_tqdm

class QtTqdm(std_tqdm):
    def __init__(self, *args,callback=None ,**kwargs):
        super().__init__(*args,**kwargs)
        self._cb = callback
    def update(self, n = 1):
        super().update(n)
        if self._cb and self.total:
            self._cb(int(self.n / self.total*100))