import numpy as np

class BenchmarkResult:
    def __init__(self, name: str = 'accuarcy'):
        self._result: list[float] = []
        self._name = name

    def append(self, result: float):
        self._result.append(result)

    def get_max(self, last = None) -> float:
        if last is None:
            return max(self._result)
        else:
            last_list = self._result[-last:]
            return max(last_list)

    def get_mean(self, last = None) -> float:
        if last is None:
            return np.mean(self._result)
        else:
            last_list = self._result[-last:]
            return np.mean(last_list)

    def get_std(self, last = None) -> float:
        if last is None:
            return np.std(self._result)
        else:
            last_list = self._result[-last:]
            return np.std(last_list)
        
    
    def format(self, name: str = None, percent: bool = True, last = None) -> str:
        if name is None:
            name = self._name

        if last is None:
            last_str = 'all'
        else:
            last_str = f'({last})'
        
        if percent == True:
            return f'{name}{last_str}: {self.get_mean(last) * 100:.2f}% ± {self.get_std(last) * 100:.2f}% (max = {self.get_max(last) * 100:.2f}%)'
        else:
            return f'{name}{last_str}: {self.get_mean(last):.4f} ± {self.get_std(last):.4f} (max = {self.get_max(last):.4f})'
    
    def get_data(self) -> list[float]:
        return self._result
    
    def get(self, idx: int) -> float:
        return self._result[idx]