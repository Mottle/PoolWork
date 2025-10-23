import numpy as np

class BenchmarkResult:
    def __init__(self):
        self._result: list[float] = []
        self._best: float = -1.0

    def append(self, result: float):
        self._result.append(result)
        self._best = max(self._best, result)

    def get_max(self) -> float:
        return self._best

    def get_mean(self) -> float:
        return np.mean(self._result)

    def get_std(self) -> float:
        return np.std(self._result)
    
    def format(self, name: str = 'acc') -> str:
        return f'{name} = {self.get_mean() * 100:.2f} ± {self.get_std() * 100:.2f} (best = {self.get_max() * 100:.2f})'
    
    def get_data(self) -> list[float]:
        return self._result
    
    def get(self, idx: int) -> float:
        return self._result[idx]
    
    def merge(self, other: 'BenchmarkResult') -> 'BenchmarkResult':
        self._result += other.get_data()
        self._best = max(self._best, other.get_max())
        return self