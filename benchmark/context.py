from pathlib import Path
import sys

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
# print(sys.path)