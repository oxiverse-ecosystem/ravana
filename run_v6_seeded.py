import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
seed = int(sys.argv[1])
np.random.seed(seed)
print(f"=== SEED: {seed} ===")
from experiment_triple_benchmark_v6 import run_benchmark
run_benchmark()
