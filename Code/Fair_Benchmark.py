import time
import numpy as np
from AI_Blackjack_CPU import AI_Blackjack_CPU
from AI_Blackjack_CUDA_numba import AI_Blackjack_GPU
from AI_Blackjack_CUDA_cupy_simplified_blackjack import AI_Blackjack_CuPy

# Zakładam, że masz gotowe klasy:
# AI_Blackjack_CPU
# AI_Blackjack_GPU (Numba CUDA)
# AI_Blackjack_CuPy

EPISODES = 100000
BATCH_SIZE = 1024

def benchmark_and_show_policy(ai_class, backend_name="CPU"):
    print(f"=== Benchmark {backend_name} ===")
    ai = ai_class(epsilon=0.1)
    start = time.time()
    if backend_name == "CPU":
        ai.train(episodes=EPISODES)
    else:
        ai.train(episodes=EPISODES, batch_size=BATCH_SIZE)
    end = time.time()
    print(f"{backend_name} training time: {end - start:.2f} s\n")

    # Wypisz pełną politykę (akcje HIT/STAND) po treningu
    ai.evaluate_policy(print_n=True)
    return ai

if __name__ == "__main__":
    # CPU
    ai_cpu = benchmark_and_show_policy(AI_Blackjack_CPU, "CPU")

    # GPU Numba
    ai_gpu = benchmark_and_show_policy(AI_Blackjack_GPU, "GPU (Numba CUDA)")

    # GPU CuPy
    ai_cupy = benchmark_and_show_policy(AI_Blackjack_CuPy, "GPU (CuPy)")

