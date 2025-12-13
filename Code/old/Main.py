from Code.old.AI_Blackjack import AI_Blackjack
from Code.cupy.AI_Blackjack_CUDA_cupy import AI_Blackjack_CUDA_cupy
import time

if __name__ == "__main__":
    episodes = 10000000


    ai = AI_Blackjack()
    start = time.time()
    ai.train(episodes=episodes)
    end = time.time()
    ai.evaluate_policy()
    print(f"Czas wykonania: {end - start:.4f} sekundy")
    ai = AI_Blackjack_CUDA_cupy()
    start = time.time()
    ai.train(episodes=episodes)
    end = time.time()
    ai.evaluate_policy()
    print(f"Czas wykonania: {end - start:.4f} sekundy")
