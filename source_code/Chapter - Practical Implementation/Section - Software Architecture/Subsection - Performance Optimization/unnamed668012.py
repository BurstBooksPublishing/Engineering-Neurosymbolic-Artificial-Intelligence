from functools import lru_cache

# A recursive function to compute Fibonacci numbers as an example of symbolic reasoning
@lru_cache(maxsize=128)  # Cache the most recent 128 calls
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Calculate Fibonacci number
print(fibonacci(10))