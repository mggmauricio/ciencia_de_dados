#  Mostrar na tela a lista dos primeiros 1000 números primos.

primes = []

def is_prime(number):
    if number <= 1:
        return False
    for i in range(2, int(number**0.5) + 1):
        if number % i == 0:
            return False
    return True

num = 0
while len(primes) < 1000:
    if is_prime(num):
        primes.append(num)
    num += 1

print(primes)