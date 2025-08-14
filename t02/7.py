# Encontrar a soma dos números inteiros a partir de 1 até N, com incrementos de 3 em 3. Exemplo: 1+4+7+10+13+16+19, para N=19.
N = int(input("Digite um número: "))
sum = 0
for i in range(1, N+1, 3):
    sum += i
print(f"A soma dos números inteiros a partir de 1 até {N}, com incrementos de 3 em 3 é: {sum}")
