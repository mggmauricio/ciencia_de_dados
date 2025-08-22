# 3) Escreva uma função que receba um número  inteiro "a" e retorne uma lista com todos os divisores de "a". Exemplo: os divisores de 12 são 12,6,4,3,2 e 1. Exemplo de uso: print(divisores(x))
def divisores(a):
    if a <= 0:
        return []  # Retorna lista vazia para números não positivos
    return [i for i in range(1, a + 1) if a % i == 0]

print("Divisores de 12:", divisores(12))
print("Divisores de 15:", divisores(15))