# 1) Escreva uma função que calcule o valor médio de uma lista de números.
def soma(lista):
    return sum(lista)
def media(lista):
    return soma(lista) / len(lista) if lista else 0

print("Média de [1, 2, 3, 4, 5]:", media([1, 2, 3, 4, 5]))