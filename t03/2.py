# 2) Escreva uma função que receba dois números inteiros "a","b" e verifique se "a" é divisível por "b". Retornar verdadeiro caso sejam divisíveis, e falso em caso contrário. 
def divisivel(a, b):
    if b == 0:
        return False  # Evita divisão por zero
    return a % b == 0

print("10 é divisível por 2:", divisivel(10, 2))
print("10 é divisível por 3:", divisivel(10, 3))