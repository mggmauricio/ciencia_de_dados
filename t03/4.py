# 4) Escreva uma função que testa se um número "a" dado é primo. Deve retornar verdadeiro apenas se "a" for primo, e falso caso contrário. Exemplo de uso: if (x>0 and primo(x)):
def primo(a):
    if a <= 1:
        return False  # Números menores ou iguais a 1 não são primos
    for i in range(2, int(a**0.5) + 1):
        if a % i == 0:
            return False  # Encontrou um divisor, não é primo
    return True  # Não encontrou divisores, é primo

print("5 é primo:", primo(5))
print("10 é primo:", primo(10))