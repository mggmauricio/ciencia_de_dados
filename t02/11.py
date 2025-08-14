#  Calcular a expressão x2+y2/(x-y)2 a partir dos valores x e y dados.

def expression(x, y):
    if x == y:
        return "Erro: x e y devem ser diferentes"
    return (x**2 + y**2) / ((x-y)**2)

x = float(input("Digite o valor de x: "))
y = float(input("Digite o valor de y: "))
print(f"O valor da expressão é: {expression(x, y)}")