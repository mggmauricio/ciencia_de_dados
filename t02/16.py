
#  Escreva um programa que receba dois números e um operador (+,-,* ou /) , e faça a operação matemática definida pelo sinal.

num1 = float(input("Digite o primeiro número: "))
num2 = float(input("Digite o segundo número: "))
operator = input("Digite o operador (+,-,* ou /): ")

if operator == "+":
    print(num1 + num2)
elif operator == "-":
    print(num1 - num2)
elif operator == "*":
    print(num1 * num2)
elif operator == "/":
    print(num1 / num2)
else:
    print("Operador inválido!")
