#  Dado o dia de nascimento de uma pessoa, calcular o número total de dias vividos até a data de hoje.
from datetime import datetime

nascimento = datetime.strptime(input("Digite o dia de nascimento: "), "%d/%m/%Y")
hoje = datetime.now()
diferenca = hoje - nascimento
dias = diferenca.days
print(f"O número total de dias vividos até a data de hoje é: {dias}")
