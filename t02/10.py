#  Calcule a idade aproximada de uma pessoa a partir do ano de seu nascimento.
from datetime import datetime

ano_nascimento = int(input("Digite o ano de nascimento: "))
idade = datetime.now().year - ano_nascimento
print(f"A idade aproximada é: {idade} anos")