#  A partir de um valor dado em segundos, calcular o valor correspondente em horas, minutos e segundos. 


segundos = int(input("Digite um valor em segundos: "))
horas = segundos // 3600
minutos = (segundos % 3600) // 60
segundos = segundos % 60
print(f"{horas}h {minutos}m {segundos}s")
