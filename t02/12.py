#  Faça um programa que leia uma senha, até que a senha esteja correta.
correct_password = "123456"
password = ""
while password != correct_password:
    password = input("Digite a senha: ")
    if password == correct_password:
        break;
    else:
        print("Senha incorreta! Tente novamente.")
print("Senha correta!")
