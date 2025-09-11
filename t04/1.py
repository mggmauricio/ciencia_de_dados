import csv
from datetime import datetime


def calculate_age(date: str) -> int:
    age = datetime.now().year - datetime.strptime(date, "%B %d, %Y").year   
    return age

def calculate_mean_age(data: list) -> float:
    return sum(calculate_age(row[2]) for row in data) / len(data)

with open("actors.csv", "r") as file:
    reader = csv.reader(file)
    header = next(reader)
    data = list(reader)
    mean_age = calculate_mean_age(data)
    print(mean_age)



print("A média de idade dos atores é: ", mean_age)