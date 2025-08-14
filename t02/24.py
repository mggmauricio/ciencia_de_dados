#  A partir de um vetor com 20 valores inteiros, o qual deve ser fornecido como entrada, verificar: (a) se todos os valores pertencem ao intervalo fechado [0,100] e, em caso afirmativo: (b) qual o valor mais frequente (moda).

import numpy as np

vetor = np.array([1, 2, 3, 4, 5, 6, 7, 8, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

if np.all((vetor >= 0) & (vetor <= 100)):
    moda = np.argmax(np.bincount(vetor))
    print(f"O valor mais frequente (moda) é: {moda}")
else:
    print("Todos os valores não pertencem ao intervalo fechado [0,100]")
