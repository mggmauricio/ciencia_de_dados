"""
Análise de Classificação Linear
Dataset de classificação binária com 2 atributos
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# ============================================
# 1. CRIAR O DATASET
# ============================================
print("=== DATASET DE CLASSIFICAÇÃO ===\n")

# Dados fornecidos
data = {
    'Data': ['X1', 'X2', 'X3', 'X4', 'X5'],
    'Attribute 1': [0.10, 0.60, 0.85, 0.35, 0.30],
    'Attribute 2': [0.50, 0.40, 0.12, 0.50, 0.80],
    'Class': ['A', 'A', 'B', 'B', 'A']
}

df = pd.DataFrame(data)
print("Dataset:")
print(df.to_string(index=False))
print()

# Preparar dados para o modelo
X = df[['Attribute 1', 'Attribute 2']].values
y = df['Class'].values

# Converter classes para numérico (A=0, B=1)
y_numeric = np.where(y == 'A', 0, 1)

print(f"Atributos (X):\n{X}")
print(f"\nClasses (y): {y}")
print(f"Classes numéricas: {y_numeric}\n")

# ============================================
# 2. PLOTAR O DATASET COM CORES DIFERENTES
# ============================================
print("=== PLOTAGEM DO DATASET ===\n")

plt.figure(figsize=(12, 5))

# Subplot 1: Dataset original
plt.subplot(1, 2, 1)
# Plotar classe A (azul)
mask_a = y == 'A'
plt.scatter(X[mask_a, 0], X[mask_a, 1], c='blue', s=100, marker='o', 
           label='Classe A', edgecolors='black', linewidths=2)
# Plotar classe B (vermelho)
mask_b = y == 'B'
plt.scatter(X[mask_b, 0], X[mask_b, 1], c='red', s=100, marker='s', 
           label='Classe B', edgecolors='black', linewidths=2)

# Adicionar labels para cada ponto
for i, (x1, x2, label) in enumerate(zip(X[:, 0], X[:, 1], df['Data'])):
    plt.annotate(label, (x1, x2), xytext=(5, 5), textcoords='offset points', fontsize=10)

plt.xlabel('Attribute 1', fontsize=12)
plt.ylabel('Attribute 2', fontsize=12)
plt.title('Dataset - Visualização das Classes', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(-0.1, 1.0)
plt.ylim(0.0, 0.9)

# ============================================
# 3. TREINAR CLASSIFICADOR LINEAR
# ============================================
print("=== TREINAMENTO DO CLASSIFICADOR LINEAR ===\n")



model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X, y_numeric)

print("Classificador Linear (Regressão Logística) treinado!")
print(f"Coeficientes: {model.coef_[0]}")
print(f"Intercepto: {model.intercept_[0]}\n")

# Fazer predições
y_pred = model.predict(X)
y_pred_proba = model.predict_proba(X)

print("Predições:")
for i, (true_class, pred_class, proba) in enumerate(zip(y, y_pred, y_pred_proba)):
    pred_label = 'A' if pred_class == 0 else 'B'
    print(f"{df['Data'][i]}: Classe Real={true_class}, Predita={pred_label}, "
          f"Probabilidade A={proba[0]:.3f}, B={proba[1]:.3f}")

# ============================================
# 4. PLOTAR O CLASSIFICADOR LINEAR
# ============================================
print("\n=== PLOTAGEM DO CLASSIFICADOR LINEAR ===\n")

# Subplot 2: Dataset com classificador
plt.subplot(1, 2, 2)

# Plotar pontos
plt.scatter(X[mask_a, 0], X[mask_a, 1], c='blue', s=100, marker='o', 
           label='Classe A', edgecolors='black', linewidths=2)
plt.scatter(X[mask_b, 0], X[mask_b, 1], c='red', s=100, marker='s', 
           label='Classe B', edgecolors='black', linewidths=2)

# Plotar fronteira de decisão
x_min, x_max = -0.1, 1.0
y_min, y_max = 0.0, 0.9
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotar região de decisão
plt.contourf(xx, yy, Z, alpha=0.3, colors=['lightblue', 'lightcoral'], levels=[0, 0.5, 1])
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)

# Plotar linha de decisão (fronteira)
# w0*x0 + w1*x1 + b = 0 => x1 = -(w0*x0 + b)/w1
w0, w1 = model.coef_[0]
b = model.intercept_[0]
x_line = np.linspace(x_min, x_max, 100)
y_line = -(w0 * x_line + b) / w1
plt.plot(x_line, y_line, 'k-', linewidth=2, label='Fronteira de Decisão')

# Adicionar labels
for i, (x1, x2, label) in enumerate(zip(X[:, 0], X[:, 1], df['Data'])):
    plt.annotate(label, (x1, x2), xytext=(5, 5), textcoords='offset points', fontsize=10)

plt.xlabel('Attribute 1', fontsize=12)
plt.ylabel('Attribute 2', fontsize=12)
plt.title('Dataset com Classificador Linear', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.tight_layout()
# Obter o diretório do script
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, 'classificacao_linear.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Gráfico salvo em '{output_path}'\n")
# plt.show()  # Comentado para não bloquear a execução
plt.close()

# ============================================
# 5. CALCULAR MÉTRICAS
# ============================================
print("=== MÉTRICAS DE AVALIAÇÃO ===\n")

# Calcular métricas
accuracy = accuracy_score(y_numeric, y_pred)
precision = precision_score(y_numeric, y_pred, average='binary')
recall = recall_score(y_numeric, y_pred, average='binary')
f1 = f1_score(y_numeric, y_pred, average='binary')

# Matriz de confusão
cm = confusion_matrix(y_numeric, y_pred)

print("Matriz de Confusão:")
print("                 Predito")
print("              Classe A  Classe B")
print(f"Real Classe A    {cm[0,0]:4d}     {cm[0,1]:4d}")
print(f"     Classe B    {cm[1,0]:4d}     {cm[1,1]:4d}")
print()

print("Métricas:")
print(f"  Acurácia (Accuracy):  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precisão (Precision): {precision:.4f} ({precision*100:.2f}%)")
print(f"  Recall (Sensibilidade): {recall:.4f} ({recall*100:.2f}%)")
print(f"  F1 Score:             {f1:.4f} ({f1*100:.2f}%)")
print()

# Explicação das métricas
print("Explicação das Métricas:")
print("  - Acurácia: Proporção de predições corretas sobre o total")
print("  - Precisão: Proporção de predições positivas que são realmente positivas")
print("  - Recall: Proporção de casos positivos que foram corretamente identificados")
print("  - F1 Score: Média harmônica entre Precisão e Recall")
print()

# Detalhamento por classe
print("Métricas por Classe:")
tn, fp, fn, tp = cm.ravel()
print(f"\nClasse A (Negativa):")
print(f"  Verdadeiros Negativos (TN): {tn}")
print(f"  Falsos Positivos (FP): {fp}")
print(f"  Especificidade: {tn/(tn+fp) if (tn+fp) > 0 else 0:.4f}")

print(f"\nClasse B (Positiva):")
print(f"  Verdadeiros Positivos (TP): {tp}")
print(f"  Falsos Negativos (FN): {fn}")
print(f"  Sensibilidade (Recall): {recall:.4f}")

print("\n" + "="*60)
print("Análise concluída!")
print("="*60)

