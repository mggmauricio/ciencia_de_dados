"""
Análise de Regressão Linear
Relação entre idade do motorista e distância de visão
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

print("=" * 70)
print("ANÁLISE DE REGRESSÃO LINEAR")
print("Idade do Motorista vs Distância de Visão")
print("=" * 70)

# ============================================
# 1. CRIAR DATASET
# ============================================
print("\n1. DADOS")
print("-" * 70)

# Dados fornecidos
idade = np.array([20, 32, 41, 49, 66])  # Variável independente (x)
distancia = np.array([590, 410, 460, 380, 350])  # Variável dependente (y)

# Criar DataFrame
df = pd.DataFrame({
    'Idade (anos)': idade,
    'Distância (m)': distancia
})

print(df.to_string(index=False))
print(f"\nTotal de observações: {len(idade)}")

# ============================================
# 2. CALCULAR COEFICIENTE DE CORRELAÇÃO DE PEARSON
# ============================================
print("\n" + "=" * 70)
print("2. COEFICIENTE DE CORRELAÇÃO LINEAR DE PEARSON")
print("=" * 70)

# Método 1: Usando numpy
correlacao_numpy = np.corrcoef(idade, distancia)[0, 1]

# Método 2: Usando scipy.stats
correlacao_scipy, p_value = stats.pearsonr(idade, distancia)

# Método 3: Cálculo manual
n = len(idade)
x_mean = np.mean(idade)
y_mean = np.mean(distancia)

numerador = np.sum((idade - x_mean) * (distancia - y_mean))
denominador_x = np.sqrt(np.sum((idade - x_mean)**2))
denominador_y = np.sqrt(np.sum((distancia - y_mean)**2))

correlacao_manual = numerador / (denominador_x * denominador_y)

print(f"\nCoeficiente de Correlação de Pearson (r):")
print(f"  Usando NumPy:     r = {correlacao_numpy:.6f}")
print(f"  Usando SciPy:     r = {correlacao_scipy:.6f}")
print(f"  Cálculo Manual:   r = {correlacao_manual:.6f}")

print(f"\nValor de r² (coeficiente de determinação): {correlacao_numpy**2:.6f}")

# Interpretação
r_abs = abs(correlacao_numpy)
if r_abs >= 0.9:
    forca = "muito forte"
elif r_abs >= 0.7:
    forca = "forte"
elif r_abs >= 0.5:
    forca = "moderada"
elif r_abs >= 0.3:
    forca = "fraca"
else:
    forca = "muito fraca"

direcao = "negativa" if correlacao_numpy < 0 else "positiva"

print(f"\nInterpretação:")
print(f"  Correlação {direcao} e {forca}")
print(f"  p-value: {p_value:.6f}")

if p_value < 0.05:
    print(f"  A correlação é estatisticamente significativa (p < 0.05)")
else:
    print(f"  A correlação NÃO é estatisticamente significativa (p >= 0.05)")

# ============================================
# 3. OBTER MODELO DE REGRESSÃO LINEAR y = a + bx
# ============================================
print("\n" + "=" * 70)
print("3. MODELO DE REGRESSÃO LINEAR: y = a + bx")
print("=" * 70)

# Método 1: Usando scikit-learn
X = idade.reshape(-1, 1)  # Precisa ser 2D para sklearn
y = distancia

modelo_sklearn = LinearRegression()
modelo_sklearn.fit(X, y)

a_sklearn = modelo_sklearn.intercept_  # Intercepto (a)
b_sklearn = modelo_sklearn.coef_[0]    # Coeficiente angular (b)

# Método 2: Cálculo manual usando fórmulas
# b = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
# a = ȳ - b*x̄

b_manual = np.sum((idade - x_mean) * (distancia - y_mean)) / np.sum((idade - x_mean)**2)
a_manual = y_mean - b_manual * x_mean

# Método 3: Usando scipy.stats.linregress
slope, intercept, r_value, p_value_reg, std_err = stats.linregress(idade, distancia)

print(f"\nParâmetros do modelo:")
print(f"  Usando scikit-learn:")
print(f"    a (intercepto) = {a_sklearn:.4f}")
print(f"    b (coeficiente angular) = {b_sklearn:.4f}")
print(f"    Equação: y = {a_sklearn:.4f} + {b_sklearn:.4f}*x")

print(f"\n  Usando cálculo manual:")
print(f"    a (intercepto) = {a_manual:.4f}")
print(f"    b (coeficiente angular) = {b_manual:.4f}")
print(f"    Equação: y = {a_manual:.4f} + {b_manual:.4f}*x")

print(f"\n  Usando scipy.stats:")
print(f"    a (intercepto) = {intercept:.4f}")
print(f"    b (coeficiente angular) = {slope:.4f}")
print(f"    Equação: y = {intercept:.4f} + {slope:.4f}*x")

# Usar valores do scikit-learn (mais preciso)
a = a_sklearn
b = b_sklearn

# Calcular R²
y_pred = a + b * idade
ss_res = np.sum((distancia - y_pred)**2)
ss_tot = np.sum((distancia - y_mean)**2)
r2 = 1 - (ss_res / ss_tot)

print(f"\nCoeficiente de determinação (R²): {r2:.6f}")
print(f"  {r2*100:.2f}% da variabilidade em y é explicada pelo modelo")

# ============================================
# 4. PREDIZER DISTÂNCIA PARA MOTORISTA DE 75 ANOS
# ============================================
print("\n" + "=" * 70)
print("4. PREDIÇÃO PARA MOTORISTA DE 75 ANOS")
print("=" * 70)

idade_pred = 75
distancia_pred = a + b * idade_pred

print(f"\nIdade: {idade_pred} anos")
print(f"Distância predita: {distancia_pred:.2f} metros")

# Intervalo de confiança (aproximado)
# Erro padrão residual
n = len(idade)
mse = ss_res / (n - 2)  # Mean Squared Error
se_residual = np.sqrt(mse)

# Erro padrão da predição
x_pred = idade_pred
x_mean = np.mean(idade)
sxx = np.sum((idade - x_mean)**2)
se_pred = se_residual * np.sqrt(1 + 1/n + (x_pred - x_mean)**2 / sxx)

# Intervalo de confiança 95% (usando t-student com n-2 graus de liberdade)
t_critical = stats.t.ppf(0.975, n - 2)
ic_inferior = distancia_pred - t_critical * se_pred
ic_superior = distancia_pred + t_critical * se_pred

print(f"\nIntervalo de Confiança 95%:")
print(f"  [{ic_inferior:.2f}, {ic_superior:.2f}] metros")

# ============================================
# 5. VISUALIZAÇÃO
# ============================================
print("\n" + "=" * 70)
print("5. GERANDO VISUALIZAÇÃO")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico 1: Dados e linha de regressão
ax1 = axes[0]
ax1.scatter(idade, distancia, s=100, color='blue', alpha=0.7, 
           edgecolors='black', linewidths=1.5, label='Dados observados', zorder=3)

# Linha de regressão
idade_plot = np.linspace(idade.min() - 5, idade.max() + 20, 100)
distancia_plot = a + b * idade_plot
ax1.plot(idade_plot, distancia_plot, 'r-', linewidth=2, 
        label=f'Regressão: y = {a:.2f} + {b:.2f}x', zorder=2)

# Ponto predito para 75 anos
ax1.scatter(idade_pred, distancia_pred, s=200, color='red', 
           marker='*', edgecolors='black', linewidths=2, 
           label=f'Predição (75 anos): {distancia_pred:.1f}m', zorder=4)

ax1.set_xlabel('Idade (anos)', fontsize=12)
ax1.set_ylabel('Distância de Visão (metros)', fontsize=12)
ax1.set_title('Regressão Linear: Idade vs Distância de Visão', 
             fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(15, 80)

# Gráfico 2: Resíduos
ax2 = axes[1]
residuos = distancia - y_pred
ax2.scatter(idade, residuos, s=100, color='green', alpha=0.7, 
           edgecolors='black', linewidths=1.5, zorder=3)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, zorder=2)
ax2.set_xlabel('Idade (anos)', fontsize=12)
ax2.set_ylabel('Resíduos (metros)', fontsize=12)
ax2.set_title('Gráfico de Resíduos', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Salvar figura
script_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(script_dir, 'regressao_linear.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nGráfico salvo em: {output_file}")

# ============================================
# 6. RESUMO ESTATÍSTICO
# ============================================
print("\n" + "=" * 70)
print("6. RESUMO ESTATÍSTICO")
print("=" * 70)

print(f"\nEstatísticas Descritivas - Idade:")
print(f"  Média: {np.mean(idade):.2f} anos")
print(f"  Desvio Padrão: {np.std(idade, ddof=1):.2f} anos")
print(f"  Mínimo: {np.min(idade)} anos")
print(f"  Máximo: {np.max(idade)} anos")

print(f"\nEstatísticas Descritivas - Distância:")
print(f"  Média: {np.mean(distancia):.2f} metros")
print(f"  Desvio Padrão: {np.std(distancia, ddof=1):.2f} metros")
print(f"  Mínimo: {np.min(distancia)} metros")
print(f"  Máximo: {np.max(distancia)} metros")

print(f"\nModelo de Regressão:")
print(f"  Equação: y = {a:.4f} + {b:.4f}*x")
print(f"  Coeficiente de Correlação (r): {correlacao_numpy:.6f}")
print(f"  Coeficiente de Determinação (R²): {r2:.6f}")
print(f"  Erro Padrão Residual: {se_residual:.4f} metros")

print(f"\nPredição:")
print(f"  Idade: 75 anos")
print(f"  Distância predita: {distancia_pred:.2f} metros")
print(f"  Intervalo de Confiança 95%: [{ic_inferior:.2f}, {ic_superior:.2f}] metros")

print("\n" + "=" * 70)
print("ANÁLISE CONCLUÍDA!")
print("=" * 70)

# Fechar figura (comentado para não bloquear)
plt.close()

