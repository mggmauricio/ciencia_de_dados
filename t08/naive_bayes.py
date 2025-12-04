"""
Classificador Naïve Bayes
Análise da tabela verdade com variáveis A, B, C e saída Y
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import math

# ============================================
# 1. CRIAR DATASET A PARTIR DA TABELA VERDADE
# ============================================
print("=" * 70)
print("CLASSIFICADOR NAÏVE BAYES")
print("=" * 70)

# Dados da tabela verdade fornecida
A = [0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1]
B = [0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1]
C = [0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1]
Y = [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]

# Criar DataFrame
df = pd.DataFrame({'A': A, 'B': B, 'C': C, 'Y': Y})

print("\n1. DATASET (Tabela Verdade)")
print("-" * 70)
print(df.to_string(index=False))
print(f"\nTotal de instâncias: {len(df)}")
print(f"Distribuição de classes Y: {dict(Counter(Y))}")
print()

# ============================================
# 2. CÁLCULO MANUAL DO NAÏVE BAYES
# ============================================
print("=" * 70)
print("2. CÁLCULO MANUAL - NAÏVE BAYES")
print("=" * 70)
print("\nInstância a predizer: A=0, B=0, C=1")
print("-" * 70)

# Calcular probabilidades a priori P(Y)
total = len(df)
y_counts = Counter(Y)
P_Y0 = y_counts[0] / total
P_Y1 = y_counts[1] / total

print(f"\nProbabilidades a priori:")
print(f"  P(Y=0) = {y_counts[0]}/{total} = {P_Y0:.4f}")
print(f"  P(Y=1) = {y_counts[1]}/{total} = {P_Y1:.4f}")

# Calcular probabilidades condicionais P(A|Y), P(B|Y), P(C|Y)
def calcular_probabilidade_condicional(df, atributo, valor_atributo, classe):
    """Calcula P(atributo=valor_atributo | Y=classe)"""
    df_classe = df[df['Y'] == classe]
    if len(df_classe) == 0:
        return 0.0
    count = len(df_classe[df_classe[atributo] == valor_atributo])
    return count / len(df_classe)

# Para Y=0
P_A0_Y0 = calcular_probabilidade_condicional(df, 'A', 0, 0)
P_B0_Y0 = calcular_probabilidade_condicional(df, 'B', 0, 0)
P_C1_Y0 = calcular_probabilidade_condicional(df, 'C', 1, 0)

# Para Y=1
P_A0_Y1 = calcular_probabilidade_condicional(df, 'A', 0, 1)
P_B0_Y1 = calcular_probabilidade_condicional(df, 'B', 0, 1)
P_C1_Y1 = calcular_probabilidade_condicional(df, 'C', 1, 1)

print(f"\nProbabilidades condicionais para Y=0:")
print(f"  P(A=0|Y=0) = {P_A0_Y0:.4f}")
print(f"  P(B=0|Y=0) = {P_B0_Y0:.4f}")
print(f"  P(C=1|Y=0) = {P_C1_Y0:.4f}")

print(f"\nProbabilidades condicionais para Y=1:")
print(f"  P(A=0|Y=1) = {P_A0_Y1:.4f}")
print(f"  P(B=0|Y=1) = {P_B0_Y1:.4f}")
print(f"  P(C=1|Y=1) = {P_C1_Y1:.4f}")

# Calcular probabilidades posteriores usando Naïve Bayes
# P(Y|A,B,C) ∝ P(Y) * P(A|Y) * P(B|Y) * P(C|Y)

# Para Y=0
prob_Y0 = P_Y0 * P_A0_Y0 * P_B0_Y0 * P_C1_Y0

# Para Y=1
prob_Y1 = P_Y1 * P_A0_Y1 * P_B0_Y1 * P_C1_Y1

print(f"\nProbabilidades não normalizadas:")
print(f"  P(Y=0|A=0,B=0,C=1) ∝ {P_Y0:.4f} × {P_A0_Y0:.4f} × {P_B0_Y0:.4f} × {P_C1_Y0:.4f} = {prob_Y0:.6f}")
print(f"  P(Y=1|A=0,B=0,C=1) ∝ {P_Y1:.4f} × {P_A0_Y1:.4f} × {P_B0_Y1:.4f} × {P_C1_Y1:.4f} = {prob_Y1:.6f}")

# Normalizar probabilidades
soma = prob_Y0 + prob_Y1
if soma > 0:
    prob_Y0_norm = prob_Y0 / soma
    prob_Y1_norm = prob_Y1 / soma
else:
    prob_Y0_norm = 0.5
    prob_Y1_norm = 0.5

print(f"\nProbabilidades normalizadas:")
print(f"  P(Y=0|A=0,B=0,C=1) = {prob_Y0_norm:.4f}")
print(f"  P(Y=1|A=0,B=0,C=1) = {prob_Y1_norm:.4f}")

# Predição
predicao_manual = 0 if prob_Y0_norm > prob_Y1_norm else 1
print(f"\nRESULTADO DA PREDIÇÃO MANUAL:")
print(f"  Classe predita: Y = {predicao_manual}")
print(f"  (Probabilidade Y=0: {prob_Y0_norm:.4f}, Y=1: {prob_Y1_norm:.4f})")

# ============================================
# 3. IMPLEMENTAÇÃO DO ZERO - NAÏVE BAYES
# ============================================
print("\n" + "=" * 70)
print("3. IMPLEMENTAÇÃO DO ZERO - NAÏVE BAYES")
print("=" * 70)

class NaiveBayes:
    """Implementação do classificador Naïve Bayes do zero"""
    
    def __init__(self):
        self.prior_probs = {}
        self.conditional_probs = {}
        self.classes = None
    
    def fit(self, X, y):
        """Treina o classificador"""
        self.classes = np.unique(y)
        n_samples = len(y)
        
        # Calcular probabilidades a priori
        for cls in self.classes:
            self.prior_probs[cls] = np.sum(y == cls) / n_samples
        
        # Calcular probabilidades condicionais
        n_features = X.shape[1]
        for cls in self.classes:
            self.conditional_probs[cls] = {}
            X_cls = X[y == cls]
            
            for feature_idx in range(n_features):
                self.conditional_probs[cls][feature_idx] = {}
                unique_values = np.unique(X[:, feature_idx])
                
                for value in unique_values:
                    count = np.sum(X_cls[:, feature_idx] == value)
                    # Usar suavização de Laplace (add-1 smoothing)
                    self.conditional_probs[cls][feature_idx][value] = (count + 1) / (len(X_cls) + len(unique_values))
    
    def predict_proba(self, X):
        """Retorna probabilidades para cada classe"""
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        probabilities = np.zeros((n_samples, n_classes))
        
        for i, sample in enumerate(X):
            for j, cls in enumerate(self.classes):
                # Iniciar com probabilidade a priori
                prob = self.prior_probs[cls]
                
                # Multiplicar pelas probabilidades condicionais
                for feature_idx, value in enumerate(sample):
                    if value in self.conditional_probs[cls][feature_idx]:
                        prob *= self.conditional_probs[cls][feature_idx][value]
                    else:
                        # Se valor nunca visto, usar probabilidade muito pequena
                        prob *= 0.001
                
                probabilities[i, j] = prob
            
            # Normalizar
            if probabilities[i].sum() > 0:
                probabilities[i] /= probabilities[i].sum()
        
        return probabilities
    
    def predict(self, X):
        """Prediz a classe"""
        probabilities = self.predict_proba(X)
        return self.classes[np.argmax(probabilities, axis=1)]

# Preparar dados
X = df[['A', 'B', 'C']].values
y = df['Y'].values

# Treinar modelo
nb_custom = NaiveBayes()
nb_custom.fit(X, y)

# Predizer para A=0, B=0, C=1
instancia_teste = np.array([[0, 0, 1]])
probabilidades = nb_custom.predict_proba(instancia_teste)
predicao_custom = nb_custom.predict(instancia_teste)

print(f"\nPredição para A=0, B=0, C=1:")
print(f"  P(Y=0) = {probabilidades[0][0]:.4f}")
print(f"  P(Y=1) = {probabilidades[0][1]:.4f}")
print(f"  Classe predita: Y = {predicao_custom[0]}")

# Avaliar no dataset completo
y_pred_custom = nb_custom.predict(X)
acuracia_custom = accuracy_score(y, y_pred_custom)
print(f"\nAcurácia no dataset completo: {acuracia_custom:.4f}")

# ============================================
# 4. USANDO BIBLIOTECA SCIKIT-LEARN
# ============================================
print("\n" + "=" * 70)
print("4. USANDO BIBLIOTECA SCIKIT-LEARN")
print("=" * 70)

# Bernoulli Naïve Bayes (para dados binários)
print("\n4.1. Bernoulli Naïve Bayes (para dados binários):")
nb_bernoulli = BernoulliNB(alpha=1.0)  # alpha=1.0 é suavização de Laplace
nb_bernoulli.fit(X, y)

prob_bernoulli = nb_bernoulli.predict_proba(instancia_teste)
pred_bernoulli = nb_bernoulli.predict(instancia_teste)

print(f"  Predição para A=0, B=0, C=1:")
print(f"    P(Y=0) = {prob_bernoulli[0][0]:.4f}")
print(f"    P(Y=1) = {prob_bernoulli[0][1]:.4f}")
print(f"    Classe predita: Y = {pred_bernoulli[0]}")

y_pred_bernoulli = nb_bernoulli.predict(X)
acuracia_bernoulli = accuracy_score(y, y_pred_bernoulli)
print(f"  Acurácia no dataset completo: {acuracia_bernoulli:.4f}")

# Gaussian Naïve Bayes (assume distribuição normal)
print("\n4.2. Gaussian Naïve Bayes:")
nb_gaussian = GaussianNB()
nb_gaussian.fit(X, y)

prob_gaussian = nb_gaussian.predict_proba(instancia_teste)
pred_gaussian = nb_gaussian.predict(instancia_teste)

print(f"  Predição para A=0, B=0, C=1:")
print(f"    P(Y=0) = {prob_gaussian[0][0]:.4f}")
print(f"    P(Y=1) = {prob_gaussian[0][1]:.4f}")
print(f"    Classe predita: Y = {pred_gaussian[0]}")

y_pred_gaussian = nb_gaussian.predict(X)
acuracia_gaussian = accuracy_score(y, y_pred_gaussian)
print(f"  Acurácia no dataset completo: {acuracia_gaussian:.4f}")

# ============================================
# 5. COMPARAÇÃO DOS RESULTADOS
# ============================================
print("\n" + "=" * 70)
print("5. COMPARAÇÃO DOS RESULTADOS")
print("=" * 70)
print("\nInstância: A=0, B=0, C=1")
print("-" * 70)
print(f"{'Método':<25} {'P(Y=0)':<12} {'P(Y=1)':<12} {'Predição':<10}")
print("-" * 70)
print(f"{'Cálculo Manual':<25} {prob_Y0_norm:<12.4f} {prob_Y1_norm:<12.4f} {predicao_manual:<10}")
print(f"{'Implementação Própria':<25} {probabilidades[0][0]:<12.4f} {probabilidades[0][1]:<12.4f} {predicao_custom[0]:<10}")
print(f"{'Bernoulli NB (sklearn)':<25} {prob_bernoulli[0][0]:<12.4f} {prob_bernoulli[0][1]:<12.4f} {pred_bernoulli[0]:<10}")
print(f"{'Gaussian NB (sklearn)':<25} {prob_gaussian[0][0]:<12.4f} {prob_gaussian[0][1]:<12.4f} {pred_gaussian[0]:<10}")
