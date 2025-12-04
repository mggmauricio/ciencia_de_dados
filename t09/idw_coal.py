"""
Interpolação IDW (Inverse Distance Weighting) para o dataset coal.csv
Cria mapa interpolado do valor calorífico com resolução 50x50 pontos
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def idwr(x, y, z, xnew, ynew, power=2):
    """
    Interpolação IDW (Inverse Distance Weighting)
    
    Parâmetros:
    -----------
    x, y : arrays
        Coordenadas dos pontos conhecidos
    z : array
        Valores nos pontos conhecidos
    xnew, ynew : arrays
        Coordenadas dos pontos a interpolar
    power : float, default=2
        Potência da distância (geralmente 2)
    
    Retorna:
    --------
    znew : array
        Valores interpolados
    """
    znew = np.zeros(len(xnew))
    
    for i in range(len(xnew)):
        # Calcular distâncias
        distances = np.sqrt((x - xnew[i])**2 + (y - ynew[i])**2)
        
        # Evitar divisão por zero (se ponto coincide com amostra)
        if np.any(distances == 0):
            idx = np.where(distances == 0)[0][0]
            znew[i] = z[idx]
        else:
            # Calcular pesos (inverso da distância elevada à potência)
            weights = 1.0 / (distances ** power)
            
            # Interpolar
            znew[i] = np.sum(weights * z) / np.sum(weights)
    
    return znew

print("=" * 70)
print("INTERPOLAÇÃO IDW - VALOR CALORÍFICO DO CARVÃO")
print("=" * 70)

# ============================================
# 1. LER ARQUIVO COAL.CSV
# ============================================
print("\n1. Lendo arquivo coal.csv...")

# Ler arquivo pulando as primeiras 4 linhas (cabeçalhos)
df = pd.read_csv("coal.csv", skiprows=4, header=None)

# Nomes das colunas (baseado na estrutura do arquivo)
# X co-ordinate, Y co-ordinate, Elevation of seam, Thickness of seam,
# Calorific Value (MJ), Ash content (%), Sulphur value (%)
df.columns = ['X', 'Y', 'Elevation', 'Thickness', 'Calorific_Value', 'Ash_Content', 'Sulphur']

# Converter para numpy array
vec = df[['X', 'Y', 'Calorific_Value']].to_numpy()

# Extrair coordenadas e valor calorífico
x = vec[:, 0]  # Coordenada X
y = vec[:, 1]  # Coordenada Y
z = vec[:, 2]  # Valor Calorífico (MJ)

print(f"   Total de pontos: {len(x)}")
print(f"   Coordenada X: min={x.min():.1f}, max={x.max():.1f}")
print(f"   Coordenada Y: min={y.min():.1f}, max={y.max():.1f}")
print(f"   Valor Calorífico: min={z.min():.2f} MJ, max={z.max():.2f} MJ")
print(f"   Valor Calorífico médio: {z.mean():.2f} MJ")

# ============================================
# 2. CRIAR GRADE REGULAR DE 50x50 PONTOS
# ============================================
print("\n2. Criando grade regular de 50x50 pontos...")

npt = 50
dx = (max(x) - min(x)) / npt
dy = (max(y) - min(y)) / npt

xx = np.arange(min(x), max(x), dx)
yy = np.arange(min(y), max(y), dy)

xnew, ynew = np.meshgrid(xx, yy)
xnew = xnew.reshape(npt * npt, 1)[:, 0]
ynew = ynew.reshape(npt * npt, 1)[:, 0]

print(f"   Resolução: {npt}x{npt} = {npt*npt} pontos")
print(f"   Passo X (dx): {dx:.2f}")
print(f"   Passo Y (dy): {dy:.2f}")

# ============================================
# 3. APLICAR INTERPOLAÇÃO IDW
# ============================================
print("\n3. Aplicando interpolação IDW...")

znew = idwr(x, y, z, xnew, ynew)

print(f"   Interpolação concluída!")
print(f"   Valores interpolados: min={znew.min():.2f} MJ, max={znew.max():.2f} MJ")

# ============================================
# 4. CONVERTER PARA MATRIZ
# ============================================
print("\n4. Convertendo resultado para matriz...")

znew_mat = znew.reshape(50, 50)

# ============================================
# 5. VISUALIZAR RESULTADO
# ============================================
print("\n5. Gerando visualização...")

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, aspect=0.67)

# Criar heatmap
sns.heatmap(znew_mat, 
            xticklabels=False, 
            yticklabels=False,
            cmap='viridis',
            cbar_kws={'label': 'Valor Calorífico (MJ)'})

# Adicionar pontos de entrada (amostras originais)
for i in range(len(x)):
    # Converter coordenadas para índices da matriz
    x_idx = round((x[i] - min(x)) / dx)
    y_idx = round((y[i] - min(y)) / dy)
    
    # Garantir que os índices estão dentro dos limites
    x_idx = max(0, min(npt - 1, x_idx))
    y_idx = max(0, min(npt - 1, y_idx))
    
    # Plotar marcador preto (contorno)
    ax.scatter(x_idx, y_idx, marker='*', s=200, color='black', 
               edgecolors='white', linewidths=1.5, zorder=3)
    
    # Plotar marcador amarelo (preenchimento)
    ax.scatter(x_idx, y_idx, marker='*', s=150, color='yellow', zorder=4)

# Configurar título e labels
plt.title('Mapa Interpolado IDW - Valor Calorífico do Carvão\n' +
          f'Resolução: {npt}x{npt} pontos | {len(x)} pontos de amostra',
          fontsize=14, fontweight='bold', pad=20)

plt.xlabel('Coordenada X', fontsize=12)
plt.ylabel('Coordenada Y', fontsize=12)

# Inverter eixo Y para corresponder às coordenadas originais
ax.invert_yaxis()

# Salvar figura
output_file = 'coal_idw_map.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n   Gráfico salvo em: {output_file}")

# Mostrar estatísticas
print("\n" + "=" * 70)
print("ESTATÍSTICAS")
print("=" * 70)
print(f"\nPontos de amostra: {len(x)}")
print(f"Valor Calorífico original:")
print(f"  Mínimo: {z.min():.2f} MJ")
print(f"  Máximo: {z.max():.2f} MJ")
print(f"  Média:  {z.mean():.2f} MJ")
print(f"  Desvio padrão: {z.std():.2f} MJ")

print(f"\nValor Calorífico interpolado (grade 50x50):")
print(f"  Mínimo: {znew.min():.2f} MJ")
print(f"  Máximo: {znew.max():.2f} MJ")
print(f"  Média:  {znew.mean():.2f} MJ")
print(f"  Desvio padrão: {znew.std():.2f} MJ")

print("\n" + "=" * 70)
print("INTERPOLAÇÃO CONCLUÍDA!")
print("=" * 70)

# Mostrar gráfico (comentado para não bloquear)
# plt.show()
plt.close()

