"""
Script para criar e popular tabelas no banco de dados Northwind:
- Tabela de Clientes (com dados da API randomuser.me)
- Tabela de Fornecedores (com dados da API randomuser.me)
- Tabela de Produtos
- Tabela de Pedidos
"""

import sqlite3
import requests
import random
import json

# Conectar ao banco de dados
conn = sqlite3.connect('Northwind.db')
cursor = conn.cursor()

print("=== Criando e populando tabelas ===\n")

# ============================================
# 1. CRIAR TABELA DE CLIENTES
# ============================================
print("1. Criando tabela Clientes...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS Clientes (
        ClienteID INTEGER PRIMARY KEY AUTOINCREMENT,
        Nome TEXT NOT NULL,
        Sobrenome TEXT NOT NULL,
        Email TEXT UNIQUE NOT NULL,
        Telefone TEXT,
        DataNascimento DATE,
        Cidade TEXT,
        Estado TEXT,
        Pais TEXT,
        CodigoPostal TEXT,
        DataCadastro DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")
print("   [OK] Tabela Clientes criada\n")

# ============================================
# 2. CRIAR TABELA DE FORNECEDORES
# ============================================
print("2. Criando tabela Fornecedores...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS Fornecedores (
        FornecedorID INTEGER PRIMARY KEY AUTOINCREMENT,
        Nome TEXT NOT NULL,
        Sobrenome TEXT NOT NULL,
        Email TEXT UNIQUE NOT NULL,
        Telefone TEXT,
        Cidade TEXT,
        Estado TEXT,
        Pais TEXT,
        CodigoPostal TEXT,
        NomeEmpresa TEXT,
        DataCadastro DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")
print("   [OK] Tabela Fornecedores criada\n")

# ============================================
# 3. CRIAR TABELA DE PRODUTOS
# ============================================
print("3. Criando tabela Produtos...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS Produtos (
        CodigoProduto INTEGER PRIMARY KEY AUTOINCREMENT,
        NomeProduto TEXT NOT NULL,
        Descricao TEXT,
        PrecoUnitario REAL,
        UnidadeEstoque INTEGER DEFAULT 0,
        DataCadastro DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")
print("   [OK] Tabela Produtos criada\n")

# ============================================
# 4. CRIAR TABELA DE PEDIDOS
# ============================================
print("4. Criando tabela Pedidos...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS Pedidos (
        PedidoID INTEGER PRIMARY KEY AUTOINCREMENT,
        ClienteID INTEGER NOT NULL,
        FornecedorID INTEGER NOT NULL,
        CodigoProduto INTEGER NOT NULL,
        Quantidade INTEGER NOT NULL CHECK(Quantidade > 0),
        DataPedido DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (ClienteID) REFERENCES Clientes(ClienteID),
        FOREIGN KEY (FornecedorID) REFERENCES Fornecedores(FornecedorID),
        FOREIGN KEY (CodigoProduto) REFERENCES Produtos(CodigoProduto)
    )
""")
print("   [OK] Tabela Pedidos criada\n")

# ============================================
# 5. POPULAR TABELA DE CLIENTES COM API
# ============================================
print("5. Populando tabela Clientes com dados da API randomuser.me...")
try:
    # Buscar 20 clientes
    response = requests.get('https://randomuser.me/api/?results=20&nat=br,us,gb&noinfo')
    response.raise_for_status()
    data = response.json()
    
    clientes_inseridos = 0
    for person in data['results']:
        try:
            cursor.execute("""
                INSERT INTO Clientes (Nome, Sobrenome, Email, Telefone, DataNascimento, 
                                   Cidade, Estado, Pais, CodigoPostal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                person['name']['first'],
                person['name']['last'],
                person['email'],
                person['phone'],
                person['dob']['date'][:10],  # Apenas a data (YYYY-MM-DD)
                person['location']['city'],
                person['location']['state'],
                person['location']['country'],
                person['location']['postcode']
            ))
            clientes_inseridos += 1
        except sqlite3.IntegrityError:
            # Email duplicado, pular
            continue
    
    print(f"   [OK] {clientes_inseridos} clientes inseridos\n")
except Exception as e:
    print(f"   [ERRO] Erro ao popular clientes: {e}\n")

# ============================================
# 6. POPULAR TABELA DE FORNECEDORES COM API
# ============================================
print("6. Populando tabela Fornecedores com dados da API randomuser.me...")
try:
    # Buscar 15 fornecedores
    response = requests.get('https://randomuser.me/api/?results=15&nat=br,us,gb&noinfo')
    response.raise_for_status()
    data = response.json()
    
    fornecedores_inseridos = 0
    empresas = ['Tech Solutions', 'Global Supply', 'Prime Distributors', 
                'Elite Products', 'Mega Corp', 'Super Suppliers', 'Top Trade']
    
    for person in data['results']:
        try:
            cursor.execute("""
                INSERT INTO Fornecedores (Nome, Sobrenome, Email, Telefone, 
                                        Cidade, Estado, Pais, CodigoPostal, NomeEmpresa)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                person['name']['first'],
                person['name']['last'],
                person['email'],
                person['phone'],
                person['location']['city'],
                person['location']['state'],
                person['location']['country'],
                person['location']['postcode'],
                random.choice(empresas) + ' ' + person['name']['last']
            ))
            fornecedores_inseridos += 1
        except sqlite3.IntegrityError:
            # Email duplicado, pular
            continue
    
    print(f"   [OK] {fornecedores_inseridos} fornecedores inseridos\n")
except Exception as e:
    print(f"   [ERRO] Erro ao popular fornecedores: {e}\n")

# ============================================
# 7. POPULAR TABELA DE PRODUTOS
# ============================================
print("7. Populando tabela Produtos...")
produtos = [
    ('Notebook Dell', 'Notebook Dell Inspiron 15', 2999.90, 50),
    ('Mouse Logitech', 'Mouse sem fio Logitech MX Master', 299.90, 100),
    ('Teclado Mecânico', 'Teclado mecânico RGB', 599.90, 75),
    ('Monitor LG', 'Monitor LG 27 polegadas 4K', 1899.90, 30),
    ('Webcam HD', 'Webcam Full HD 1080p', 399.90, 60),
    ('Headset Gamer', 'Headset gamer com microfone', 499.90, 80),
    ('SSD 1TB', 'SSD SATA 1TB', 599.90, 45),
    ('Memória RAM 16GB', 'Memória RAM DDR4 16GB', 449.90, 90),
    ('Placa de Vídeo', 'Placa de vídeo RTX 3060', 2499.90, 20),
    ('Fonte 750W', 'Fonte de alimentação 750W', 599.90, 40),
    ('Gabinete ATX', 'Gabinete gamer com RGB', 699.90, 35),
    ('Cooler CPU', 'Cooler para processador', 299.90, 55),
    ('HD Externo 2TB', 'HD externo USB 3.0 2TB', 699.90, 25),
    ('Pen Drive 64GB', 'Pen Drive USB 3.0 64GB', 49.90, 200),
    ('Hub USB', 'Hub USB 3.0 com 7 portas', 149.90, 70)
]

for produto in produtos:
    cursor.execute("""
        INSERT INTO Produtos (NomeProduto, Descricao, PrecoUnitario, UnidadeEstoque)
        VALUES (?, ?, ?, ?)
    """, produto)

print(f"   [OK] {len(produtos)} produtos inseridos\n")

# ============================================
# 8. POPULAR TABELA DE PEDIDOS
# ============================================
print("8. Populando tabela Pedidos...")

# Buscar IDs existentes
cursor.execute("SELECT ClienteID FROM Clientes")
clientes_ids = [row[0] for row in cursor.fetchall()]

cursor.execute("SELECT FornecedorID FROM Fornecedores")
fornecedores_ids = [row[0] for row in cursor.fetchall()]

cursor.execute("SELECT CodigoProduto FROM Produtos")
produtos_ids = [row[0] for row in cursor.fetchall()]

# Criar 50 pedidos aleatórios
pedidos_inseridos = 0
for _ in range(50):
    if clientes_ids and fornecedores_ids and produtos_ids:
        cursor.execute("""
            INSERT INTO Pedidos (ClienteID, FornecedorID, CodigoProduto, Quantidade)
            VALUES (?, ?, ?, ?)
        """, (
            random.choice(clientes_ids),
            random.choice(fornecedores_ids),
            random.choice(produtos_ids),
            random.randint(1, 10)
        ))
        pedidos_inseridos += 1

print(f"   [OK] {pedidos_inseridos} pedidos inseridos\n")

# ============================================
# COMMIT E VERIFICAÇÃO
# ============================================
conn.commit()

print("=== Verificação das tabelas criadas ===\n")

# Contar registros
tabelas = ['Clientes', 'Fornecedores', 'Produtos', 'Pedidos']
for tabela in tabelas:
    cursor.execute(f"SELECT COUNT(*) FROM {tabela}")
    count = cursor.fetchone()[0]
    print(f"{tabela}: {count} registros")

print("\n[OK] Todas as tabelas foram criadas e populadas com sucesso!")

# Fechar conexão
conn.close()

