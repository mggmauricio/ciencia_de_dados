"""
Script para testar as consultas SQL do arquivo queries.sql
"""

import sqlite3

conn = sqlite3.connect('Northwind.db')
cursor = conn.cursor()

print("=" * 70)
print("TESTE DAS CONSULTAS SQL")
print("=" * 70)

# ============================================
# Consulta 1: Territórios e códigos
# ============================================
print("\n1) TERRITÓRIOS E CÓDIGOS")
print("-" * 70)
cursor.execute("""
    SELECT 
        TerritoryID AS Codigo,
        TerritoryDescription AS Territorio,
        RegionID AS CodigoRegiao
    FROM 
        Territories
    ORDER BY 
        TerritoryID
    LIMIT 10
""")
print(f"{'Código':<10} {'Território':<40} {'Região':<10}")
print("-" * 70)
for row in cursor.fetchall():
    print(f"{row[0]:<10} {row[1].strip():<40} {row[2]:<10}")
print(f"\nTotal de territórios: {cursor.execute('SELECT COUNT(*) FROM Territories').fetchone()[0]}")

# ============================================
# Consulta 2: Funcionários que atendem Chicago
# ============================================
print("\n\n2) FUNCIONÁRIOS QUE ATENDEM CHICAGO (Código: 60601)")
print("-" * 70)
cursor.execute("""
    SELECT DISTINCT
        e.EmployeeID,
        e.FirstName || ' ' || e.LastName AS NomeCompleto,
        t.TerritoryID,
        t.TerritoryDescription AS Territorio
    FROM 
        Employees e
    INNER JOIN 
        EmployeeTerritories et ON e.EmployeeID = et.EmployeeID
    INNER JOIN 
        Territories t ON et.TerritoryID = t.TerritoryID
    WHERE 
        t.TerritoryID = '60601'
    ORDER BY 
        e.LastName, e.FirstName
""")
print(f"{'ID':<5} {'Nome Completo':<30} {'Código':<10} {'Território':<20}")
print("-" * 70)
for row in cursor.fetchall():
    print(f"{row[0]:<5} {row[1]:<30} {row[2]:<10} {row[3].strip():<20}")

# ============================================
# Consulta 3: Pedidos dos funcionários de Chicago
# ============================================
print("\n\n3) PEDIDOS DOS FUNCIONÁRIOS QUE ATENDEM CHICAGO")
print("-" * 70)
cursor.execute("""
    SELECT 
        o.OrderID AS CodigoPedido,
        o.OrderDate AS DataPedido,
        e.FirstName || ' ' || e.LastName AS Funcionario,
        t.TerritoryID,
        t.TerritoryDescription AS Territorio
    FROM 
        Orders o
    INNER JOIN 
        Employees e ON o.EmployeeID = e.EmployeeID
    INNER JOIN 
        EmployeeTerritories et ON e.EmployeeID = et.EmployeeID
    INNER JOIN 
        Territories t ON et.TerritoryID = t.TerritoryID
    WHERE 
        t.TerritoryID = '60601'
    ORDER BY 
        o.OrderDate DESC
    LIMIT 10
""")
print(f"{'Pedido':<8} {'Data':<12} {'Funcionário':<25} {'Código':<8} {'Território':<15}")
print("-" * 70)
for row in cursor.fetchall():
    data = row[1][:10] if row[1] else 'N/A'
    print(f"{row[0]:<8} {data:<12} {row[2]:<25} {row[3]:<8} {row[4].strip():<15}")

total_pedidos = cursor.execute("""
    SELECT COUNT(*) FROM Orders o
    INNER JOIN Employees e ON o.EmployeeID = e.EmployeeID
    INNER JOIN EmployeeTerritories et ON e.EmployeeID = et.EmployeeID
    INNER JOIN Territories t ON et.TerritoryID = t.TerritoryID
    WHERE t.TerritoryID = '60601'
""").fetchone()[0]
print(f"\nTotal de pedidos: {total_pedidos}")

# ============================================
# Verificação das novas tabelas
# ============================================
print("\n\n4) VERIFICAÇÃO DAS NOVAS TABELAS CRIADAS")
print("-" * 70)

tabelas = ['Clientes', 'Fornecedores', 'Produtos', 'Pedidos']
for tabela in tabelas:
    cursor.execute(f"SELECT COUNT(*) FROM {tabela}")
    count = cursor.fetchone()[0]
    print(f"{tabela}: {count} registros")

# Mostrar alguns exemplos
print("\n--- Exemplo de Clientes ---")
cursor.execute("SELECT ClienteID, Nome, Sobrenome, Email, Cidade FROM Clientes LIMIT 5")
for row in cursor.fetchall():
    print(f"ID: {row[0]}, Nome: {row[1]} {row[2]}, Email: {row[3]}, Cidade: {row[4]}")

print("\n--- Exemplo de Fornecedores ---")
cursor.execute("SELECT FornecedorID, Nome, Sobrenome, NomeEmpresa, Cidade FROM Fornecedores LIMIT 5")
for row in cursor.fetchall():
    print(f"ID: {row[0]}, Nome: {row[1]} {row[2]}, Empresa: {row[3]}, Cidade: {row[4]}")

print("\n--- Exemplo de Produtos ---")
cursor.execute("SELECT CodigoProduto, NomeProduto, PrecoUnitario, UnidadeEstoque FROM Produtos LIMIT 5")
for row in cursor.fetchall():
    print(f"Código: {row[0]}, Nome: {row[1]}, Preço: R$ {row[2]:.2f}, Estoque: {row[3]}")

print("\n--- Exemplo de Pedidos ---")
cursor.execute("""
    SELECT p.PedidoID, c.Nome || ' ' || c.Sobrenome AS Cliente, 
           f.NomeEmpresa AS Fornecedor, pr.NomeProduto, p.Quantidade
    FROM Pedidos p
    INNER JOIN Clientes c ON p.ClienteID = c.ClienteID
    INNER JOIN Fornecedores f ON p.FornecedorID = f.FornecedorID
    INNER JOIN Produtos pr ON p.CodigoProduto = pr.CodigoProduto
    LIMIT 5
""")
for row in cursor.fetchall():
    print(f"Pedido {row[0]}: Cliente: {row[1]}, Fornecedor: {row[2]}, Produto: {row[3]}, Qtd: {row[4]}")

print("\n" + "=" * 70)
print("Testes concluídos!")
print("=" * 70)

conn.close()

