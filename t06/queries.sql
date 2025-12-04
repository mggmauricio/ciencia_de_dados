-- ============================================
-- CONSULTAS SQL - BANCO DE DADOS NORTHWIND
-- ============================================

-- 1) Mostrar todos os territórios e seus códigos (tabela "Territories")
SELECT 
    TerritoryID AS Codigo,
    TerritoryDescription AS Territorio,
    RegionID AS CodigoRegiao
FROM 
    Territories
ORDER BY 
    TerritoryID;

-- ============================================

-- 2) Mostrar todos os nomes dos funcionários que atendem o território relacionado a Chicago
-- O código de Chicago é '60601'
SELECT DISTINCT
    e.EmployeeID,
    e.FirstName AS Nome,
    e.LastName AS Sobrenome,
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
    e.LastName, e.FirstName;

-- ============================================

-- 3) Mostrar os pedidos (tabela "Orders") de todos os funcionários que atendem o território relacionado a Chicago
-- O código de Chicago é '60601'
SELECT 
    o.OrderID AS CodigoPedido,
    o.OrderDate AS DataPedido,
    o.RequiredDate AS DataRequerida,
    o.ShippedDate AS DataEnvio,
    o.ShipCity AS CidadeEnvio,
    o.ShipCountry AS PaisEnvio,
    e.EmployeeID,
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
    o.OrderDate DESC;

-- ============================================

