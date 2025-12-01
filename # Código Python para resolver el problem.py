import numpy as np

print("\n===== RESOLVEDOR DE ARMADURAS 2D - MÉTODO DE RIGIDEZ =====\n")

# ================================================================
# INGRESO DE NODOS
# ================================================================
n_nodos = int(input("¿Cuántos nodos tiene la armadura? "))

nodes = {}
for i in range(1, n_nodos + 1):
    print(f"\nNodo {i}:")
    x = float(input("  x = "))
    y = float(input("  y = "))
    nodes[i] = np.array([x, y])

# ================================================================
# INGRESO DE MIEMBROS
# ================================================================
n_members = int(input("\n¿Cuántos miembros (barras) tiene la armadura? "))

members = []
EA_list = []

for i in range(1, n_members + 1):
    print(f"\nMiembro {i}:")
    n1 = int(input("  Nodo inicial: "))
    n2 = int(input("  Nodo final: "))
    EA = float(input("  EA del miembro: "))
    members.append((n1, n2))
    EA_list.append(EA)

# ================================================================
# MATRIZ GLOBAL
# ================================================================
ndof = 2 * n_nodos
K = np.zeros((ndof, ndof))

for idx, (i, j) in enumerate(members):
    xi, yi = nodes[i]
    xj, yj = nodes[j]

    dx = xj - xi
    dy = yj - yi
    L = np.hypot(dx, dy)
    cx = dx / L
    cy = dy / L
    EA = EA_list[idx]

    k_local = (EA / L) * np.array([
        [ cx*cx, cx*cy, -cx*cx, -cx*cy],
        [ cx*cy, cy*cy, -cx*cy, -cy*cy],
        [-cx*cx, -cx*cy, cx*cx, cx*cy],
        [-cx*cy, -cy*cy, cx*cy, cy*cy]
    ])

    dof = [
        2*(i-1), 2*(i-1)+1,
        2*(j-1), 2*(j-1)+1
    ]

    for a in range(4):
        for b in range(4):
            K[dof[a], dof[b]] += k_local[a, b]

# ================================================================
# CARGAS
# ================================================================
F = np.zeros(ndof)

n_cargas = int(input("\n¿Cuántas cargas nodales existen? "))

for _ in range(n_cargas):
    print("\nCarga nodal:")
    nodo = int(input("  Nodo: "))
    fx = float(input("  Fx = "))
    fy = float(input("  Fy = "))
    F[2*(nodo-1)] += fx
    F[2*(nodo-1) + 1] += fy

# ================================================================
# APOYOS (RESTRICCIONES)
# ================================================================
fixed = []
print("\n=== Restricciones (apoyos) ===")
print("Ingrese DOFs fijos. Ejemplo: Nodo 1 Ux = 1, Uy = 2.")

n_rest = int(input("¿Cuántas restricciones hay? "))

for _ in range(n_rest):
    nodo = int(input("\n  Nodo: "))
    tipo = input("  Grado fijo (Ux/ Uy): ").lower()

    if tipo == "ux":
        fixed.append(2*(nodo-1))
    elif tipo == "uy":
        fixed.append(2*(nodo-1)+1)

free = [i for i in range(ndof) if i not in fixed]

# MATRICES REDUCIDAS
K_ff = K[np.ix_(free, free)]
F_f = F[free]

# ================================================================
# SOLUCIÓN
# ================================================================
u = np.zeros(ndof)
u_free = np.linalg.solve(K_ff, F_f)
u[free] = u_free

# ================================================================
# REACCIONES
# ================================================================
K_cf = K[np.ix_(fixed, free)]
R = K_cf.dot(u_free)

# ================================================================
# FUERZAS EN BARRAS
# ================================================================
forces = []

print("\n===== RESULTADOS =====\n")

print("\n--- DESPLAZAMIENTOS NODALES ---")
for i in range(n_nodos):
    print(f"Nodo {i+1}: Ux = {u[2*i]:.6f}, Uy = {u[2*i+1]:.6f}")

print("\n--- REACCIONES EN APOYOS ---")
for i, dof in enumerate(fixed):
    nodo = dof//2 + 1
    comp = "Ux" if dof % 2 == 0 else "Uy"
    print(f"Reacción en nodo {nodo} {comp}: {R[i]:.6f}")

print("\n--- FUERZAS AXIALES EN MIEMBROS ---")
for idx, (i, j) in enumerate(members):
    xi, yi = nodes[i]
    xj, yj = nodes[j]
    dx = xj - xi
    dy = yj - yi
    L = np.hypot(dx, dy)
    cx = dx / L
    cy = dy / L

    dof = [
        2*(i-1), 2*(i-1)+1,
        2*(j-1), 2*(j-1)+1
    ]

    u_local = u[dof]
    EA = EA_list[idx]

    T = (EA / L) * np.array([-cx, -cy, cx, cy]).dot(u_local)

    print(f"Miembro {i}-{j}: {T:.6f} (tensión + / compresión -)")
