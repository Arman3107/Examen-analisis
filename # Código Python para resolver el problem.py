import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Método de Rigidez 2D - Armaduras", layout="wide")

st.title("🧱 Método de Rigidez 2D para Armaduras (Streamlit)")

st.write("""
Aplicación interactiva para resolver armaduras por el **Método de Rigidez**.
Ingrese nodos, barras, apoyos y cargas, y obtenga desplazamientos, reacciones y fuerzas internas.
""")

# -----------------------------------------------------
# Sección 1: Datos de nodos
# -----------------------------------------------------

st.header("1️⃣ Nodos (coordenadas)")

nodes_df = st.data_editor(
    pd.DataFrame(
        {"Nodo": [1, 2], "x": [0.0, 4.0], "y": [0.0, 0.0]}
    ),
    num_rows="dynamic",
    key="nodes"
)

# -----------------------------------------------------
# Sección 2: Barras
# -----------------------------------------------------

st.header("2️⃣ Barras (conexiones y EA)")

bars_df = st.data_editor(
    pd.DataFrame(
        {"Barra": [1], "i": [1], "j": [2], "EA": [200000]}
    ),
    num_rows="dynamic",
    key="bars"
)

# -----------------------------------------------------
# Sección 3: Restricciones
# -----------------------------------------------------

st.header("3️⃣ Restricciones (Apoyos)")

supports_df = st.data_editor(
    pd.DataFrame(
        {"Nodo": [1], "Ux_fijo": [True], "Uy_fijo": [True]}
    ),
    num_rows="dynamic",
    key="supports"
)

# -----------------------------------------------------
# Sección 4: Cargas
# -----------------------------------------------------

st.header("4️⃣ Cargas nodales")

loads_df = st.data_editor(
    pd.DataFrame(
        {"Nodo": [2], "Fx": [0.0], "Fy": [-100.0]}
    ),
    num_rows="dynamic",
    key="loads"
)

# -----------------------------------------------------
# Función del método de rigidez
# -----------------------------------------------------

def solve_truss(nodes, bars, supports, loads):
    # Convert DataFrames to arrays
    coords = {int(n): np.array([float(x), float(y)]) for n, x, y in nodes}
    num_nodes = len(coords)
    dof = 2 * num_nodes

    # Global stiffness matrix
    K = np.zeros((dof, dof))

    # Assemble bars
    for _, i, j, EA in bars:
        i = int(i); j = int(j)
        xi, yi = coords[i]
        xj, yj = coords[j]
        L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
        c = (xj - xi) / L
        s = (yj - yi) / L

        k_local = (EA / L) * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s]
        ])

        index = [
            2*(i-1), 2*(i-1)+1,
            2*(j-1), 2*(j-1)+1
        ]

        for a in range(4):
            for b in range(4):
                K[index[a], index[b]] += k_local[a, b]

    # Load vector
    F = np.zeros(dof)
    for node, Fx, Fy in loads:
        F[2*(node-1)] = Fx
        F[2*(node-1)+1] = Fy

    # Apply supports
    fixed_dofs = []
    for node, ux, uy in supports:
        if ux:
            fixed_dofs.append(2*(node-1))
        if uy:
            fixed_dofs.append(2*(node-1)+1)

    free = [i for i in range(dof) if i not in fixed_dofs]

    Kff = K[np.ix_(free, free)]
    Ff = F[free]

    # Solve
    Uf = np.linalg.solve(Kff, Ff)

    U = np.zeros(dof)
    U[free] = Uf

    # Reactions
    R = K @ U - F

    # Bar forces
    bar_forces = []
    for _, i, j, EA in bars:
        i = int(i); j = int(j)
        xi, yi = coords[i]
        xj, yj = coords[j]
        L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
        c = (xj - xi) / L
        s = (yj - yi) / L

        u_i = U[2*(i-1):2*(i-1)+2]
        u_j = U[2*(j-1):2*(j-1)+2]

        axial = EA / L * np.array([-c, -s, c, s]) @ np.hstack([u_i, u_j])
        bar_forces.append(axial)

    return U, R, bar_forces


# -----------------------------------------------------
# Botón para calcular
# -----------------------------------------------------

st.header("5️⃣ Resolver armadura")

if st.button(" Calcular ", type="primary"):

    U, R, bar_forces = solve_truss(
        nodes_df.values,
        bars_df.values,
        supports_df.values,
        loads_df.values
    )

    st.success("Cálculo completado")

    st.subheader("Desplazamientos nodales")
    st.write(U.reshape(-1, 2))

    st.subheader("Reacciones en apoyos")
    st.write(R.reshape(-1, 2))

    st.subheader("Fuerzas axiales en barras")
    st.write(bar_forces)


