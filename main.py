"""
Comentaris para hacer:
1. Umax no se esta utilizando. / Solucionado




"""


from gurobipy import GRB, Model, quicksum
import pandas as pd
import os

# Configuración inicial
archivo = "datos.xlsx"
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, archivo)

# Verificar existencia del archivo
if not os.path.exists(file_path):
    raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

# Listar hojas disponibles
print("\nVerificando hojas del archivo Excel...")
hojas_disponibles = pd.ExcelFile(file_path).sheet_names
print("Hojas encontradas:", hojas_disponibles)

# Nombres de hojas (ajustar según lo que muestre el print anterior)
nombres_hojas = {
    'zonas': None,  # Se determinará automáticamente
    'regadores': None,
    'costos_activacion': None,
    'regadores_iniciales': None,
    'capacidad_agua': None,
    'horas_no_permitidas': None,
    'parametros_globales': None
}

# Buscar coincidencias aproximadas para cada hoja
for hoja_real in hojas_disponibles:
    hoja_lower = hoja_real.lower()
    if 'zona' in hoja_lower and nombres_hojas['zonas'] is None:
        nombres_hojas['zonas'] = hoja_real
    elif 'regador' in hoja_lower and 'inicial' not in hoja_lower and nombres_hojas['regadores'] is None:
        nombres_hojas['regadores'] = hoja_real
    elif 'activacion' in hoja_lower and nombres_hojas['costos_activacion'] is None:
        nombres_hojas['costos_activacion'] = hoja_real
    elif 'inicial' in hoja_lower and nombres_hojas['regadores_iniciales'] is None:
        nombres_hojas['regadores_iniciales'] = hoja_real
    elif 'capacidad' in hoja_lower and nombres_hojas['capacidad_agua'] is None:
        nombres_hojas['capacidad_agua'] = hoja_real
    elif 'permitida' in hoja_lower and nombres_hojas['horas_no_permitidas'] is None:
        nombres_hojas['horas_no_permitidas'] = hoja_real
    elif 'parametro' in hoja_lower or 'global' in hoja_lower and nombres_hojas['parametros_globales'] is None:
        nombres_hojas['parametros_globales'] = hoja_real
    elif 'dias' in hoja_lower and nombres_hojas.get('dias') is None:
        nombres_hojas['dias'] = hoja_real

# Mostrar mapeo de hojas
print("\nNombres de hojas detectados:")
for key, value in nombres_hojas.items():
    print(f"{key}: {value}")

# Cargar datos usando los nombres detectados
try:
    zonas_df = pd.read_excel(file_path, sheet_name=nombres_hojas['zonas']).dropna(subset=['zona'])
    regadores_df = pd.read_excel(file_path, sheet_name=nombres_hojas['regadores']).dropna(subset=['regador'])
    activaciones_df = pd.read_excel(file_path, sheet_name=nombres_hojas['costos_activacion'])
    iniciales_df = pd.read_excel(file_path, sheet_name=nombres_hojas['regadores_iniciales'])
    capacidad_df = pd.read_excel(file_path, sheet_name=nombres_hojas['capacidad_agua'])
    no_permitidas_df = pd.read_excel(file_path, sheet_name=nombres_hojas['horas_no_permitidas'])
    param_globales_df = pd.read_excel(file_path, sheet_name=nombres_hojas['parametros_globales'], header=None)
    # tabla capacidad_agua

    # tabla dias
    dias_df  =pd.read_excel(file_path, sheet_name=nombres_hojas['dias'])
    
    print("\n¡Datos cargados correctamente!")
    
except Exception as e:
    print(f"\nError al cargar datos: {str(e)}")
    print("\nPor favor verifica:")
    print("1. Que el archivo Excel tenga las hojas con los nombres esperados")
    print("2. Que los nombres de columnas coincidan exactamente")
    print("\nHojas disponibles:", hojas_disponibles)
    exit()

# Verificación básica de datos cargados
print("\nResumen de datos cargados:")
print(f"- Zonas: {len(zonas_df)} registros")
print(f"- Regadores: {len(regadores_df)} tipos")
print(f"- Costos activación: {len(activaciones_df)} registros")
print(f"- Regadores iniciales: {len(iniciales_df)} registros")
print(f"- Capacidad agua: {len(capacidad_df)} horarios")
print(f"- Horas no permitidas: {len(no_permitidas_df)} registros")
print(f"- Parámetros globales: {len(param_globales_df)} líneas")

# CONJUNTOS ----------------------------------------------------------------
R = regadores_df["regador"].unique().tolist()  # Tipos de regadores
N = zonas_df["zona"].unique().tolist()        # Zonas del parque
T = capacidad_df["hora"].unique().tolist()                       # Horas del día (0-23)
                           # Días del horizonte de planificación (7 días)
D = dias_df["dias"].tolist()  # Días del horizonte de planificación (7 días)
Th = no_permitidas_df["hora"].tolist()        # Horas no permitidas para riego

print("\nConjuntos definidos:")
print(f"- Tipos de regadores (R): {R}")
print(f"- Zonas (N): {N}")
print(f"- Horas del día (T): {T}")
print(f"- Horas no permitidas (Th): {Th}")
print(f"- Días del horizonte de planificación (D): {D}")
#horas


# PARÁMETROS ------------------------------------------------------------
# Zonas
An = dict(zip(zonas_df["zona"], zonas_df["area_m2 (An)"]))
Jn = dict(zip(zonas_df["zona"], zonas_df["litros_prom (Jn)"]))
Ln = dict(zip(zonas_df["zona"], zonas_df["max_horas (Ln)"]))
Qn = dict(zip(zonas_df["zona"], zonas_df["agua_inicial (Qn)"]))
Aminn = dict(zip(zonas_df["zona"], zonas_df["agua_min (Aminn)"]))
Amaxn = dict(zip(zonas_df["zona"], zonas_df["agua_max (Amaxn)"]))
Carean = dict(zip(zonas_df["zona"], zonas_df["costo_replantar (C area n)"]))

# Regadores
Fr = dict(zip(regadores_df["regador"], regadores_df["area_cubre_m2 (Fr)"]))
Cr = dict(zip(regadores_df["regador"], regadores_df["costo_instalacion (Cr)"]))
Er = dict(zip(regadores_df["regador"], regadores_df["costo_mant (Er)"]))
Sr = dict(zip(regadores_df["regador"], regadores_df["costo_remocion (Sr)"]))
βr = dict(zip(regadores_df["regador"], regadores_df["eficiencia (βr)"]))
Cantr = dict(zip(regadores_df["regador"], regadores_df["litros_hora (Cant r)"])) 

# Costos de activación 
Hrn = {(row["regador"], row["zona"]): row["costo_activacion (Hrn)"] for _, row in activaciones_df.iterrows()}

# Regadores iniciales
Rrn = {(row["regador"], row["zona"]): row["cantidad_inicial (Rrn)"] for _, row in iniciales_df.iterrows()}

# Capacidad máxima de agua por hora capacidad_litros (Dt)

Dt = dict(zip(capacidad_df["hora"], capacidad_df["capacidad_litros (Dt)"]))

# Parámetros globales
K = 2       # Costo por litro de agua
Mbig = 1000   # Valor grande para restricciones
U = 5       # Días para instalación
U_minus = 1  # Días para remoción


print("\nParámetros cargados correctamente")

# Verificación de parámetros faltantes
print("\nVerificando combinaciones faltantes:")
for r in R:
    for n in N:
        if (r, n) not in Hrn:
            print(f"Falta costo activación para {r} en {n} - asignando valor por defecto 100")
            Hrn[(r, n)] = 100  # Valor por defecto
        
        if (r, n) not in Rrn:
            print(f"Falta cantidad inicial para {r} en {n} - asignando 0")
            Rrn[(r, n)] = 0

# MODELO ----------------------------------------------------------------
m = Model("Optimizacion del sistema de riego PARQUE METROPOLITANO")
m.setParam("TimeLimit", 30 * 60)  # Límite de tiempo de 30 minutos
m.setParam("MIPGap", 0.02)        # Gap de optimalidad del 2%

# VARIABLES DE DECISIÓN -------------------------------------------------
Y = {}  # Y[r,n,d]: Inventario de regadores r en zona n al final del día d
Z = {}  # Z[r,n,t,d]: 1 si se riega con r en zona n a hora t en día d
V = {}  # V[r,n,d]: Regadores r comprados en zona n en día d
V_minus = {}  # V_minus[r,n,d]: Regadores r quitados en zona n en día d
X = {}  # X[n,r,t,d]: Litros de agua regados en zona n con r a hora t en día d
I = {}  # I[n,t,d]: Agua disponible en el suelo en zona n a hora t en día d
W = {}  # W[n,t,d]: Error de agua en litros para zona n a hora t en día d
G = {}  # G[n,t,d]: 1 si se riega en zona n a hora t en día d

# Inicializar variables
for r in R:
    for n in N:
        for d in D:
            Y[r,n,d] = m.addVar(vtype=GRB.INTEGER, name=f"Y_{r}_{n}_{d}")
            V[r,n,d] = m.addVar(vtype=GRB.INTEGER, name=f"V_{r}_{n}_{d}")
            V_minus[r,n,d] = m.addVar(vtype=GRB.INTEGER, name=f"Vminus_{r}_{n}_{d}")
            
            for t in T:
                Z[r,n,t,d] = m.addVar(vtype=GRB.BINARY, name=f"Z_{r}_{n}_{t}_{d}")
                X[n,r,t,d] = m.addVar(vtype=GRB.CONTINUOUS, name=f"X_{n}_{r}_{t}_{d}")

for n in N:
    for t in T:
        for d in D:
            I[n,t,d] = m.addVar(vtype=GRB.CONTINUOUS, name=f"I_{n}_{t}_{d}")
            W[n,t,d] = m.addVar(vtype=GRB.CONTINUOUS, name=f"W_{n}_{t}_{d}")
            G[n,t,d] = m.addVar(vtype=GRB.BINARY, name=f"G_{n}_{t}_{d}")

print("\nVariables de decisión creadas")

# FUNCIÓN OBJETIVO -----------------------------------------------------
# Minimizar costos totales: instalación, remoción, mantenimiento, activación, agua y errores
obj = quicksum(
    V[r,n,d] * Cr[r] +                  # Costo de comprar regadores
    V_minus[r,n,d] * Sr[r] +            # Costo de remover regadores
    Y[r,n,d] * Er[r] +                  # Costo de mantenimiento
    Z[r,n,t,d] * Hrn[r,n] +             # Costo de activación
    X[n,r,t,d] * K +                    # Costo del agua utilizada
    W[n,t,d] * Carean[n]                # Costo por errores de riego
    for r in R for n in N for t in T for d in D
)

m.setObjective(obj, GRB.MINIMIZE)
print("\nFunciÓn objetivo establecida")

# RESTRICCIONES --------------------------------------------------------
print("\nAgregando restricciones...")

# 1. Balance hídrico mínimo (no permitir que el agua en suelo sea menor que Aminn - error)
m.addConstrs(
    (I[n,t,d] >= Aminn[n] - W[n,t,d] 
    for n in N for t in T for d in D),
    name="min_water"
)

# 2. Balance hídrico máximo (no permitir que el agua en suelo sea mayor que Amaxn + error)
m.addConstrs(
    (I[n,t,d] <= Amaxn[n] + W[n,t,d] 
    for n in N for t in T for d in D),
    name="max_water"
)

# 3. Balance de agua para t > 0, d > 0
m.addConstrs(
    (I[n,t,d] == quicksum(X[n,r,t,d] * βr[r] for r in R) + I[n,t-1,d] - Jn[n]
    for n in N for t in T if t > 0 for d in D),
    name="water_balance_t>0"
)

# 4. Balance de agua para t=0, d > 0
m.addConstrs(
    (I[n,0,d] == quicksum(X[n,r,23,d-1] * βr[r] for r in R) + I[n,23,d-1] - Jn[n]
    for n in N for d in D if d > 1),
    name="water_balance_d>0"
)

# 5. Condición inicial de agua (d=0, t=0)
m.addConstrs(
    (I[n,0,1] == quicksum(X[n,r,0,1] * βr[r] for r in R) - Jn[n] + Qn[n]
    for n in N),
    name="initial_water"
)

# 6. Relación entre Z y Y (solo se puede regar si hay regadores disponibles)
m.addConstrs(
    (Z[r,n,t,d] <= Y[r,n,d]
    for r in R for n in N for t in T for d in D),
    name="Z_Y_relation"
)

# 7. Activar Z si se usa el riego r
m.addConstrs(
    (Mbig * Z[r,n,t,d] >= X[n,r,t,d]
    for r in R for n in N for t in T for d in D),
    name="activate_Z"
)

# 8. Control de inventario de regadores (para d >= U)
m.addConstrs(
    (Y[r,n,d] == Y[r,n,d-1] + V[r,n,d-U] - V_minus[r,n,d-U_minus]
    for r in R for n in N for d in D if d > U),
    name="inventory_control_d>=U"
)

# 9. Control de inventario de regadores (para U_minus <= d < U)
m.addConstrs(
    (Y[r,n,d] == Y[r,n,d-1] - V_minus[r,n,d-U_minus]
    for r in R for n in N for d in D if U_minus < d < U),
    name="inventory_control_Uminus<=d<U"
)

# 10. Inventario inicial de regadores (para d < U_minus)
m.addConstrs(
    (Y[r,n,d] == Rrn[r,n]
    for r in R for n in N for d in D if d < U_minus),
    name="initial_inventory"
)

# 11. Prohibir riego durante horas no permitidas
m.addConstrs(
    (quicksum(Z[r,n,t,d] for r in R for n in N) == 0
    for t in Th for d in D),
    name="no_irrigation_hours"
)

# 12. Relación entre regadores disponibles y capacidad de riego
m.addConstrs(
    (Y[r,n,d] * Cantr[r] >= X[n,r,t,d]
    for r in R for n in N for t in T for d in D),
    name="regador_capacity"
)

# 13. Restricción de capacidad total de agua por hora
m.addConstrs(
    (quicksum(X[n,r,t,d] for n in N for r in R) <= Dt[t]
    for t in T for d in D),
    name="total_water_capacity"
)

# 14. Definir variable G (si se riega en zona n a hora t en día d)
m.addConstrs(
    (quicksum(Z[r,n,t,d] for r in R) <= Mbig * G[n,t,d]
    for n in N for t in T for d in D),
    name="define_G"
)

# 15. Garantizar que el área regada cubra al menos el área necesaria
m.addConstrs(
    (quicksum(Y[r,n,d] * Fr[r] for r in R) >= An[n] * G[n,t,d]
    for n in N for t in T for d in D),
    name="area_coverage"
)

# 16. No superar horas máximas consecutivas de riego por zona
for n in N:
    max_horas = int(Ln[n])
    for d in D:
        # Para horas normales dentro del día
        for t in range(24 - max_horas):
            m.addConstr(
                quicksum(G[n,k,d] for k in range(t, t + max_horas + 1)) <= max_horas,
                name=f"max_consec_hours_{n}_{t}_{d}"
            )

# Restricción 17: No superar las horas máximas consecutivas de riego entre días
m.addConstrs(
    (quicksum(G[n,k,d] for k in range(23 - int(Ln[n]) + t, 24)) + 
     quicksum(G[n,k,d+1] for k in range(0, t+2)) <= int(Ln[n])
     for n in N for d in D if d < max(D) for t in range(int(Ln[n]) + 1)),
    name="max_consecutive_hours_cross_days"
)

# Restricción 18: Capacidad máxima del sistema de agua por hora
m.addConstrs(
    (quicksum(X[n,r,t,d] for n in N for r in R) <= Dt[t]
     for t in T for d in D),
    name="max_water_capacity_per_hour"
)            

print("\nTodas las restricciones agregadas")

# OPTIMIZAR ------------------------------------------------------------
m.optimize()

# RESULTADOS -----------------------------------------------------------
if m.status == GRB.OPTIMAL:
    print("\nSolución óptima encontrada")
    print(f"Costo total: {m.objVal:,.2f} CLP")
    
    # Mostrar compras de regadores
    print("\nCompras de regadores:")
    compras = [(d, r, n, V[r,n,d].X) for r in R for n in N for d in D if V[r,n,d].X > 0]
    if compras:
        for d, r, n, qty in sorted(compras, key=lambda x: (x[0], x[1], x[2])):
            print(f"Día {d}: Comprar {qty:.0f} regadores {r} para zona {n}")
    else:
        print("No se recomiendan compras de regadores")
    
    # Mostrar remociones de regadores
    print("\nRemociones de regadores:")
    remociones = [(d, r, n, V_minus[r,n,d].X) for r in R for n in N for d in D if V_minus[r,n,d].X > 0]
    if remociones:
        for d, r, n, qty in sorted(remociones, key=lambda x: (x[0], x[1], x[2])):
            print(f"Día {d}: Remover {qty:.0f} regadores {r} de zona {n}")
    else:
        print("No se recomiendan remociones de regadores")
    
    # Mostrar agua utilizada por día y hora
    print("\nAgua utilizada por día y hora:")
    for d in D:
        print(f"\nDía {d}:")
        for t in T:
            total_agua = sum(X[n,r,t,d].X for n in N for r in R)
            if total_agua > 0:
                print(f"  Hora {t}: {total_agua:,.2f} litros")
    
    # Mostrar errores de riego
    print("\nErrores de riego (excesos o déficits):")
    errores = [(d, n, t, W[n,t,d].X) for n in N for t in T for d in D if W[n,t,d].X > 0]
    if errores:
        for d, n, t, error in sorted(errores, key=lambda x: (x[0], x[1], x[2])):
            print(f"Día {d}, Zona {n}, Hora {t}: Error de {error:.2f} litros")
    else:
        print("No se detectaron errores de riego")
    
    # Mostrar resumen de costos
    print("\nDesglose de costos:")
    costos = {
        "Instalación": sum(V[r,n,d].X * Cr[r] for r in R for n in N for d in D),
        "Remoción": sum(V_minus[r,n,d].X * Sr[r] for r in R for n in N for d in D),
        "Mantenimiento": sum(Y[r,n,d].X * Er[r] for r in R for n in N for d in D),
        "Activación": sum(Z[r,n,t,d].X * Hrn[r,n] for r in R for n in N for t in T for d in D),
        "Agua": sum(X[n,r,t,d].X * K for n in N for r in R for t in T for d in D),
        "Errores": sum(W[n,t,d].X * Carean[n] for n in N for t in T for d in D)
    }
    
    for concepto, valor in costos.items():
        print(f"{concepto}: {valor:,.2f} CLP")
    
else:
    print("\nNo se encontró solución óptima")
    print(f"Estado del modelo: {m.status}")
    if m.status == GRB.TIME_LIMIT:
        print("Se alcanzó el límite de tiempo. Solución actual:")
        print(f"Mejor cota inferior: {m.ObjBound:,.2f}")
        print(f"Mejor solución encontrada: {m.ObjVal:,.2f}")
    elif m.status == GRB.INFEASIBLE:
        print("El modelo es infactible. Considera revisar las restricciones.")
    elif m.status == GRB.UNBOUNDED:
        print("El modelo es no acotado. Revisa la función objetivo.")

# Exportar resultados a Excel
try:
    resultados = []
    for n in N:
        for r in R:
            for d in D:
                for t in T:
                    if X[n,r,t,d].X > 0 or Z[r,n,t,d].X > 0:
                        resultados.append({
                            "Zona": n,
                            "Regador": r,
                            "Día": d,
                            "Hora": t,
                            "Agua (litros)": X[n,r,t,d].X,
                            "Regando": Z[r,n,t,d].X,
                            "Regadores disponibles": Y[r,n,d].X
                        })
    
    df_resultados = pd.DataFrame(resultados)
    output_path = os.path.join(script_dir, "resultados_riego.xlsx")
    df_resultados.to_excel(output_path, index=False)
    print(f"\nResultados exportados a {output_path}")
    
except Exception as e:
    print(f"\nError al exportar resultados: {str(e)}")