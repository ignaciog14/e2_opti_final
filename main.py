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

# Continuar con el resto del modelo de optimización...
# [Aquí iría el resto de tu código para el modelo Gurobi]

# CONJUNTOS ----------------------------------------------------------------

R = regadores_df["regador"].unique().tolist()
N = zonas_df["zona"].unique().tolist()

T = capacidad_df["hora"].tolist()  
'''
H = 6  # Puedes cambiar esto según cuántos días estés modelando
D = list(range(H + 1))
'''
Th = no_permitidas_df["hora"].tolist()

print("Conjuntos listos")

# PARAMETROS ------------------------------------------------------------


# Zonas
# zona	area_m2 (An)	litros_prom (Jn)	max_horas (Ln)	agua_inicial (Qn)	agua_min (Aminn)	agua_max (Amaxn)	costo_replantar (C area n)
An = dict(zip(zonas_df["zona"], zonas_df["area_m2 (An)"]))
Jn = dict(zip(zonas_df["zona"], zonas_df["litros_prom (Jn)"]))
Ln = dict(zip(zonas_df["zona"], zonas_df["max_horas (Ln)"]))
Qn = dict(zip(zonas_df["zona"], zonas_df["agua_inicial (Qn)"]))
Aminn = dict(zip(zonas_df["zona"], zonas_df["agua_min (Aminn)"]))
Amaxn = dict(zip(zonas_df["zona"], zonas_df["agua_max (Amaxn)"]))
Carean = dict(zip(zonas_df["zona"], zonas_df["costo_replantar (C area n)"]))

# Regadores
#regador	area_cubre_m2 (Fr)	costo_instalacion (Cr)	costo_mant (Er)	costo_remocion (Sr)	eficiencia (βr)	litros_hora (Cant r)
Fr = dict(zip(regadores_df["regador"], regadores_df["area_cubre_m2 (Fr)"]))
Cr = dict(zip(regadores_df["regador"], regadores_df["costo_instalacion (Cr)"]))
Er = dict(zip(regadores_df["regador"], regadores_df["costo_mant (Er)"]))
Sr = dict(zip(regadores_df["regador"], regadores_df["costo_remocion (Sr)"]))
βr = dict(zip(regadores_df["regador"], regadores_df["eficiencia (βr)"]))
Cantr = dict(zip(regadores_df["regador"], regadores_df["litros_hora (Cant r)"]))

# Costos de activación riego r en zona n
#regador	zona	costo_activacion (Hrn)
Hrn = {(row["regador"], row["zona"]): row["costo_activacion (Hrn)"] for _, row in activaciones_df.iterrows()}

# Regadores iniciales
#regador	zona	cantidad_inicial (Rrn)
Rrn = {(row["regador"], row["zona"]): row["cantidad_inicial (Rrn)"] for _, row in iniciales_df.iterrows()}

# Capacidad máxima de agua por hora
#hora	capacidad_litros (Dt)
Dt = dict(zip(capacidad_df["hora"], capacidad_df["capacidad_litros (Dt)"]))

# Parametros globales

K = 2         # Costo por litro
Mbig = 10000   # Valor grande para restricciones
U = 5          # Días de instalación
U_minus = 1    # Días de remoción
Umax = 10      # Tiempo máximo entre instalación y remoción


print("Parametros listos")



# MODELO ----------------------------------------------------------------



m = Model("Optimizacion del sistema de riego PARQUE METROPOLITANO")
m.setParam("TimeLimit", 30 * 60)
m.setParam("MIPGap", 0.02)  # Permite parar con 2% de gap


# VARIABLES -------------------------------------------------------------

# Inventario de regadores r en zona n al tiempo t
Yrnt = m.addVars(R, N, T, vtype=GRB.INTEGER, name="Yrnt")

# Si se riega con regador r en zona n al tiempo t (variable binaria)
Zrnt = m.addVars(R, N, T, vtype=GRB.BINARY, name="Zrnt")

# Regadores comprados
Vrnt = m.addVars(R, N, T, vtype=GRB.INTEGER, name="Vrnt")

# Regadores quitados
Vminus_rnt = m.addVars(R, N, T, vtype=GRB.INTEGER, name="Vminus_rnt")

# Condición de regador r en zona n (binaria, puede significar si está activo)
Condrn = m.addVars(R, N, vtype=GRB.BINARY, name="Condrn")   # ver si es necesario agregar t

# Litros de agua regados con regador r en zona n al tiempo t
Xnrt = m.addVars(N, R, T, vtype=GRB.CONTINUOUS, name="Xnrt")

# Cantidad de agua disponible en el suelo en zona n al tiempo t
Int = m.addVars(N, T, vtype=GRB.CONTINUOUS, name="Int")

# Error de agua (por exceso o falta de riego)
Wnt = m.addVars(N, T, vtype=GRB.CONTINUOUS, name="Wnt")

# FUNCIÓN OBJETIVO -----------------------------------------------------

# Minimizar costos totales: instalación, remoción, mantenimiento, activación, agua y errores
obj = quicksum(
    Vrnt[r, n, t] * Cr[r] +                # Costo de comprar regadores
    Vminus_rnt[r, n, t] * Sr[r] +           # Costo de remover regadores
    Yrnt[r, n, t] * Er[r] +                 # Costo de mantenimiento
    (Zrnt[r, n, t] * Hrn.get((r, n), 0) if (r, n) in Hrn else 0) +           # Costo de activación
    Xnrt[n, r, t] * K +                     # Costo del agua utilizada
    Wnt[n, t] * Carean[n]                   # Costo por errores de riego
    for r in R for n in N for t in T
)

# Antes de crear el modelo, verifica los parámetros
def check_for_nan_inf(data_dict, name):
    for key, value in data_dict.items():
        if pd.isna(value) or pd.isnull(value) or value == float('inf'):
            print(f"Error en {name}[{key}]: Valor inválido ({value})")
            return False
    return True

# Verificar todos los parámetros
params_to_check = [
    (An, "An"), (Jn, "Jn"), (Ln, "Ln"), (Qn, "Qn"), 
    (Aminn, "Aminn"), (Amaxn, "Amaxn"), (Carean, "Carean"),
    (Fr, "Fr"), (Cr, "Cr"), (Er, "Er"), (Sr, "Sr"), 
    (βr, "βr"), (Cantr, "Cantr"), (Hrn, "Hrn"), (Rrn, "Rrn"), 
    (Dt, "Dt")
]

for param, name in params_to_check:
    if not check_for_nan_inf(param, name):
        print(f"Corrige los valores en el Excel para {name}")
        exit()

# Verificación adicional para Hrn y Rrn (diccionarios anidados)
print("\nVerificando combinaciones faltantes:")
for r in R:
    for n in N:
        if (r, n) not in Hrn:
            print(f"Falta costo activación para {r} en {n} - asignando valor por defecto 100")
            Hrn[(r, n)] = 100  # Valor por defecto
        
        if (r, n) not in Rrn:
            print(f"Falta cantidad inicial para {r} en {n} - asignando 0")
            Rrn[(r, n)] = 0

# Verificar que no haya NaN/Inf en los parámetros globales
global_params = {'K': K, 'Mbig': Mbig, 'U': U, 'U_minus': U_minus, 'Umax': Umax}
for name, value in global_params.items():
    if pd.isna(value) or pd.isnull(value) or value == float('inf'):
        print(f"Error en parámetro global {name}: Valor inválido ({value})")
        exit()

print("\nTodos los parámetros verificados - sin valores NaN o Inf")

m.setObjective(obj, GRB.MINIMIZE)

print("Función objetivo establecida")

# RESTRICCIONES --------------------------------------------------------

print("Agregando restricciones...")

# 1. Balance hídrico mínimo (no permitir que el agua en suelo sea menor que Aminn - error)
m.addConstrs(
    (Int[n, t] >= Aminn[n] - Wnt[n, t] 
     for n in N for t in T),
    name="min_water"
)
print("Restricción de balance hídrico mínimo agregada")

# 2. Balance hídrico máximo (no permitir que el agua en suelo sea mayor que Amaxn + error)
m.addConstrs(
    (Int[n, t] <= Amaxn[n] + Wnt[n, t] 
     for n in N for t in T),
    name="max_water"
)
print("Restricción de balance hídrico máximo agregada")

# 3. Balance de agua para t > 0

m.addConstrs(
    (Int[n, t] == quicksum(Xnrt[n, r, t] * βr[r] for r in R) + Int[n, t-1] - Jn[n]
    for n in N for t in T if t > 0),
    name="water_balance"
)
print("Restricción de balance de agua para t > 0 agregada")

# 4. Condición inicial de agua (t=0)
m.addConstrs(
    (Int[n, 0] == quicksum(Xnrt[n, r, 0] * βr[r] for r in R) - Jn[n] + Qn[n]
    for n in N),
    name="initial_water"
)
print("Restricción de condición inicial de agua agregada")  

# 5. Relación entre Zrnt y Yrnt (solo se puede regar si hay regadores disponibles)
m.addConstrs(
    (Zrnt[r, n, t] <= Mbig * Yrnt[r, n, t]
     for r in R for n in N for t in T),
    name="Z_Y_relation"
)
print("Restricción de relación entre Zrnt y Yrnt agregada")

# 6. Activar Zrnt si se usa el riego r
m.addConstrs(
    (Mbig * Zrnt[r, n, t] >= Xnrt[n, r, t]
     for r in R for n in N for t in T),
    name="activate_Z"
)
print("Restricción de activación de Zrnt agregada")

# 7. Control de inventario de regadores (para t > 0)
m.addConstrs(
    (Yrnt[r, n, t] == Yrnt[r, n, t-1] + Vrnt[r, n, t-U] - Vminus_rnt[r, n, t-U_minus]
     for r in R for n in N for t in range(Umax, len(T))),
    name="inventory_control"
)
print("Restricción de control de inventario de regadores agregada")

# 8. Inventario inicial de regadores (para t < Umax)
m.addConstrs(
    (Yrnt[r, n, t] == Rrn[r, n]
     for r in R for n in N for t in range(U)),
    name="initial_inventory"
)
print("Restricción de inventario inicial de regadores agregada")

# 9. Prohibir riego durante horas no permitidas
m.addConstrs(
    (quicksum(Zrnt[r, n, t] for r in R for n in N) == 0
     for t in Th),
    name="no_irrigation_hours"
)
print("Restricción de prohibición de riego durante horas no permitidas agregada")

# 10. Relación entre regadores disponibles y capacidad de riego
m.addConstrs(
    (Yrnt[r, n, t] * Cantr[r] >= Xnrt[n, r, t]
     for r in R for n in N for t in T),
    name="regador_capacity"
)
print("Restricción de relación entre regadores disponibles y capacidad de riego agregada")

# 11. Restricción de capacidad total de agua por hora
m.addConstrs(
    (quicksum(Xnrt[n, r, t] for n in N for r in R) <= Dt[t]
     for t in T),
    name="total_water_capacity"
)
print("Restricción de capacidad total de agua por hora agregada")

# 12. No superar horas máximas consecutivas de riego por zona
for n in N:
    max_horas = Ln[n]
    for t in range(int(len(T)) - int(max_horas)):
        m.addConstr(
            quicksum(Zrnt[r, n, k] for r in R for k in range(t, int(t + max_horas + 1))) <= max_horas,
            name=f"max_consec_hours_{n}_{t}"
        )

print("Restricciones agregadas")

# OPTIMIZAR ------------------------------------------------------------

m.optimize()

# RESULTADOS -----------------------------------------------------------

if m.status == GRB.OPTIMAL:
    print("\nSolución óptima encontrada")
    print(f"Costo total: {m.objVal:,.2f} CLP")
    
    # Mostrar compras de regadores
    print("\nCompras de regadores:")
    for t in T:
        for r in R:
            for n in N:
                if Vrnt[r, n, t].X > 0:
                    print(f"Hora {t}: Comprar {Vrnt[r, n, t].X} regadores {r} para zona {n}")
    
    # Mostrar remociones de regadores
    print("\nRemociones de regadores:")
    for t in T:
        for r in R:
            for n in N:
                if Vminus_rnt[r, n, t].X > 0:
                    print(f"Hora {t}: Remover {Vminus_rnt[r, n, t].X} regadores {r} de zona {n}")
    
    # Mostrar agua utilizada por hora
    print("\nAgua utilizada por hora:")
    agua_utilizada_total = 0
    for t in T:
        total_agua = sum(Xnrt[n, r, t].X for n in N for r in R)
        agua_utilizada_total += total_agua
        print(f"Hora {t}: {total_agua:,.2f} litros")

    # mostrar agua total utilizada
    #print(f"\nAgua total utilizada: {agua_utilizada_total:,.2f} litros")
    
    # Mostrar errores de riego
    print("\nErrores de riego (excesos o déficits):")
    for n in N:
        for t in T:
            if Wnt[n, t].X > 0:
                print(f"Zona {n}, hora {t}: Error de {Wnt[n, t].X:.2f} litros")

    def mostrar_variable(nombre, variable):
        print(f"\n--- {nombre} ---")
        hay_valores = False
        for idx in variable.keys():
            val = variable[idx].X
            if abs(val) > 1e-6:  # mostrar solo si es significativo
                print(f"{nombre}[{idx}] = {val}")
                hay_valores = True
        if not hay_valores:
            print("Todos los valores son cero.")

    mostrar_variable("Yrnt", Yrnt)
    #mostrar_variable("Zrnt", Zrnt)
    mostrar_variable("Vrnt", Vrnt)
    mostrar_variable("Vminus_rnt", Vminus_rnt)
    #mostrar_variable("Condrn", Condrn)
    #mostrar_variable("Xnrt", Xnrt)
    #mostrar_variable("Int", Int)
    mostrar_variable("Wnt", Wnt)
    
else:
    print("No se encontró solución óptima")
    print(f"Estado del modelo: {m.status}")