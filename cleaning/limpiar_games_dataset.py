import pandas as pd

# Corrige el encabezado de games.csv, donde "Discount" y "DLC count" 
# estaban unidos, lo que generaba una estructura incorrecta en las columnas.

INPUT_FILE = 'data/raw/games.csv'
FIXED_FILE = 'data/raw/games_fixed.csv'


print("Iniciando limpieza de games.csv...")

old_header = "DiscountDLC count"
new_header = "Discount,DLC count"


# Leer encabezado original
with open(INPUT_FILE, "r", encoding="utf-8", newline="") as file:
    header = file.readline()


# Comprobar que el problema existe
if old_header not in header:

    print("No se encontró el encabezado que necesita corrección.")
    print(f"Se esperaba encontrar: {old_header}")

else:

    # Corregir el encabezado
    header_fixed = header.replace(
        old_header,
        new_header,
        1
    )

    # Crear el nuevo CSV
    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as source, open(
        FIXED_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as target:

        # Saltar encabezado original
        source.readline()

        # Escribir encabezado corregido
        target.write(header_fixed)

        # Copiar el resto del dataset sin modificarlo
        for line in source:
            target.write(line)

    print(f"Archivo corregido: {FIXED_FILE}")


# Comprobar el resultado
df_fixed = pd.read_csv(
    FIXED_FILE,
    low_memory=False
)

print(f"\nFilas: {df_fixed.shape[0]}")
print(f"Columnas: {df_fixed.shape[1]}")

print("\nColumnas del dataset corregido:")
print(df_fixed.columns.tolist())

print("\nSample del dataset corregido:")
print(df_fixed.head())

print("\nLimpieza finalizada.")