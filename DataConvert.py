import csv

# Nombres de los archivos
input_file = 'data_venta.txt'
output_file = 'ventass.csv'
# Leer y procesar datos desde el archivo de texto
data = []
with open(input_file, 'r', encoding='latin-1') as f:
    for line in f:
        # Eliminar comillas simples y espacios en exceso
        parts = line.strip().replace("'", "").split(',')
        # Tomar solo los primeros 5 elementos (por si hay comas en la descripción)
        code = parts[0]
        date = parts[1].split()[0]  # Quitar hora
        item_code = parts[2]
        description = parts[3]
        quantity = parts[4]
        data.append([code, date, item_code, description, quantity])

# Escribir a archivo CSV
with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Code', 'Date', 'Item Code', 'Description', 'Quantity'])
    writer.writerows(data)

print(f"Archivo '{output_file}' generado exitosamente.")
