#!/usr/bin/env python3
import json

# Cargar número de chunks por part_file
with open("chunks_per_part.json") as f:
    parts = json.load(f)

# Generar lista de tareas reales
tasks = []
for part in sorted(parts.keys(), key=int):
    n_chunks = parts[part]
    for chunk in range(n_chunks):
        tasks.append((int(part), chunk))

# Guardar en archivo plano
with open("tasks.txt", "w") as f:
    for part, chunk in tasks:
        f.write(f"{part} {chunk}\n")

print(f"{len(tasks)} tasks escritas en tasks.txt")
