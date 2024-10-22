import re
import csv
import glob

def parse_file(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # Récupérer le nombre de slots
    slots_match = re.search(r'Total slots allocated (\d+)', content)
    slots = int(slots_match.group(1)) if slots_match else None

    # Récupérer tous les temps d'exécution
    time_matches = re.findall(r'Total execution time: ([\d.]+) seconds', content)
    execution_times = [float(time) for time in time_matches]

    return slots, execution_times

# Traiter plusieurs fichiers et collecter les résultats
files = glob.glob('chol_meas/weak_scalability/weak_*.txt')
results = []

for file in files:
    slots, execution_times = parse_file(file)
    if execution_times:
        results.append({
            'Processes': slots,
            'time_jacobi': execution_times[0],
            'time_gauss_seidel': execution_times[1] if len(execution_times) > 1 else None
        })

# Trier les résultats par nombre de processus croissants
results = sorted(results, key=lambda x: x['Processes'])

# Écrire les résultats dans un fichier CSV
csv_filename = 'chol_meas/weak_scalability_measurements.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['Processes', 'time_jacobi', 'time_gauss_seidel']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow(result)

print(f"Results saved to {csv_filename}")
