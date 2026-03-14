import os
import ramanspy as rp

base_path = os.path.expanduser('~/Code/Data_SH')

subsets = [
    'LR-Raman',
    'excellent_oriented',
    'excellent_unoriented',
    'fair_oriented',
    'fair_unoriented',
    'poor_unoriented',
    'unrated_oriented',
    'unrated_unoriented',
]

print(f"{'Subset':<30} {'Spectra':>10} {'Classes':>10} {'Avg/Class':>10}")
print("-" * 65)

for subset in subsets:
    path = os.path.join(base_path, subset)

    if not os.path.exists(path):
        print(f"{subset:<30} {'NOT FOUND':>10}")
        continue

    try:
        spectra, metadata = rp.datasets.rruff(path, download=False)
        labels = [m['##NAMES'].split(',')[0].strip() for m in metadata]
        n_spectra = len(spectra)
        n_classes = len(set(labels))
        avg = n_spectra / n_classes

        print(f"{subset:<30} {n_spectra:>10} {n_classes:>10} {avg:>10.1f}")

    except Exception as e:
        print(f"{subset:<30} ERROR: {e}")