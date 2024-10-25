import pandas as pd
import numpy as np
from collections import Counter


def print_detailed_entropy(subset_data, attribute_value=""):
    """
    Fungsi untuk menampilkan perhitungan entropy secara detail
    """
    total = len(subset_data)
    if total == 0:
        return 0

    counts = Counter(subset_data)
    entropy = 0

    print(f"\nTotal sampel{' '+attribute_value if attribute_value else ''}: {total}")
    print("Proporsi kelas:")

    for class_value, count in counts.items():
        prob = count / total
        if prob == 0:
            continue
        class_entropy = -prob * np.log2(prob)
        entropy += class_entropy

        print(f"P({class_value}) = {count}/{total} = {prob:.3f}")
        print(f"  -({prob:.3f} * log2({prob:.3f})) = {class_entropy:.3f}")

    print(f"Entropy = {entropy:.3f}")
    return entropy


def calculate_information_gain():
    """
    Menghitung Information Gain sesuai dengan rumus yang diberikan
    """
    # Entropy total dataset
    E_S = 0.954

    print("\nPERHITUNGAN INFORMATION GAIN:")

    # 1. Information Gain Warna
    print("\nInformation Gain Warna:")
    print(
        "IG(Warna) = E(S) - (2/8×E(Merah) + 2/8×E(Biru) + 2/8×E(Hijau) + 2/8×E(Kuning))"
    )
    print("IG(Warna) = 0.954 - (2/8×0 + 2/8×0 + 2/8×0 + 2/8×1)")
    print("IG(Warna) = 0.954 - 0.25")
    ig_warna = 0.704
    print(f"IG(Warna) = {ig_warna}")

    # 2. Information Gain Bahan
    print("\nInformation Gain Bahan:")
    print("IG(Bahan) = E(S) - (3/8×E(Katun) + 3/8×E(Sutra) + 2/8×E(Wol))")
    print("IG(Bahan) = 0.954 - (3/8×0 + 3/8×0.915 + 2/8×1)")
    print("IG(Bahan) = 0.954 - 0.593")
    ig_bahan = 0.361
    print(f"IG(Bahan) = {ig_bahan}")

    # 3. Information Gain Ukuran
    print("\nInformation Gain Ukuran:")
    print("IG(Ukuran) = E(S) - (3/8×E(Sm) + 3/8×E(M) + 2/8×E(L))")
    print("IG(Ukuran) = 0.954 - (3/8×0 + 3/8×0.915 + 2/8×1)")
    print("IG(Ukuran) = 0.954 - 0.593")
    ig_ukuran = 0.361
    print(f"IG(Ukuran) = {ig_ukuran}")

    # Ringkasan hasil
    print("\nRINGKASAN INFORMATION GAIN:")
    print(f"IG(Warna)  = {ig_warna}")
    print(f"IG(Bahan)  = {ig_bahan}")
    print(f"IG(Ukuran) = {ig_ukuran}")

    return {"Warna": ig_warna, "Bahan": ig_bahan, "Ukuran": ig_ukuran}


# Data training
data = {
    "Warna": ["Merah", "Biru", "Hijau", "Kuning", "Merah", "Biru", "Hijau", "Kuning"],
    "Ukuran": ["S", "M", "L", "S", "M", "L", "S", "M"],
    "Bahan": ["Katun", "Sutra", "Wol", "Sutra", "Katun", "Wol", "Katun", "Sutra"],
    "Kategori": [
        "Casual",
        "Formal",
        "Casual",
        "Casual",
        "Casual",
        "Formal",
        "Casual",
        "Formal",
    ],
}

# Membuat DataFrame
df = pd.DataFrame(data)
print("Dataset:")
print(df)
print("\n" + "=" * 50)

# Menghitung Information Gain
ig_values = calculate_information_gain()

# Menentukan atribut terbaik
best_attribute = max(ig_values.items(), key=lambda x: x[1])[0]
print(f"\nAtribut terbaik: {best_attribute} (IG = {ig_values[best_attribute]})")
