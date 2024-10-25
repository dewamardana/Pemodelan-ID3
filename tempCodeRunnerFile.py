import math


# Fungsi untuk menghitung entropi
def entropy(probabilities):
    return -sum([p * math.log2(p) if p > 0 else 0 for p in probabilities])


# Dataset
total_casual = 5
total_formal = 3
total_data = total_casual + total_formal

# Entropy seluruh dataset
p_casual = total_casual / total_data
p_formal = total_formal / total_data
entropy_total = entropy([p_casual, p_formal])
print(f"Entropy Total Dataset (H(S)): {entropy_total:.3f}")


# Entropy untuk atribut Warna
def calc_entropy_warna():
    warna_entropy = {
        "Merah": entropy([2 / 2, 0 / 2]),
        "Biru": entropy([0 / 2, 2 / 2]),
        "Hijau": entropy([2 / 2, 0 / 2]),
        "Kuning": entropy([1 / 2, 1 / 2]),
    }
    warna_weights = {"Merah": 2 / 8, "Biru": 2 / 8, "Hijau": 2 / 8, "Kuning": 2 / 8}
    ig_warna = entropy_total - sum(
        [warna_entropy[key] * warna_weights[key] for key in warna_entropy]
    )
    return ig_warna


# Entropy untuk atribut Bahan
def calc_entropy_bahan():
    bahan_entropy = {
        "Katun": entropy([3 / 3, 0 / 3]),
        "Sutra": entropy([1 / 4, 3 / 4]),
        "Wol": entropy([1 / 2, 1 / 2]),
    }
    bahan_weights = {"Katun": 3 / 8, "Sutra": 4 / 8, "Wol": 2 / 8}
    ig_bahan = entropy_total - sum(
        [bahan_entropy[key] * bahan_weights[key] for key in bahan_entropy]
    )
    return ig_bahan


# Entropy untuk atribut Size
def calc_entropy_size():
    size_entropy = {
        "Kecil": entropy([2 / 3, 1 / 3]),
        "Sedang": entropy([1 / 2, 1 / 2]),
        "Besar": entropy([2 / 3, 1 / 3]),
    }
    size_weights = {"Kecil": 3 / 8, "Sedang": 2 / 8, "Besar": 3 / 8}
    ig_size = entropy_total - sum(
        [size_entropy[key] * size_weights[key] for key in size_entropy]
    )
    return ig_size


# Menghitung Information Gain untuk setiap atribut
ig_warna = calc_entropy_warna()
ig_bahan = calc_entropy_bahan()
ig_size = calc_entropy_size()

print(f"Information Gain untuk atribut Warna: {ig_warna:.3f}")
print(f"Information Gain untuk atribut Bahan: {ig_bahan:.3f}")
print(f"Information Gain untuk atribut Size: {ig_size:.3f}")

# Menentukan root node
if ig_warna > ig_bahan and ig_warna > ig_size:
    root_node = "Warna"
elif ig_bahan > ig_warna and ig_bahan > ig_size:
    root_node = "Bahan"
else:
    root_node = "Size"
print(
    f"Atribut {root_node} dipilih sebagai Root Node dalam Decision Tree karena memiliki Information Gain tertinggi."
)
