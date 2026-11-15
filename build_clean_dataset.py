import os
import shutil
import random
import hashlib
from collections import defaultdict

random.seed(42)

CLASSES = ["Bacterial", "Fungal", "Hotspot", "Mange", "Allergic Dermatitis"]
PUBLIC_LOCAL = "ProcessedDataset224_PublicLocal"
EXTRA_SOURCE = "Datasets224new"
AD_STRUCTURED = "Allergic dermatitis"
AD_FLAT_DIRS = {
    "Train": "Allergic Dermatitis train",
    "Valid": "Allergic Dermatitis valid",
    "Test": "Allergic Dermatitis test",
}
OUTPUT_DIR = "ProcessedDataset224_Clean_5class"

SPLIT = (0.70, 0.15, 0.15)

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def collect(source):
    files = []
    for split in os.listdir(source):
        sp = os.path.join(source, split)
        if not os.path.isdir(sp):
            continue
        for cls in CLASSES:
            cp = os.path.join(sp, cls)
            if not os.path.isdir(cp):
                continue
            for fname in os.listdir(cp):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    continue
                fpath = os.path.join(cp, fname)
                files.append((fpath, cls, split))
    return files

def collect_flat(source_dir, class_name, split_name):
    files = []
    if not os.path.isdir(source_dir):
        return files
    for fname in os.listdir(source_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue
        fpath = os.path.join(source_dir, fname)
        files.append((fpath, class_name, split_name))
    return files

print("Scanning PublicLocal...")
public_files = collect(PUBLIC_LOCAL)
print(f"  Found {len(public_files)} images")

print("Computing hashes (PublicLocal)...")
public_by_md5 = {}
for fpath, cls, split in public_files:
    h = md5(fpath)
    if h not in public_by_md5:
        public_by_md5[h] = (fpath, cls, split)

public_unique = list(public_by_md5.values())
print(f"  Unique: {len(public_unique)} (removed {len(public_files) - len(public_unique)} duplicates)")

print(f"Scanning {EXTRA_SOURCE}...")
extra_files = collect(EXTRA_SOURCE)
print(f"  Found {len(extra_files)} images")

print("Computing hashes (Extra)...")
extra_by_md5 = {}
for fpath, cls, split in extra_files:
    h = md5(fpath)
    if h not in extra_by_md5:
        extra_by_md5[h] = (fpath, cls, split)

extra_unique = list(extra_by_md5.values())
print(f"  Unique: {len(extra_unique)} (removed {len(extra_files) - len(extra_unique)} internal dupes)")

public_hashes = set(public_by_md5.keys())
extra_new = [(p, c, s) for p, c, s in extra_unique if md5(p) not in public_hashes]
print(f"  New images (not in PublicLocal): {len(extra_new)}")

print(f"Scanning {AD_STRUCTURED} (structured)...")
ad_structured = collect(AD_STRUCTURED)
print(f"  Found {len(ad_structured)} images")

ad_struct_by_md5 = {}
for fpath, cls, split in ad_structured:
    h = md5(fpath)
    if h not in ad_struct_by_md5:
        ad_struct_by_md5[h] = (fpath, cls, split)
ad_struct_unique = list(ad_struct_by_md5.values())
print(f"  Unique: {len(ad_struct_unique)}")

print("Scanning Allergic Dermatitis flat folders...")
ad_flat_all = []
for split_name, flat_dir in AD_FLAT_DIRS.items():
    flat_files = collect_flat(flat_dir, "Allergic Dermatitis", split_name)
    print(f"  {flat_dir}: {len(flat_files)} images")
    ad_flat_all.extend(flat_files)

ad_flat_by_md5 = {}
for fpath, cls, split in ad_flat_all:
    h = md5(fpath)
    if h not in ad_flat_by_md5:
        ad_flat_by_md5[h] = (fpath, cls, split)
ad_flat_unique = list(ad_flat_by_md5.values())
print(f"  Unique: {len(ad_flat_unique)}")

existing_hashes = set(public_by_md5.keys())
ad_struct_new = [(p, c, s) for p, c, s in ad_struct_unique if md5(p) not in existing_hashes]
ad_flat_new = [(p, c, s) for p, c, s in ad_flat_unique if md5(p) not in existing_hashes]
print(f"  AD structured new (not in PublicLocal): {len(ad_struct_new)}")
print(f"  AD flat new (not in PublicLocal): {len(ad_flat_new)}")

combined_by_class = defaultdict(dict)
for fpath, cls, _ in public_unique:
    h = md5(fpath)
    combined_by_class[cls][h] = fpath
for fpath, cls, _ in extra_new:
    h = md5(fpath)
    combined_by_class[cls][h] = fpath
for fpath, cls, _ in ad_struct_new:
    h = md5(fpath)
    combined_by_class[cls][h] = fpath
for fpath, cls, _ in ad_flat_new:
    h = md5(fpath)
    combined_by_class[cls][h] = fpath

print("\n=== Combined counts per class ===")
for cls in CLASSES:
    total = len(combined_by_class[cls])
    print(f"  {cls}: {total}")

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

for split_name, ratio in [("Train", SPLIT[0]), ("Valid", SPLIT[1]), ("Test", SPLIT[2])]:
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, split_name, cls), exist_ok=True)

print("\n=== Splitting ===")
train_count = 0
valid_count = 0
test_count = 0
for cls in CLASSES:
    items = list(combined_by_class[cls].items())
    random.shuffle(items)
    n = len(items)
    n_train = int(n * SPLIT[0])
    n_valid = int(n * SPLIT[1])
    n_test = n - n_train - n_valid

    train_items = items[:n_train]
    valid_items = items[n_train:n_train + n_valid]
    test_items = items[n_train + n_valid:]

    for h, src_path in train_items:
        ext = os.path.splitext(src_path)[1].lower()
        dst = os.path.join(OUTPUT_DIR, "Train", cls, f"{h}{ext}")
        shutil.copy2(src_path, dst)
    for h, src_path in valid_items:
        ext = os.path.splitext(src_path)[1].lower()
        dst = os.path.join(OUTPUT_DIR, "Valid", cls, f"{h}{ext}")
        shutil.copy2(src_path, dst)
    for h, src_path in test_items:
        ext = os.path.splitext(src_path)[1].lower()
        dst = os.path.join(OUTPUT_DIR, "Test", cls, f"{h}{ext}")
        shutil.copy2(src_path, dst)

    train_count += n_train
    valid_count += n_valid
    test_count += n_test
    print(f"  {cls}: {n_train} train / {n_valid} valid / {n_test} test")

print(f"\n  Total: {train_count} train / {valid_count} valid / {test_count} test")
print(f"\nDone! Clean 5-class dataset saved to {OUTPUT_DIR}")
