"""
Preprocess KIIT-MiTA labels: Filter only Tank (class 5) and remap to class 0.
Creates a new dataset folder with symlinked images and filtered labels.
"""
import os
import shutil
from pathlib import Path

SRC_DIR = Path(r'v:\SRNet\datasets\KIIT-MiTA')
DST_DIR = Path(r'v:\SRNet\datasets\KIIT-MiTA-Tank')
TANK_CLASS_ID = 5  # Tank is class 5 in original dataset

def filter_labels(split):
    src_img = SRC_DIR / split / 'images'
    src_lbl = SRC_DIR / split / 'labels'
    dst_img = DST_DIR / split / 'images'
    dst_lbl = DST_DIR / split / 'labels'
    
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    
    kept = 0
    skipped = 0
    
    for lbl_file in sorted(src_lbl.glob('*.txt')):
        # Read and filter labels
        tank_lines = []
        with open(lbl_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5 and int(parts[0]) == TANK_CLASS_ID:
                    # Remap class 5 → class 0
                    parts[0] = '0'
                    tank_lines.append(' '.join(parts))
        
        if tank_lines:
            # Write filtered label
            with open(dst_lbl / lbl_file.name, 'w') as f:
                f.write('\n'.join(tank_lines) + '\n')
            
            # Copy/link corresponding image
            img_name = lbl_file.stem
            for ext in ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG']:
                src_image = src_img / (img_name + ext)
                if src_image.exists():
                    dst_image = dst_img / (img_name + ext)
                    if not dst_image.exists():
                        shutil.copy2(str(src_image), str(dst_image))
                    break
            
            kept += 1
        else:
            skipped += 1
    
    return kept, skipped

if __name__ == '__main__':
    print("=" * 60)
    print("  KIIT-MiTA → Tank-Only Dataset Preprocessing")
    print(f"  Source: {SRC_DIR}")
    print(f"  Output: {DST_DIR}")
    print(f"  Keeping: class {TANK_CLASS_ID} (Tank) → remap to class 0")
    print("=" * 60)
    
    for split in ['train', 'valid']:
        kept, skipped = filter_labels(split)
        print(f"\n  [{split}] Kept {kept} images with Tank, skipped {skipped}")
    
    print(f"\n  Done! Tank-only dataset at: {DST_DIR}")
    print("=" * 60)
