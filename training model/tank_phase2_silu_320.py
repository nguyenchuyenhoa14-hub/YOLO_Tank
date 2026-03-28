"""
KIIT-MiTA Tank - Phase 2: SiLU Fine-tune 640→320 (Teacher)
100 epochs, low LR, moderate augmentation.
"""
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

if __name__ == '__main__':
    print("=" * 60)
    print("  TANK PIPELINE - Phase 2: SiLU 640→320")
    print("  1 Class: Tank | Epochs: 100")
    print("=" * 60)

    PHASE1 = r'v:\SRNet\runs\detect\kiitmita_tank\phase1_silu_640\weights\best.pt'
    model = YOLO(PHASE1)
    model.train(
        data=r'v:/SRNet/datasets/KIIT-MiTA-Tank/kiitmita_tank.yaml',
        epochs=100, imgsz=320, batch=32, device=0,
        project='v:/SRNet/runs/detect/kiitmita_tank',
        name='phase2_silu_320',
        val=True, deterministic=True, workers=0, cache=False,
        mosaic=1.0, mixup=0.1, copy_paste=0.05,
        degrees=5.0, scale=0.5, translate=0.1,
        lr0=0.001, lrf=0.01, cos_lr=True,
        warmup_epochs=3, patience=0,
    )
    print("\n  Phase 2 DONE!")
