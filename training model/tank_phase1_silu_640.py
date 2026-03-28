"""
KIIT-MiTA Tank - Phase 1: SiLU Baseline at 640×640 (1 class)
300 epochs, aggressive augmentation, cosine LR.
"""
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

if __name__ == '__main__':
    print("=" * 60)
    print("  TANK PIPELINE - Phase 1: SiLU @ 640×640")
    print("  1 Class: Tank | Epochs: 300")
    print("=" * 60)

    model = YOLO('yolov8n.pt')
    model.train(
        data=r'v:/SRNet/datasets/KIIT-MiTA-Tank/kiitmita_tank.yaml',
        epochs=300, imgsz=640, batch=16, device=0,
        project='v:/SRNet/runs/detect/kiitmita_tank',
        name='phase1_silu_640',
        val=True, deterministic=True, workers=0, cache=False,
        mosaic=1.0, mixup=0.3, copy_paste=0.15, erasing=0.4,
        degrees=15.0, scale=0.9, translate=0.2,
        fliplr=0.5, flipud=0.1,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        lr0=0.01, lrf=0.01, cos_lr=True,
        warmup_epochs=5, patience=0,
    )
    print("\n  Phase 1 DONE!")
