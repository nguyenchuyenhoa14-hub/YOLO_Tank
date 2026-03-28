"""
KIIT-MiTA Tank - Phase 3: ReLU Student + KD (200 epochs)
Teacher: Phase 2 SiLU 320.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import ultralytics.nn.modules.conv as conv_modules
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

TEACHER_PATH = r'v:\SRNet\runs\detect\kiitmita_tank\phase2_silu_320\weights\best.pt'
IMGSZ = 320
ALPHA_KD = 0.5

conv_modules.Conv.default_act = nn.ReLU()

teacher_model = None

def setup_teacher():
    global teacher_model
    original_act = conv_modules.Conv.default_act
    conv_modules.Conv.default_act = nn.SiLU()
    teacher_model = YOLO(TEACHER_PATH)
    teacher_model.model.eval()
    for p in teacher_model.model.parameters():
        p.requires_grad = False
    conv_modules.Conv.default_act = nn.ReLU()
    print(f"  Teacher loaded: {TEACHER_PATH}")

def on_train_batch_end(trainer):
    global teacher_model
    if teacher_model is None or trainer.epoch < 3:
        return
    try:
        imgs = trainer.batch["img"].to(trainer.device)
        with torch.no_grad():
            t_out = teacher_model.model(imgs)
        s_out = trainer.model(imgs)
        kd_loss = torch.tensor(0.0, device=trainer.device)
        if isinstance(s_out, (list, tuple)) and isinstance(t_out, (list, tuple)):
            for s, t in zip(s_out, t_out):
                if s.shape == t.shape:
                    kd_loss += F.mse_loss(F.normalize(s.flatten(2), dim=-1),
                                          F.normalize(t.flatten(2), dim=-1))
        if kd_loss.item() > 0:
            (kd_loss * ALPHA_KD * 0.01).backward()
    except Exception:
        pass

if __name__ == '__main__':
    print("=" * 60)
    print("  TANK PIPELINE - Phase 3: ReLU + KD @ 320")
    print("  1 Class: Tank | Epochs: 200")
    print("=" * 60)

    setup_teacher()
    student = YOLO('yolov8n.pt')
    student.add_callback("on_train_batch_end", on_train_batch_end)

    student.train(
        data=r'v:/SRNet/datasets/KIIT-MiTA-Tank/kiitmita_tank.yaml',
        epochs=200, imgsz=IMGSZ, batch=32, device=0,
        project='v:/SRNet/runs/detect/kiitmita_tank',
        name='phase3_relu_kd',
        val=True, deterministic=True, workers=0, cache=False,
        mosaic=1.0, mixup=0.2, copy_paste=0.1,
        degrees=10.0, scale=0.7, translate=0.2, erasing=0.3,
        lr0=0.01, lrf=0.01, cos_lr=True,
        warmup_epochs=5, patience=0,
    )
    print("\n  Phase 3 DONE!")
