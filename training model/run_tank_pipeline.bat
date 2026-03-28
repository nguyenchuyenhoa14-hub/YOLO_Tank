@echo off
REM ============================================================
REM  KIIT-MiTA Tank-Only Pipeline (1 Class, 320×320)
REM  Expected: mAP50-95 ~ 0.65-0.70 (vs 0.37 with 7 classes)
REM  Estimated time: ~14-18 hours on RTX 2070
REM ============================================================

echo.
echo ============================================================
echo   KIIT-MiTA TANK-ONLY PIPELINE (1 Class, 320x320)
echo ============================================================
echo.

call C:\Users\Nguyen.DESKTOP-9B72JVV\miniconda3\Scripts\activate
call conda activate drone_ai

echo [0/5] ===== PREPROCESSING: Filter Tank labels =====
python v:\SRNet\training_scripts\preprocess_tank_only.py
if %errorlevel% neq 0 ( echo ERROR: Preprocessing failed! & pause & exit /b 1 )

echo.
echo [1/5] ===== PHASE 1: SiLU @ 640x640 (300 epochs) =====
python v:\SRNet\training_scripts\tank_phase1_silu_640.py
if %errorlevel% neq 0 ( echo ERROR: Phase 1 failed! & pause & exit /b 1 )

echo.
echo [2/5] ===== PHASE 2: SiLU 640 to 320 (100 epochs) =====
python v:\SRNet\training_scripts\tank_phase2_silu_320.py
if %errorlevel% neq 0 ( echo ERROR: Phase 2 failed! & pause & exit /b 1 )

echo.
echo [3/5] ===== PHASE 3: ReLU + KD @ 320 (200 epochs) =====
python v:\SRNet\training_scripts\tank_phase3_relu_kd.py
if %errorlevel% neq 0 ( echo ERROR: Phase 3 failed! & pause & exit /b 1 )

echo.
echo [4/5] ===== PHASE 4: QAT 200 epochs @ 320 =====
python v:\SRNet\training_scripts\tank_phase4_qat_200.py
if %errorlevel% neq 0 ( echo ERROR: Phase 4 failed! & pause & exit /b 1 )

echo.
echo [5/5] ===== PHASE 5: Export + PCQ + .mem =====
python v:\SRNet\training_scripts\tank_phase5_export.py
if %errorlevel% neq 0 ( echo ERROR: Phase 5 failed! & pause & exit /b 1 )

echo.
echo ============================================================
echo   ALL DONE! Tank-Only Pipeline Complete!
echo   .mem files: runs/detect/kiitmita_tank/yolov8n_kiitmita_tank_relu_320_qat_mem/
echo ============================================================
pause
