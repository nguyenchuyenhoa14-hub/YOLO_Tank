#!/usr/bin/env python3
"""
visualize_hardware_nms.py (v3 — Pure RTL NMS)
Reads the final NMS centroids directly from the hardware output `tank_centers.mem`
and visualizes them against the PyTorch Golden model. No Python-side suppression.
"""
import cv2, os, sys, numpy as np, shutil
from ultralytics import YOLO

IMG_PATH    = sys.argv[1] if len(sys.argv) > 1 else "pic_test/tank4.jpg"
MEM_PATH    = "rtl_output/tank_centers.mem"
_basename   = os.path.splitext(os.path.basename(IMG_PATH))[0]
OUT_PATH    = f"rtl_detection_result_{_basename}.jpg"
INPUT_SIZE  = 320
MATCH_DIST  = 60
ARTIFACTS   = "/home/hoangnguyen/.gemini/antigravity/brain/b088799d-96f1-4dba-b350-a1cfbcdd7818"

def iou(b1, b2):
    ix1=max(b1[0],b2[0]); iy1=max(b1[1],b2[1])
    ix2=min(b1[2],b2[2]); iy2=min(b1[3],b2[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    u=(b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-inter
    return inter/u if u>0 else 0.

def main():
    print("="*65)
    print("  PURE RTL HARDWARE NMS vs GOLDEN PYTORCH")
    print("="*65)

    img = cv2.imread(IMG_PATH)
    if img is None: print(f"❌ {IMG_PATH}"); sys.exit(1)
    img_320 = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))

    # 1. PyTorch Golden Annotations (Dynamic Inference)
    from ultralytics import YOLO
    model = YOLO("kiitmita_tank/pcq_finetune_200/weights/best.pt")
    img_rgb = cv2.cvtColor(img_320, cv2.COLOR_BGR2RGB)
    results = model(img_rgb, imgsz=INPUT_SIZE, conf=0.01, iou=0.45, verbose=False)
    
    golden = []
    if results and len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            b = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            conf = float(box.conf[0])
            w = b[2] - b[0]; h = b[3] - b[1]
            # Filter out edge artifacts: elongated boxes (AR>4) or touching bottom 10px
            if h > 0 and w / h > 4:
                continue
            if b[3] >= INPUT_SIZE - 10:
                continue
            cx = (b[0] + b[2]) / 2
            cy = (b[1] + b[3]) / 2
            golden.append({'cx': cx, 'cy': cy, 'box': b, 'conf': conf})
    
    print(f"\n🥇 PyTorch Golden Model ({len(golden)} targets):")
    for i, g in enumerate(golden):
        b = g['box']
        print(f"  GT{i+1:<2}: ({g['cx']:.0f},{g['cy']:.0f})  sz={(b[2]-b[0]):.0f}x{(b[3]-b[1]):.0f}  conf={g['conf']:.2f}")

    # 2. Hardware NMS Output (tank_centers.mem)
    #    Format: 3 hex words per line: word0=cx_cy  word1=x1_y1  word2=x2_y2
    #    Bit packing: [25:16]=upper10, [9:0]=lower10
    #    Terminated by DEADBEEF
    rtl = []
    if not os.path.exists(MEM_PATH):
        print(f"❌ {MEM_PATH} not found!"); sys.exit(1)
    
    with open(MEM_PATH, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('//'): continue
            tokens = line.split()
            if tokens[0] == 'DEADBEEF': break
            if len(tokens) < 3: continue
            try:
                w0 = int(tokens[0], 16)
                w1 = int(tokens[1], 16)
                w2 = int(tokens[2], 16)
                cx = (w0 >> 16) & 0x3FF
                cy = w0 & 0x3FF
                x1 = (w1 >> 16) & 0x3FF
                y1 = w1 & 0x3FF
                x2 = (w2 >> 16) & 0x3FF
                y2 = w2 & 0x3FF
                rtl.append({'cx':cx, 'cy':cy, 'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2,
                            'box':[x1, y1, x2, y2], 'ri': idx})
            except ValueError: pass

    print(f"\n📍 RTL Hardware Detections ({len(rtl)} after NMS):")
    for i, c in enumerate(rtl):
        print(f"  RTL{i+1:<2}: center=({c['cx']},{c['cy']}) bbox=[{c['x1']},{c['y1']},{c['x2']},{c['y2']}]")

    # 3. Hungarian Optimal Matching (using IoU with actual RTL bboxes)
    from scipy.optimize import linear_sum_assignment
    n_gt = len(golden); n_rt = len(rtl)
    cost = np.full((n_gt, n_rt), 9999.)
    iou_matrix = np.zeros((n_gt, n_rt))
    
    for gi, g in enumerate(golden):
        for ri, r in enumerate(rtl):
            score = iou(g['box'], r['box'])
            iou_matrix[gi, ri] = score
            if score > 0.01:  # any overlap at all
                cost[gi, ri] = 1.0 - score  # minimize cost = maximize IoU
                
    row_ind, col_ind = linear_sum_assignment(cost)
    assigned = {gi: ri for gi, ri in zip(row_ind, col_ind) if cost[gi, ri] < 0.99}

    gt_results = []
    print(f"\n📊 GT-Optimal Matching (Hungarian, IoU-based):")
    print(f"  {'GT':5} {'GT bbox':24} {'RTL bbox':24} {'IoU':6} {'Result'}")
    print(f"  " + "─"*75)

    for gi, g in enumerate(golden):
        if gi in assigned:
            ri = assigned[gi]; r = rtl[ri]
            score = iou_matrix[gi, ri]
            gb = g['box']; rb = r['box']
            dist = np.sqrt((r['cx']-g['cx'])**2 + (r['cy']-g['cy'])**2)
            status = "✅ MATCH" if score>=0.5 else ("⚠️ CLOSE" if score>=0.3 else "⚠️ OFFSET")
            print(f"  GT{gi+1:<2}: [{gb[0]:.0f},{gb[1]:.0f},{gb[2]:.0f},{gb[3]:.0f}] → "
                  f"RTL[{rb[0]},{rb[1]},{rb[2]},{rb[3]}]  "
                  f"dist={dist:4.1f}px  IoU={score:.2f}  {status}")
            gt_results.append({'gt':g,'r':r,'ri':ri,'dist':dist,'iou':score,'box':rb,'valid':True})
        else:
            print(f"  GT{gi+1:<2}: [{g['box'][0]:.0f},{g['box'][1]:.0f},{g['box'][2]:.0f},{g['box'][3]:.0f}] → ❌ MISSED")
            gt_results.append({'gt':g,'r':None,'ri':-1,'dist':999,'iou':0,'valid':False})

    matched = sum(1 for x in gt_results if x['iou'] >= 0.5)
    close   = sum(1 for x in gt_results if 0.3 <= x['iou'] < 0.5)
    missed  = sum(1 for x in gt_results if not x['valid'])
    matched_ris = {res['ri'] for res in gt_results if res['valid']}
    false_pos = sum(1 for ri in range(len(rtl)) if ri not in matched_ris)
    avg_iou = np.mean([x['iou'] for x in gt_results]) if gt_results else 0
    print(f"\n  ✅ MATCH(IoU≥0.5): {matched}/{len(golden)}   ⚠️ CLOSE: {close}/{len(golden)}   ❌ MISSED: {missed}/{len(golden)}   🔴 FP: {false_pos}")
    print(f"  Average IoU: {avg_iou:.3f}  (RTL total: {len(rtl)} detections)")

    # 4. Render
    H, W = img_320.shape[:2]
    gt_img  = img_320.copy()
    rtl_img = img_320.copy()
    MATCHED = (0, 165, 255);  CLOSE = (0, 220, 220);  NOISE = (60, 60, 180)

    for i, g in enumerate(golden):
        b = [int(v) for v in g['box']]
        cv2.rectangle(gt_img, (b[0],b[1]), (b[2],b[3]), (255,150,0), 2)  # Light Blue (BGR) for better visibility
        # Add black outline for text
        cv2.putText(gt_img, f"GT{i+1}", (b[0]+2, b[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2)
        cv2.putText(gt_img, f"GT{i+1}", (b[0]+2, b[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,150,0), 1)

    matched_ri = {res['ri'] for res in gt_results if res['valid']}
    for res in gt_results:
        if not res['valid']: continue
        r = res['r']; b = [int(v) for v in res['box']]
        color = MATCHED if res['iou'] >= 0.5 else CLOSE
        cv2.rectangle(rtl_img, (b[0],b[1]), (b[2],b[3]), color, 2)
        
        
        # Find R index by position in rtl list
        r_idx = rtl.index(r) + 1
        text = f"R{r_idx}"
        # Add black outline for text
        cv2.putText(rtl_img, text, (b[0]+2, b[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2)
        cv2.putText(rtl_img, text, (b[0]+2, b[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    for ri, r in enumerate(rtl):
        if ri not in matched_ri:
            b = [int(r['x1']), int(r['y1']), int(r['x2']), int(r['y2'])]
            cv2.rectangle(rtl_img, (b[0],b[1]), (b[2],b[3]), NOISE, 2)

    # Add black outline for headers
    cv2.putText(gt_img, "PyTorch FP32 Golden", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3)
    cv2.putText(gt_img, "PyTorch FP32 Golden", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,150,0), 1)
    
    cv2.putText(rtl_img, "RTL Hardware INT8", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3)
    cv2.putText(rtl_img, "RTL Hardware INT8", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,165,255), 1)

    divider = np.full((H, 6, 3), [80,80,80], dtype=np.uint8)
    cv2.imwrite(OUT_PATH, np.hstack([gt_img, divider, rtl_img]))
    shutil.copy(OUT_PATH, os.path.join(ARTIFACTS, OUT_PATH))
    print(f"\n🖼️  Saved → {OUT_PATH}")

if __name__ == "__main__":
    main()
