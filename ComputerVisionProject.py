import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import requests
import os
from huggingface_hub import InferenceClient
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# ── TOKEN ─────────────────────────────────────────────────────
hf_token = os.environ.get("HF_TOKEN", "YOUR_TOKEN")

print("Loading BLIP model... (first run downloads ~900MB)")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model.eval()
print("✓ BLIP loaded")

hf_client = InferenceClient(api_key=hf_token)

# ── SCREEN SIZE ────────────────────────────────────────────────
WIDTH  = 1100
HEIGHT = 720

# ── UI ZONE CONSTANTS ─────────────────────────────────────────
TOPBAR_H   = 80
STATUS_H   = 40
PROMPT_H   = 60
TRANSF_H   = 60
TRANSF_Y1  = HEIGHT - TRANSF_H
PROMPT_Y1  = TRANSF_Y1 - PROMPT_H
STATUS_Y1  = PROMPT_Y1 - STATUS_H
DZ_Y1      = TOPBAR_H       # draw-zone top
DZ_Y2      = STATUS_Y1      # draw-zone bottom  (UI zones excluded)

ERASER_SIZE = 30            # half-size of eraser square (pixels)

# ── COLOUR / TOOL STATE ───────────────────────────────────────
colorIndex  = 0
colors      = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
color_names = ["BLUE", "GREEN", "RED", "YELLOW", "ERASER"]

# ── PROMPT STATE ──────────────────────────────────────────────
prompt_text = ""
typing      = False

# ── API PROTECTION ────────────────────────────────────────────
cached_caption    = ""        # last BLIP result — reused if canvas unchanged
canvas_dirty      = False     # True whenever user draws/erases; reset after generation
is_generating     = False     # blocks re-entrant generate calls
gen_cooldown      = 0         # frame counter — prevents GEN button spam
GEN_COOLDOWN_FRAMES = 60      # ~2 sec at 30 fps before GEN can fire again

# ── STROKE POINT QUEUES ───────────────────────────────────────
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]
blue_i = green_i = red_i = yellow_i = 0

# ═══════════════════════════════════════════════════════════════
#  TWO-LAYER DRAWING SYSTEM
#
#  strokeLayer : BGR canvas (white bg). Strokes are drawn here
#                by the point-queue render loop ONCE per new point.
#                We NEVER clear and redraw all points each frame —
#                that was the bug that re-painted over eraser holes.
#
#  eraserMask  : single-channel mask (255 = visible, 0 = erased).
#                Eraser tool writes 0s into this mask.
#                Final draw-zone pixel = strokeLayer pixel  if mask==255
#                                      = white              if mask==0
#
#  This means erasing is permanent and never overwritten.
# ═══════════════════════════════════════════════════════════════
strokeLayer = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
eraserMask  = np.ones((HEIGHT, WIDTH),    dtype=np.uint8) * 255   # fully visible


def apply_eraser(cx, cy):
    """Burn a hole in eraserMask inside the draw zone only."""
    x1 = max(cx - ERASER_SIZE, 0)
    y1 = max(cy - ERASER_SIZE, DZ_Y1)
    x2 = min(cx + ERASER_SIZE, WIDTH)
    y2 = min(cy + ERASER_SIZE, DZ_Y2)
    eraserMask[y1:y2, x1:x2] = 0


def get_visible_drawing():
    """Return strokeLayer with erased regions replaced by white."""
    visible = strokeLayer.copy()
    # Where mask is 0 → white
    visible[eraserMask == 0] = 255
    return visible


def lift_pen():
    global bpoints, gpoints, rpoints, ypoints
    global blue_i, green_i, red_i, yellow_i
    bpoints.append(deque(maxlen=1024)); blue_i   += 1
    gpoints.append(deque(maxlen=1024)); green_i  += 1
    rpoints.append(deque(maxlen=1024)); red_i    += 1
    ypoints.append(deque(maxlen=1024)); yellow_i += 1


def clear_canvas():
    global bpoints, gpoints, rpoints, ypoints
    global blue_i, green_i, red_i, yellow_i
    bpoints = [deque(maxlen=1024)]; blue_i   = 0
    gpoints = [deque(maxlen=1024)]; green_i  = 0
    rpoints = [deque(maxlen=1024)]; red_i    = 0
    ypoints = [deque(maxlen=1024)]; yellow_i = 0
    strokeLayer[DZ_Y1:DZ_Y2, :] = 255        # wipe strokes
    eraserMask [DZ_Y1:DZ_Y2, :] = 255        # restore mask
    cached_caption = ""
    canvas_dirty   = False
    print(">>> CLEAR")


# ─────────────────────────────────────────────────────────────
# AI
# ─────────────────────────────────────────────────────────────

def save_canvas():
    cv2.imwrite("drawing.png", get_visible_drawing())

def auto_prompt():
    """Run BLIP only if canvas changed since last caption. Otherwise reuse cache."""
    global cached_caption, canvas_dirty
    if not canvas_dirty and cached_caption:
        print(f">>> Reusing cached caption: '{cached_caption}' (canvas unchanged)")
        return cached_caption
    try:
        print(">>> Analyzing drawing with BLIP...")
        img    = Image.open("drawing.png").convert("RGB")
        inputs = blip_processor(img, return_tensors="pt")
        with torch.no_grad():
            out = blip_model.generate(**inputs, max_new_tokens=30)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        print(f">>> BLIP caption: '{caption}'")
        cached_caption = caption
        canvas_dirty   = False
        return caption
    except Exception as e:
        print(f">>> BLIP error: {e}"); return cached_caption or ""

def generate_ai(prompt=None):
    """Generate image. Skips if already generating or on cooldown."""
    global is_generating, gen_cooldown, canvas_dirty
    if is_generating:
        print(">>> Already generating — please wait..."); return
    if gen_cooldown > 0:
        print(f">>> Cooldown active — wait a moment"); return
    is_generating = True
    try:
        print("\n>>> Converting drawing to AI image...")
        subject = (prompt.strip() if prompt and prompt.strip() else auto_prompt()) or "a creative artwork"
        ep = (f"highly detailed digital painting of {subject}, "
              f"concept art, artstation trending, sharp focus, vibrant colors, 8k, masterpiece")
        print(f">>> Enhanced prompt: '{ep}'")
        try:
            img = hf_client.text_to_image(ep, model="stabilityai/stable-diffusion-xl-base-1.0")
            img.save("generated.png")
            cv2.imshow("AI Generated Image", cv2.imread("generated.png"))
            gen_cooldown = GEN_COOLDOWN_FRAMES
            canvas_dirty = False
            print("✓ Done!\n")
        except Exception as e:
            print(f">>> SDXL failed: {e}")
            print("⚠ Try again in 30s. If it keeps failing, your token may be rate-limited.\n")
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        is_generating = False

# ─────────────────────────────────────────────────────────────
# TRANSFORMATIONS
# ─────────────────────────────────────────────────────────────

def scale_image():
    img = cv2.imread("drawing.png")
    cv2.imshow("Scaled",    cv2.resize(img, None, fx=1.5, fy=1.5))

def rotate_image():
    img = cv2.imread("drawing.png")
    h, w = img.shape[:2]
    cv2.imshow("Rotated",   cv2.warpAffine(img, cv2.getRotationMatrix2D((w//2,h//2),45,1),(w,h)))

def shift_image():
    img = cv2.imread("drawing.png")
    cv2.imshow("Shifted",   cv2.warpAffine(img, np.float32([[1,0,100],[0,1,50]]),
                                            (img.shape[1],img.shape[0])))

def reflect_image():
    img = cv2.imread("drawing.png")
    cv2.imshow("Reflected", cv2.flip(img, 1))

# ─────────────────────────────────────────────────────────────
# UI BUILD  (composited onto drawing every frame — never modifies layers)
# ─────────────────────────────────────────────────────────────

def build_display(hand_center=None):
    # Start from the visible drawing (strokes + eraser holes)
    frame = get_visible_drawing()

    # ── TOP BAR ───────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (WIDTH, TOPBAR_H), (35, 35, 45), -1)  # Dark sleek top bar
    top_btns = [
        (20,  160,  "CLEAR",  (80,  80,  80),   -1),
        (180, 320,  "BLUE",   (255, 120, 120),   0),  # BGR logic
        (340, 480,  "GREEN",  (120, 255, 120),   1),
        (500, 640,  "RED",    (120, 120, 255),   2),
        (660, 820,  "YELLOW", (0,   220, 220),   3),
        (840, 1000, "ERASER", (150, 150, 150),   4),
    ]
    for (x1, x2, label, col, tidx) in top_btns:
        active = (tidx == colorIndex)
        bg_col = col if active else (60, 60, 70)
        cv2.rectangle(frame, (x1, 12), (x2, TOPBAR_H-12), bg_col, -1)
        if active:
            cv2.rectangle(frame, (x1, 12), (x2, TOPBAR_H-12), (255, 255, 255), 2) # white border when active
        tc = (255, 255, 255) # white text
        # Center the text
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = 12 + (TOPBAR_H - 24 + text_size[1]) // 2
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, tc, 2, cv2.LINE_AA)

    # ── STATUS BAR ────────────────────────────────────────────
    cv2.rectangle(frame, (0, STATUS_Y1), (WIDTH, STATUS_Y1+STATUS_H), (45, 45, 55), -1)
    stxt = ("[TYPING MODE]  Press ENTER to confirm" if typing else
            f"Tool: {color_names[colorIndex]}   |   P=prompt   G=generate   E=eraser   S=save   Q=quit")
    cv2.putText(frame, stxt, (20, STATUS_Y1+26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

    # ── PROMPT ROW ────────────────────────────────────────────
    cv2.rectangle(frame, (0, PROMPT_Y1), (WIDTH, PROMPT_Y1+PROMPT_H), (25, 25, 35), -1)
    
    # Input box
    cv2.rectangle(frame, (100, PROMPT_Y1+10), (WIDTH-160, PROMPT_Y1+PROMPT_H-10), (55, 55, 65), -1)
    if typing:
        cv2.rectangle(frame, (100, PROMPT_Y1+10), (WIDTH-160, PROMPT_Y1+PROMPT_H-10), (200, 200, 255), 2)
    
    cv2.putText(frame, "PROMPT:", (15, PROMPT_Y1+38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
    
    display_text = prompt_text + ("|" if typing else "")
    cv2.putText(frame, display_text, (110, PROMPT_Y1+38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                
    # GEN Button
    cv2.rectangle(frame, (WIDTH-140, PROMPT_Y1+10), (WIDTH-20, PROMPT_Y1+PROMPT_H-10), (100, 200, 100), -1)
    cv2.putText(frame, "GENERATE", (WIDTH-125, PROMPT_Y1+38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    # ── TRANSFORM ROW ─────────────────────────────────────────
    cv2.rectangle(frame, (0, TRANSF_Y1), (WIDTH, HEIGHT), (35, 35, 45), -1)
    bw = WIDTH // 4
    for i, lbl in enumerate(["SCALE","ROTATE","SHIFT","REFLECT"]):
        x1 = i*bw+12; x2 = (i+1)*bw-12
        cv2.rectangle(frame, (x1, TRANSF_Y1+12), (x2, HEIGHT-12), (80, 80, 90), -1)
        text_size = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = TRANSF_Y1 + 12 + (HEIGHT - TRANSF_Y1 - 24 + text_size[1]) // 2
        cv2.putText(frame, lbl, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # ── CURSOR ────────────────────────────────────────────────
    if hand_center and DZ_Y1 <= hand_center[1] <= DZ_Y2:
        if colorIndex == 4:
            ex, ey = hand_center
            cv2.rectangle(frame,
                (max(ex-ERASER_SIZE,0),   max(ey-ERASER_SIZE, DZ_Y1)),
                (min(ex+ERASER_SIZE,WIDTH),min(ey+ERASER_SIZE, DZ_Y2)),
                (80,80,255), 2)
        else:
            cv2.circle(frame, hand_center, 6, (0,255,0), -1)

    return frame

# ─────────────────────────────────────────────────────────────
# MEDIAPIPE
# ─────────────────────────────────────────────────────────────
mpHands = mp.solutions.hands
hands   = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw  = mp.solutions.drawing_utils

# ─────────────────────────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"  AIRCANVAS AI  —  {WIDTH}×{HEIGHT}")
print("  TOP:  CLEAR|BLUE|GREEN|RED|YELLOW|ERASER")
print("  Keys: P=prompt  G=generate  E=eraser  S=save  Q=quit")
print("="*60 + "\n")

cap = cv2.VideoCapture(0)
if not cap.isOpened(): cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened(): cap = cv2.VideoCapture(1)
if not cap.isOpened(): print("ERROR: No camera."); exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Track previous point counts to know when NEW points are added
prev_counts = [0, 0, 0, 0]   # blue, green, red, yellow

landmarks    = []
hand_center  = None
hand_present = False

ret = True
while ret:
    ret, cam = cap.read()
    if not ret or cam is None: print("Camera read failed."); break

    cam    = cv2.resize(cv2.flip(cam, 1), (WIDTH, HEIGHT))
    result = hands.process(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB))

    hand_present = False
    hand_center  = None

    if result.multi_hand_landmarks:
        hand_present = True
        landmarks = []
        for hand in result.multi_hand_landmarks:
            for lm in hand.landmark:
                landmarks.append([int(lm.x * WIDTH), int(lm.y * HEIGHT)])

        center   = (landmarks[8][0], landmarks[8][1])   # index tip
        
        # 2-finger logic:
        # Index finger is up if tip (8) is higher than pip (6) (smaller Y)
        index_up = landmarks[8][1] < landmarks[6][1]
        # Middle finger is up if tip (12) is higher than pip (10)
        middle_up = landmarks[12][1] < landmarks[10][1]
        
        # Stop drawing (hover/selection mode) if middle finger is also UP or index is DOWN
        pinching = middle_up or not index_up
        hand_center = center

        button_touched = False

        # ── TOP BAR ───────────────────────────────────────────
        if center[1] <= TOPBAR_H:
            cx = center[0]
            if   20  <= cx <= 160:  clear_canvas();  button_touched = True
            elif 180 <= cx <= 320:  colorIndex=0; print(">>> BLUE");   button_touched=True
            elif 340 <= cx <= 480:  colorIndex=1; print(">>> GREEN");  button_touched=True
            elif 500 <= cx <= 640:  colorIndex=2; print(">>> RED");    button_touched=True
            elif 660 <= cx <= 820:  colorIndex=3; print(">>> YELLOW"); button_touched=True
            elif 840 <= cx <= 1000: colorIndex=4; print(">>> ERASER"); button_touched=True

        # ── PROMPT / GEN ──────────────────────────────────────
        elif PROMPT_Y1 <= center[1] <= PROMPT_Y1+PROMPT_H:
            if (WIDTH-140) <= center[0] <= (WIDTH-20) and pinching:
                save_canvas()
                generate_ai(prompt_text if prompt_text else None)
            button_touched = True

        # ── TRANSFORM ─────────────────────────────────────────
        elif TRANSF_Y1 <= center[1] <= HEIGHT:
            bw  = WIDTH // 4
            col = min(center[0] // bw, 3)
            save_canvas()
            [scale_image, rotate_image, shift_image, reflect_image][col]()
            button_touched = True

        # ── DRAW ZONE ─────────────────────────────────────────
        if not button_touched and DZ_Y1 <= center[1] <= DZ_Y2:
            if pinching:
                lift_pen()
            else:
                point_added = False
                if colorIndex == 4:
                    # ERASER — burn hole in mask (draw zone only)
                    apply_eraser(center[0], center[1])
                    canvas_dirty = True
                else:
                    active_queue = None
                    if colorIndex == 0: active_queue = bpoints[blue_i]
                    elif colorIndex == 1: active_queue = gpoints[green_i]
                    elif colorIndex == 2: active_queue = rpoints[red_i]
                    elif colorIndex == 3: active_queue = ypoints[yellow_i]
                    
                    # Only append if distance from last point > 4 pixels for perfectly stable lines
                    if len(active_queue) == 0 or np.hypot(active_queue[0][0] - center[0], active_queue[0][1] - center[1]) > 4:
                        active_queue.appendleft(center)
                        canvas_dirty = True
                        point_added = True
                
                # Genuine random sparkle effect applied exactly ONCE when the stroke is added
                if point_added and np.random.rand() > 0.4:
                    gx = center[0] + np.random.randint(-20, 21)
                    gy = center[1] + np.random.randint(-20, 21)
                    if 0 <= gx < WIDTH and DZ_Y1 <= gy <= DZ_Y2:
                        g_color = (
                            min(255, colors[colorIndex][0] + 160),
                            min(255, colors[colorIndex][1] + 160),
                            min(255, colors[colorIndex][2] + 160)
                        )
                        cv2.line(strokeLayer, (gx-4, gy), (gx+4, gy), g_color, 1)
                        cv2.line(strokeLayer, (gx, gy-4), (gx, gy+4), g_color, 1)
                        cv2.circle(strokeLayer, (gx, gy), 1, (255, 255, 255), -1)
                        cv2.circle(eraserMask, (gx, gy), 5, 255, -1)

    else:
        lift_pen()

    # ── RENDER NEW STROKE SEGMENTS onto strokeLayer ───────────
    # Only draw the current active deque (newest points).
    # Older deques were already rendered when they were active.
    # IMPORTANT: also restore eraserMask=255 wherever a new stroke
    # is drawn, so users can draw over previously erased areas.
    all_pts  = [bpoints, gpoints, rpoints, ypoints]
    all_idx  = [blue_i,  green_i, red_i,  yellow_i]
    for ci, (pts, qi) in enumerate(zip(all_pts, all_idx)):
        q = pts[qi]   # active deque
        for k in range(1, len(q)):
            p1, p2 = q[k-1], q[k]
            if p1 is None or p2 is None:
                continue
            # Draw stroke onto strokeLayer
            cv2.line(strokeLayer, p1, p2, colors[ci], 4)
            # Restore eraser mask along this stroke so it's visible
            cv2.line(eraserMask,  p1, p2, 255, 4)

    # ── BUILD + SHOW ──────────────────────────────────────────
    display = build_display(hand_center)

    # Draw hand skeleton on display only (not on any persistent layer)
    if hand_present and result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(display, hand, mpHands.HAND_CONNECTIONS)

    cv2.imshow("AirCanvas AI", display)

    key = cv2.waitKey(1) & 0xFF
    if gen_cooldown > 0:
        gen_cooldown -= 1          # tick down every frame (~30fps = 2s cooldown)
    if key == ord('q'): break
    if key == ord('s'): save_canvas(); print("✓ Saved!")
    if key == ord('e'): colorIndex=4;  print(">>> Eraser ON")
    if key == ord('p'): typing=True; prompt_text=""; print(">>> Type prompt then ENTER")
    if key == ord('g'):
        save_canvas()
        generate_ai(prompt_text if prompt_text else None)   # None → uses cache if clean

    if typing:
        if   key == 13:  typing=False; print(f">>> Prompt: '{prompt_text}'")
        elif key == 8:   prompt_text = prompt_text[:-1]
        elif key not in (255, 0xFF): prompt_text += chr(key)

cap.release()
cv2.destroyAllWindows()

