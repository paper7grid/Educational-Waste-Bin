#!/usr/bin/env python3
# smart_bin.py  —  RUN WITH: sudo python3 smart_bin.py

import cv2
import numpy as np
import time
import math
from collections import defaultdict
from ai_edge_litert.interpreter import Interpreter
from picamera2 import Picamera2
from rpi_ws281x import PixelStrip, Color
from gpiozero import Button

# ───────────────────────────────────────────────
#  DISPLAY
# ───────────────────────────────────────────────
W, H = 1280, 720

# ───────────────────────────────────────────────
#  PNG BACKGROUNDS  — place next to smart_bin.py
# ───────────────────────────────────────────────
_PNG = {
    "sleep":   "scene1.png",
    "quiz":    "quiz.png",
    "correct": "correct.png",
    "wrong":   "wrong.png",
}

def _load_png(key):
    img = cv2.imread(_PNG[key], cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

PNG = {k: _load_png(k) for k in _PNG}

def show_png(canvas, key):
    img = PNG.get(key)
    if img is not None:
        np.copyto(canvas, img)
    else:
        canvas[:] = C_BG

# ───────────────────────────────────────────────
#  TEXT POSITION CONSTANTS  ← tweak these
# ───────────────────────────────────────────────
CORRECT_TEXT_Y = 230 # ← Y of "Recycle: 99%" on correct.png
WRONG_LABEL_Y  = 460   # ← Y of "It is actually" on wrong.png
WRONG_ITEM_Y   = 590   # ← Y of item + % on wrong.png

# ───────────────────────────────────────────────
#  DESIGN TOKENS  (BGR)
# ───────────────────────────────────────────────
C_BG     = ( 10,  12,  10)
C_WHITE  = (235, 238, 235)
C_DIM    = (105, 110, 105)
C_PANEL  = ( 18,  24,  18)
C_BORDER = ( 48,  58,  48)
C_GREEN  = ( 72, 200,  82)
C_BLUE   = (215, 148,  44)
C_GREY   = (200, 200, 200)

BIN_COL   = {"COMPOST": C_GREEN, "RECYCLE": C_BLUE, "TRASH": C_GREY}
BIN_LABEL = {"COMPOST": "Compost", "RECYCLE": "Recycle", "TRASH": "Trash"}

F  = cv2.FONT_HERSHEY_SIMPLEX
FB = cv2.FONT_HERSHEY_DUPLEX

# ───────────────────────────────────────────────
#  LEDs
# ───────────────────────────────────────────────
strip = PixelStrip(180, 18, brightness=200)
strip.begin()
ZONES   = {"COMPOST":(0,60), "RECYCLE":(60,120), "TRASH":(120,180)}
LCOLORS = {
    "COMPOST": Color(0,210,70),
    "RECYCLE": Color(44,148,215),
    "TRASH": Color(255, 0, 0),
}
led_off_at = 0.0

def led_clear():
    for i in range(180): strip.setPixelColor(i, Color(0,0,0))
    strip.show()

def led_show(cat, dur=15):
    global led_off_at
    led_clear()
    s, e = ZONES[cat]
    for i in range(s, e): strip.setPixelColor(i, LCOLORS[cat])
    strip.show()
    led_off_at = time.time() + dur

# ───────────────────────────────────────────────
#  BUTTONS   GPIO 2=Compost  3=Recycle  4=Trash
# ───────────────────────────────────────────────
B_COMPOST = Button(2, bounce_time=0.1)
B_RECYCLE = Button(3, bounce_time=0.1)
B_TRASH   = Button(4, bounce_time=0.1)

flags      = {"any":False, "c":False, "r":False, "t":False}
press_time = {"c":-99.0, "r":-99.0, "t":-99.0}

def on_c(): flags["c"]=flags["any"]=True; press_time["c"]=time.time()
def on_r(): flags["r"]=flags["any"]=True; press_time["r"]=time.time()
def on_t(): flags["t"]=flags["any"]=True; press_time["t"]=time.time()

B_COMPOST.when_pressed = on_c
B_RECYCLE.when_pressed = on_r
B_TRASH.when_pressed   = on_t

def clr(): flags.update({"any":False,"c":False,"r":False,"t":False})

# ───────────────────────────────────────────────
#  BIN MAPPING
# ───────────────────────────────────────────────
def get_bin(label):
    l = label.strip().lower()
    if "compost" in l: return "COMPOST"
    if "recycle" in l: return "RECYCLE"
    return "TRASH"

# ───────────────────────────────────────────────
#  DRAWING HELPERS
# ───────────────────────────────────────────────
def tsz(txt, scale, thick=2, font=F):
    return cv2.getTextSize(txt, font, scale, thick)[0]

def put(canvas, txt, x, y, scale, col, thick=2, font=F):
    cv2.putText(canvas, txt, (x, y), font, scale, (0,0,0), thick+3, cv2.LINE_8)
    cv2.putText(canvas, txt, (x, y), font, scale, col,     thick,   cv2.LINE_8)

def putc(canvas, txt, y, scale, col, thick=2, font=F):
    w2 = tsz(txt, scale, thick, font)[0]
    put(canvas, txt, (W-w2)//2, y, scale, col, thick, font)

def rr(canvas, x, y, w2, h2, col, r=16, t=-1):
    r = min(r, w2//2, h2//2)
    cv2.rectangle(canvas,(x+r,y),(x+w2-r,y+h2), col, t)
    cv2.rectangle(canvas,(x,y+r),(x+w2,y+h2-r), col, t)
    for px, py in [(x+r,y+r),(x+w2-r,y+r),(x+r,y+h2-r),(x+w2-r,y+h2-r)]:
        cv2.circle(canvas,(px,py),r,col,t)

def panel_box(canvas, x, y, w2, h2, r=18, alpha=0.85):
    ov = canvas.copy()
    rr(ov, x, y, w2, h2, C_PANEL, r=r)
    cv2.addWeighted(ov, alpha, canvas, 1.0-alpha, 0, canvas)
    rr(canvas, x, y, w2, h2, C_BORDER, r=r, t=2)

def pbar(canvas, elapsed, total, x, y, bw, bh, col):
    rr(canvas, x, y, bw, bh, (28,33,28), r=bh//2)
    fill = max(4, int(bw * min(elapsed/total, 1.0)))
    rr(canvas, x, y, fill, bh, col, r=bh//2)

# ── COLOUR DEBUG — set to True once, look at the 4 quadrants, then set False ──
COLOUR_DEBUG = False
DISPLAY_CONV = "NONE"

def to_bgr(raw):
    frame = np.ascontiguousarray(raw[:, :, :3] if raw.shape[2] == 4 else raw)
    if DISPLAY_CONV == "RGB2BGR":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    elif DISPLAY_CONV == "BGR2RGB":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif DISPLAY_CONV == "YUV2BGR":
        return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)
    else:  # NONE — raw passthrough
        return frame

# ── FLIP CONTROL — change FLIP_CODE if image is still mirrored/upside-down:
#   1  = horizontal mirror (left↔right)   ← start here
#   0  = vertical flip (upside-down)
#  -1  = both axes
#  None = no flip
FLIP_CODE = 1

def cam_frame(raw):
    bgr = to_bgr(raw)
    if FLIP_CODE is not None:
        bgr = cv2.flip(bgr, FLIP_CODE)
    ch, cw = bgr.shape[:2]
    s = max(W/cw, H/ch)
    nw, nh = int(cw*s), int(ch*s)
    big = cv2.resize(bgr,(nw,nh),interpolation=cv2.INTER_LINEAR)
    x0, y0 = (nw-W)//2, (nh-H)//2
    out = np.ascontiguousarray(big[y0:y0+H, x0:x0+W])

    if COLOUR_DEBUG:
        # Show 4 small conversion options so you can pick the correct one
        raw3 = np.ascontiguousarray(raw[:, :, :3] if raw.shape[2] == 4 else raw)
        options = [
            ("NONE",      raw3),
            ("RGB2BGR",   cv2.cvtColor(raw3, cv2.COLOR_RGB2BGR)),
            ("BGR2RGB",   cv2.cvtColor(raw3, cv2.COLOR_BGR2RGB)),
            ("FLIP_CHAN", raw3[:, :, ::-1]),
        ]
        tw, th = W//4, H//4
        for i, (label, img) in enumerate(options):
            thumb = cv2.resize(img, (tw, th))
            x = i * tw
            out[0:th, x:x+tw] = thumb
            cv2.putText(out, label, (x+6, th-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return out

# ───────────────────────────────────────────────
#  SCAN RETICLE
# ───────────────────────────────────────────────
def draw_reticle(canvas, cx, cy, size, col, t_now):
    arm = size // 3; thk = 3
    for sx, sy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
        bx = cx + sx * size; by = cy + sy * size
        cv2.line(canvas,(bx,by),(bx - sx*arm, by), col, thk, cv2.LINE_AA)
        cv2.line(canvas,(bx,by),(bx, by - sy*arm), col, thk, cv2.LINE_AA)
    gap = 14
    cv2.line(canvas,(cx-size+arm,cy),(cx-gap,cy), col,1,cv2.LINE_AA)
    cv2.line(canvas,(cx+gap,cy),(cx+size-arm,cy), col,1,cv2.LINE_AA)
    cv2.line(canvas,(cx,cy-size+arm),(cx,cy-gap), col,1,cv2.LINE_AA)
    cv2.line(canvas,(cx,cy+gap),(cx,cy+size-arm), col,1,cv2.LINE_AA)
    cv2.circle(canvas,(cx,cy),4,col,-1,cv2.LINE_AA)
    sweep = int((t_now % 2.5) / 2.5 * (2*size)) - size
    sy2 = cy + sweep
    if cy - size < sy2 < cy + size:
        cv2.line(canvas,(cx-size,sy2),(cx+size,sy2),col,1,cv2.LINE_AA)

# ───────────────────────────────────────────────
#  QUIZ CIRCLE BUTTON
# ───────────────────────────────────────────────
def quiz_circle(canvas, cx, cy, rad, col, cat, label, pt):
    age = time.time() - pt
    r = int(rad*(1.0 - 0.15*math.sin(age/0.3*math.pi))) if 0 < age < 0.3 else rad
    dark = tuple(max(0, c//5) for c in col)
    cv2.circle(canvas,(cx,cy),r,dark,-1,cv2.LINE_AA)
    cv2.circle(canvas,(cx,cy),r,col,5,cv2.LINE_AA)
    hl = tuple(min(255,int(c*1.5)) for c in col)
    cv2.ellipse(canvas,(cx-r//6,cy-r//6),(r//4,r//5),310,0,120,hl,3,cv2.LINE_AA)
    _icon(canvas, cx, cy - r//10, int(r*0.42), cat, col)
    lbl_w = tsz(label,1.1,2,FB)[0]
    put(canvas, label, cx-lbl_w//2, cy+r+46, 1.1, col, 2, FB)
    if 0 < age < 0.22:
        alpha = 1.0 - age/0.22
        flash = canvas.copy()
        cv2.circle(flash,(cx,cy),r,tuple(min(255,int(c*0.7)) for c in col),-1,cv2.LINE_AA)
        cv2.addWeighted(flash,alpha*0.30,canvas,1-alpha*0.30,0,canvas)

def _icon(canvas, cx, cy, s, cat, col):
    hl = tuple(min(255, int(c*1.5)) for c in col)
    if cat == "COMPOST":
        cv2.ellipse(canvas,(cx,cy-s//8),(int(s*.55),int(s*.38)),-40,0,360,hl,-1,cv2.LINE_AA)
        vein = tuple(max(0,c//4) for c in col)
        cv2.line(canvas,(cx-s//3,cy+s//5),(cx+s//5,cy-s//2),vein,3,cv2.LINE_AA)
        cv2.line(canvas,(cx+s//5,cy+s//5),(cx-s//8,cy+s//2),hl,4,cv2.LINE_AA)
    elif cat == "RECYCLE":
        for a_deg in [90,210,330]:
            a1=math.radians(a_deg); a2=math.radians(a_deg+120)
            p1=(cx+int(s*.55*math.cos(a1)),cy-int(s*.55*math.sin(a1)))
            p2=(cx+int(s*.55*math.cos(a2)),cy-int(s*.55*math.sin(a2)))
            cv2.line(canvas,p1,p2,hl,4,cv2.LINE_AA)
            ha=int(s*.22)
            for da in [0.7,-0.7]:
                he=a2+math.pi+da
                cv2.line(canvas,p2,(p2[0]+int(ha*math.cos(he)),p2[1]-int(ha*math.sin(he))),hl,4,cv2.LINE_AA)
    else:
        tw=int(s*.7); th=int(s*.75); tx=cx-tw//2; ty=cy-th//2
        body=np.array([[tx+tw//8,ty],[tx+tw-tw//8,ty],[tx+tw,ty+th],[tx,ty+th]],np.int32)
        cv2.fillConvexPoly(canvas,body,hl)
        cv2.rectangle(canvas,(tx-tw//9,ty-th//8),(tx+tw+tw//9,ty+th//12),hl,-1)
        cv2.rectangle(canvas,(cx-tw//5,ty-th*5//16),(cx+tw//5,ty-th//10),hl,-1)
        stripe=tuple(max(0,c//4) for c in col)
        for lx in [tx+tw//4,cx,tx+3*tw//4]:
            cv2.line(canvas,(lx,ty+th//6),(lx,ty+th-th//10),stripe,3,cv2.LINE_AA)

# ───────────────────────────────────────────────
#  MODEL + LABELS
# ───────────────────────────────────────────────
print("\n[1] Loading model_unquant4.tflite ...")
interp = Interpreter(model_path="model_unquant4.tflite")
interp.allocate_tensors()
inp_d = interp.get_input_details()
out_d = interp.get_output_details()
print("    OK")

print("[2] Loading labels4.txt ...")
with open("labels4.txt") as f:
    labels = []
    for line in f:
        parts = line.strip().split(" ",1)
        labels.append(parts[1].strip() if len(parts)>1 else parts[0])
print(f"    {len(labels)} labels: {labels}\n")

# ───────────────────────────────────────────────
#  CAMERA
# ───────────────────────────────────────────────
camera = None

def cam_on():
    global camera
    for attempt in range(3):
        try:
            camera = Picamera2()
            camera.configure(camera.create_preview_configuration(
                main={"size":(820,616),"format":"RGB888"}))
            camera.start()
            time.sleep(2)
            print("    Camera on.")
            return
        except RuntimeError as e:
            print(f"    cam_on attempt {attempt+1} failed: {e}")
            try:
                if camera: camera.close()
            except Exception:
                pass
            camera = None
            time.sleep(1.5)
    print("    ERROR: Could not open camera after 3 attempts.")

def cam_off():
    global camera
    if camera:
        try:
            camera.stop()
            camera.close()
        except Exception as e:
            print(f"    cam_off warning: {e}")
        finally:
            camera = None
            time.sleep(0.5)
        print("    Camera off.")

# ───────────────────────────────────────────────
#  STATE MACHINE
# ───────────────────────────────────────────────
SLEEP, COUNTDOWN, SCAN, QUIZ, RESULT = 0, 2, 3, 4, 5

state      = SLEEP
last_act   = time.time()
SLEEP_AFTER = 600

COUNTDOWN_DUR   = 5.0
countdown_start = 0.0
scan_start      = 0.0
SCAN_DUR        = 5.0
res_at          = 0.0
RES_DUR         = 8.0

predictions = []
conf_accum  = defaultdict(float)
count_accum = defaultdict(int)
frame_n     = 0
CONF_THRESH = 35.0

f_label = f_cat = ""
f_conf  = 0.0
r_happy = True
raw_last = None

# ───────────────────────────────────────────────
#  WINDOW
# ───────────────────────────────────────────────
cv2.namedWindow("Smart Bin", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Smart Bin", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("Smart Bin", np.zeros((H,W,3),np.uint8)); cv2.waitKey(1)
print("Ready.\n")

try:
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        now = time.time()

        if state != SLEEP and now - last_act > SLEEP_AFTER:
            cam_off(); led_clear(); clr(); state=SLEEP; continue

        # ═══════════════════════════════════════
        #  SLEEP  — scene1.png fullscreen
        # ═══════════════════════════════════════
        if state == SLEEP:
            canvas = np.zeros((H,W,3), np.uint8)
            show_png(canvas, "sleep")
            cv2.imshow("Smart Bin", canvas)
            if flags["any"] or key==ord(' '):
                clr(); cam_on(); last_act=now
                countdown_start=now; state=COUNTDOWN
                print("→ Countdown...")
            continue

        # ── camera capture (needed by all states below) ──
        raw_last = camera.capture_array()
        canvas   = cam_frame(raw_last)
        frame_n += 1

        # ═══════════════════════════════════════
        #  COUNTDOWN  —  3  2  1 on camera
        # ═══════════════════════════════════════
        if state == COUNTDOWN:
            last_act   = now
            elapsed_cd = now - countdown_start
            remaining  = COUNTDOWN_DUR - elapsed_cd

            canvas = (canvas.astype(np.float32)*0.55).astype(np.uint8)
            draw_reticle(canvas, W//2, H//2, 200, C_GREEN, now)

            digit = str(max(1, math.ceil(remaining)))
            frac  = remaining - math.floor(remaining)
            scale = 7.0 + 1.5 * (frac ** 0.3)
            dw, dh = tsz(digit, scale, 6, FB)
            cv2.putText(canvas, digit,
                        ((W-dw)//2+4, H//2+dh//2+4),
                        FB, scale, (0,0,0), 10, cv2.LINE_AA)
            cv2.putText(canvas, digit,
                        ((W-dw)//2, H//2+dh//2),
                        FB, scale, C_GREEN, 6, cv2.LINE_AA)
            putc(canvas,"Get ready to scan!", H-60, 1.0, C_WHITE, 2, FB)

            if elapsed_cd >= COUNTDOWN_DUR:
                predictions=[]; conf_accum=defaultdict(float)
                count_accum=defaultdict(int); frame_n=0
                scan_start=now; state=SCAN
                print("→ Scanning...")

        # ═══════════════════════════════════════
        #  SCAN  —  camera + reticle + progress
        # ═══════════════════════════════════════
        elif state == SCAN:
            elapsed  = now - scan_start
            last_act = now
            progress = min(elapsed / SCAN_DUR, 1.0)

            canvas = (canvas.astype(np.float32)*0.72).astype(np.uint8)
            draw_reticle(canvas, W//2, H//2-20, 200, C_GREEN, now)

            pct     = int(progress * 100)
            pb_x=64; pb_y=H-38; pb_w=W-128; pb_h=10
            put(canvas,"ANALYZING",pb_x,pb_y-14,0.80,C_GREEN,2,FB)
            pct_str=f"{pct}%"
            pw=tsz(pct_str,0.80,2,FB)[0]
            put(canvas,pct_str,pb_x+pb_w-pw,pb_y-14,0.80,C_WHITE,2,FB)
            pbar(canvas,elapsed,SCAN_DUR,pb_x,pb_y,pb_w,pb_h,C_GREEN)

            if frame_n % 10 == 0:
                # raw_last is BGR (confirmed) — convert to RGB for the model
                frame3 = raw_last[:, :, :3] if raw_last.shape[2] == 4 else raw_last
                rgb = cv2.cvtColor(np.ascontiguousarray(frame3), cv2.COLOR_BGR2RGB)
                img = cv2.resize(rgb, (224,224))
                img = np.expand_dims(img.astype(np.float32)/255.0, 0)
                interp.set_tensor(inp_d[0]["index"],img)
                interp.invoke()
                preds=interp.get_tensor(out_d[0]["index"])[0]
                idx=int(np.argmax(preds)); conf=float(preds[idx])*100
                top3=np.argsort(preds)[::-1][:min(3,len(labels))]
                print("  "+"  |  ".join(f"{labels[i]} {preds[i]*100:.1f}%" for i in top3))
                if conf>=CONF_THRESH:
                    predictions.append((idx,conf))
                    conf_accum[idx]+=conf; count_accum[idx]+=1
                else:
                    print(f"  (skipped {conf:.1f}%)")

            if elapsed >= SCAN_DUR:
                if conf_accum:
                    best=max(conf_accum,key=lambda x:conf_accum[x]/count_accum[x])
                    f_conf=conf_accum[best]/count_accum[best]
                else:
                    p2=interp.get_tensor(out_d[0]["index"])[0]
                    best=int(np.argmax(p2)); f_conf=float(p2[best])*100
                f_label=labels[best]; f_cat=get_bin(f_label)
                print(f"\n-> '{f_label}' -> {f_cat} ({f_conf:.1f}%)\n")
                state=QUIZ

        # ═══════════════════════════════════════
        #  QUIZ  —  quiz.png fullscreen
        # ═══════════════════════════════════════
        elif state == QUIZ:
            last_act = now
            canvas = np.zeros((H,W,3), np.uint8)
            show_png(canvas, "quiz")

            guess = None
            if   flags["c"] or key==ord('c'): guess="COMPOST"
            elif flags["r"] or key==ord('r'): guess="RECYCLE"
            elif flags["t"] or key==ord('t'): guess="TRASH"

            if guess:
                clr(); last_act=now
                r_happy = (guess == f_cat)
                print(f"  Guess:{guess}  Correct:{f_cat}  -> {'CORRECT' if r_happy else 'WRONG'}")
                led_show(f_cat); res_at=now; state=RESULT

        # ═══════════════════════════════════════
        #  RESULT  —  correct.png or wrong.png
        # ═══════════════════════════════════════
        elif state == RESULT:
            last_act  = now
            elapsed_r = now - res_at
            canvas = np.zeros((H,W,3), np.uint8)

            if r_happy:
                show_png(canvas, "correct")
                # "Recycle: 99%" centred — move CORRECT_TEXT_Y to reposition
                item_str = f"{BIN_LABEL[f_cat]}: {f_conf:.0f}%"
                putc(canvas, item_str, CORRECT_TEXT_Y, 1.8, C_WHITE, 3, FB)

            else:
                show_png(canvas, "wrong")
                # "It is actually"  — move WRONG_LABEL_Y to reposition
                putc(canvas, "It is actually", WRONG_LABEL_Y, 1.1, C_WHITE, 2, FB)
                # "Recycle  99%"    — move WRONG_ITEM_Y to reposition
                item_str = f"{BIN_LABEL[f_cat]}  {f_conf:.0f}%"
                putc(canvas, item_str, WRONG_ITEM_Y, 1.8, C_WHITE, 3, FB)

            # thin countdown bar at very bottom
            bar_col = C_GREEN if r_happy else C_GREY
            pbar(canvas, elapsed_r, RES_DUR, (W-700)//2, H-20, 700, 10, bar_col)

            if elapsed_r >= RES_DUR:
                led_clear(); cam_off(); clr(); state=SLEEP

        # LED auto-off
        if led_off_at and now > led_off_at: led_clear(); led_off_at=0.0

        cv2.imshow("Smart Bin", canvas)
        cv2.setWindowProperty("Smart Bin", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        clr()

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    led_clear(); cam_off(); cv2.destroyAllWindows(); print("Done!")
