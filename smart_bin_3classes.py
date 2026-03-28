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
#  DESIGN TOKENS  (all colours in BGR)
# ───────────────────────────────────────────────
C_BG        = ( 12,  14,  12)
C_WHITE     = (235, 238, 235)
C_DIM       = (105, 110, 105)
C_PANEL     = ( 18,  24,  18)
C_BORDER    = ( 48,  58,  48)

C_GREEN     = ( 72, 200,  82)   # compost
C_BLUE      = (215, 148,  44)   # recycle  (BGR of rgb 44,148,215)
C_GREY      = (200, 200, 200)   # trash

BIN_COL  = {"COMPOST": C_GREEN, "RECYCLE": C_BLUE, "TRASH": C_GREY}
BIN_LABEL= {"COMPOST": "Compost","RECYCLE": "Recycle","TRASH": "Trash"}

F  = cv2.FONT_HERSHEY_SIMPLEX
FB = cv2.FONT_HERSHEY_DUPLEX

# ───────────────────────────────────────────────
#  LEDs
# ───────────────────────────────────────────────
strip = PixelStrip(180, 18, brightness=200)
strip.begin()
ZONES   = {"COMPOST":(0,60),"RECYCLE":(60,120),"TRASH":(120,180)}
LCOLORS = {
    "COMPOST": Color(0,210,70),
    "RECYCLE": Color(44,148,215),
    "TRASH":   Color(210,210,210),
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

flags      = {"any":False,"c":False,"r":False,"t":False}
press_time = {"c":-99.0,"r":-99.0,"t":-99.0}

def on_c(): flags["c"]=flags["any"]=True; press_time["c"]=time.time()
def on_r(): flags["r"]=flags["any"]=True; press_time["r"]=time.time()
def on_t(): flags["t"]=flags["any"]=True; press_time["t"]=time.time()

B_COMPOST.when_pressed = on_c
B_RECYCLE.when_pressed = on_r
B_TRASH.when_pressed   = on_t

def clr(): flags.update({"any":False,"c":False,"r":False,"t":False})

# ───────────────────────────────────────────────
#  BIN MAPPING  (3-class)
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
    """Crisp text — hard outline then fill, no sub-pixel blur."""
    cv2.putText(canvas, txt, (x, y), font, scale, (0,0,0), thick+3, cv2.LINE_8)
    cv2.putText(canvas, txt, (x, y), font, scale, col,     thick,   cv2.LINE_8)

def putc(canvas, txt, y, scale, col, thick=2, font=F):
    """Horizontally centred crisp text."""
    w2 = tsz(txt, scale, thick, font)[0]
    put(canvas, txt, (W-w2)//2, y, scale, col, thick, font)

def rr(canvas, x, y, w2, h2, col, r=16, t=-1):
    """Rounded rectangle."""
    r = min(r, w2//2, h2//2)
    cv2.rectangle(canvas,(x+r,y),(x+w2-r,y+h2), col, t)
    cv2.rectangle(canvas,(x,y+r),(x+w2,y+h2-r), col, t)
    for px, py in [(x+r,y+r),(x+w2-r,y+r),(x+r,y+h2-r),(x+w2-r,y+h2-r)]:
        cv2.circle(canvas,(px,py),r,col,t)

def panel_box(canvas, x, y, w2, h2, r=18):
    """Dark semi-transparent panel with border."""
    ov = canvas.copy()
    rr(ov, x, y, w2, h2, C_PANEL, r=r)
    cv2.addWeighted(ov, 0.85, canvas, 0.15, 0, canvas)
    rr(canvas, x, y, w2, h2, C_BORDER, r=r, t=2)

def pbar(canvas, elapsed, total, x, y, bw, bh, col):
    rr(canvas, x, y, bw, bh, (28,33,28), r=bh//2)
    fill = max(4, int(bw * min(elapsed/total, 1.0)))
    rr(canvas, x, y, fill, bh, col, r=bh//2)

def to_bgr(raw):
    if raw.ndim==3 and raw.shape[2]==4:
        return np.ascontiguousarray(raw[:,:,1:4])
    return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)

def cam_frame(raw):
    bgr = to_bgr(raw)
    ch, cw = bgr.shape[:2]
    s = max(W/cw, H/ch)
    nw, nh = int(cw*s), int(ch*s)
    big = cv2.resize(bgr,(nw,nh),interpolation=cv2.INTER_LINEAR)
    x0, y0 = (nw-W)//2, (nh-H)//2
    return np.ascontiguousarray(big[y0:y0+H, x0:x0+W])

# ───────────────────────────────────────────────
#  SCAN RETICLE
# ───────────────────────────────────────────────
def draw_reticle(canvas, cx, cy, size, col, t_now):
    """Corner-bracket viewfinder + crosshair + slow scan line."""
    arm = size // 3
    thk = 3

    # four corner L-brackets
    for sx, sy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
        bx = cx + sx * size
        by = cy + sy * size
        cv2.line(canvas,(bx,by),(bx - sx*arm, by),    col, thk, cv2.LINE_AA)
        cv2.line(canvas,(bx,by),(bx, by - sy*arm),    col, thk, cv2.LINE_AA)

    # crosshair (gap in centre)
    gap = 14
    cv2.line(canvas,(cx-size+arm, cy),(cx-gap, cy), col, 1, cv2.LINE_AA)
    cv2.line(canvas,(cx+gap, cy),(cx+size-arm, cy), col, 1, cv2.LINE_AA)
    cv2.line(canvas,(cx, cy-size+arm),(cx, cy-gap), col, 1, cv2.LINE_AA)
    cv2.line(canvas,(cx, cy+gap),(cx, cy+size-arm), col, 1, cv2.LINE_AA)

    # centre dot
    cv2.circle(canvas,(cx,cy), 4, col, -1, cv2.LINE_AA)

    # single horizontal scan sweep (1 line, no glow)
    sweep = int((t_now % 2.5) / 2.5 * (2*size)) - size
    sy2 = cy + sweep
    if cy - size < sy2 < cy + size:
        cv2.line(canvas,(cx-size, sy2),(cx+size, sy2), col, 1, cv2.LINE_AA)

# ───────────────────────────────────────────────
#  QUIZ CIRCLE BUTTON
# ───────────────────────────────────────────────
def quiz_circle(canvas, cx, cy, rad, col, cat, label, pt):
    """Clean circle button — squish on press, crisp label below."""
    age = time.time() - pt
    r = int(rad * (1.0 - 0.15 * math.sin(age/0.3*math.pi))) \
        if 0 < age < 0.3 else rad

    # dark fill
    dark = tuple(max(0, c//5) for c in col)
    cv2.circle(canvas,(cx,cy), r, dark, -1, cv2.LINE_AA)
    # coloured border
    cv2.circle(canvas,(cx,cy), r, col,  5,  cv2.LINE_AA)
    # highlight arc
    hl = tuple(min(255, int(c*1.5)) for c in col)
    cv2.ellipse(canvas,(cx-r//6,cy-r//6),(r//4,r//5),
                310, 0, 120, hl, 3, cv2.LINE_AA)

    # icon
    _icon(canvas, cx, cy - r//10, int(r*0.42), cat, col)

    # label — pre-measure so it's perfectly centred under this circle
    lbl_w = tsz(label, 1.1, 2, FB)[0]
    put(canvas, label, cx - lbl_w//2, cy + r + 46, 1.1, col, 2, FB)

    # press flash
    if 0 < age < 0.22:
        alpha = 1.0 - age/0.22
        flash = canvas.copy()
        cv2.circle(flash,(cx,cy), r, tuple(min(255,int(c*0.7)) for c in col),-1,cv2.LINE_AA)
        cv2.addWeighted(flash, alpha*0.30, canvas, 1-alpha*0.30, 0, canvas)

def _icon(canvas, cx, cy, s, cat, col):
    hl = tuple(min(255, int(c*1.5)) for c in col)
    if cat == "COMPOST":
        cv2.ellipse(canvas,(cx,cy-s//8),(int(s*.55),int(s*.38)),-40,0,360,hl,-1,cv2.LINE_AA)
        vein = tuple(max(0,c//4) for c in col)
        cv2.line(canvas,(cx-s//3,cy+s//5),(cx+s//5,cy-s//2),vein,3,cv2.LINE_AA)
        cv2.line(canvas,(cx+s//5,cy+s//5),(cx-s//8,cy+s//2),hl,4,cv2.LINE_AA)
    elif cat == "RECYCLE":
        for a_deg in [90, 210, 330]:
            a1 = math.radians(a_deg); a2 = math.radians(a_deg+120)
            p1=(cx+int(s*.55*math.cos(a1)),cy-int(s*.55*math.sin(a1)))
            p2=(cx+int(s*.55*math.cos(a2)),cy-int(s*.55*math.sin(a2)))
            cv2.line(canvas,p1,p2,hl,4,cv2.LINE_AA)
            ha=int(s*.22)
            for da in [0.7,-0.7]:
                he=a2+math.pi+da
                cv2.line(canvas,p2,(p2[0]+int(ha*math.cos(he)),p2[1]-int(ha*math.sin(he))),hl,4,cv2.LINE_AA)
    else:  # TRASH
        tw=int(s*.7); th=int(s*.75)
        tx=cx-tw//2; ty=cy-th//2
        body=np.array([[tx+tw//8,ty],[tx+tw-tw//8,ty],[tx+tw,ty+th],[tx,ty+th]],np.int32)
        cv2.fillConvexPoly(canvas,body,hl)
        cv2.rectangle(canvas,(tx-tw//9,ty-th//8),(tx+tw+tw//9,ty+th//12),hl,-1)
        cv2.rectangle(canvas,(cx-tw//5,ty-th*5//16),(cx+tw//5,ty-th//10),hl,-1)
        stripe=tuple(max(0,c//4) for c in col)
        for lx in [tx+tw//4, cx, tx+3*tw//4]:
            cv2.line(canvas,(lx,ty+th//6),(lx,ty+th-th//10),stripe,3,cv2.LINE_AA)

# ───────────────────────────────────────────────
#  FACE
# ───────────────────────────────────────────────
def draw_face(canvas, cx, cy, r, happy):
    col = C_GREEN if happy else (110,130,230)
    dark= tuple(max(0,c//5) for c in col)
    hl  = tuple(min(255,c+80) for c in col)
    cv2.circle(canvas,(cx,cy),r,dark,-1,cv2.LINE_AA)
    cv2.circle(canvas,(cx,cy),r,col, 4, cv2.LINE_AA)
    ex=r//3
    for ox in [-ex,ex]:
        cv2.circle(canvas,(cx+ox,cy-r//6),r//8,col,-1,cv2.LINE_AA)
        cv2.circle(canvas,(cx+ox-r//18,cy-r//6-r//16),r//14,hl,-1,cv2.LINE_AA)
    if happy:
        cv2.ellipse(canvas,(cx,cy+r//10),(r//2,r//3),0,15,165,col,4,cv2.LINE_AA)
    else:
        cv2.ellipse(canvas,(cx,cy+r//2),(r//2,r//3),0,195,345,col,4,cv2.LINE_AA)

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
    camera = Picamera2()
    camera.configure(camera.create_preview_configuration(
        main={"size":(820,616),"format":"RGB888"}))
    camera.start(); time.sleep(2); print("    Camera on.")

def cam_off():
    global camera
    if camera: camera.stop(); camera=None; print("    Camera off.")

# ───────────────────────────────────────────────
#  STATE MACHINE
# ───────────────────────────────────────────────
SLEEP,IDLE,SCAN,QUIZ,RESULT = 0,1,2,3,4
state      = SLEEP
last_act   = time.time()
SLEEP_AFTER= 600

scan_start = 0.0; SCAN_DUR = 5.0
res_at     = 0.0; RES_DUR  = 13.0

predictions = []
conf_accum  = defaultdict(float)
count_accum = defaultdict(int)
frame_n     = 0
CONF_THRESH = 35.0

f_label=f_cat=""
f_conf =0.0
r_msg=r_sub=""
r_happy  =True
raw_last =None

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
        #  SLEEP
        # ═══════════════════════════════════════
        if state == SLEEP:
            canvas = np.full((H,W,3), C_BG, np.uint8)

            # ── title ──
            putc(canvas, "SMART BIN",      H//2 - 52, 4.5, C_WHITE, 6, FB)
            putc(canvas, "Learn where your waste belongs",
                         H//2 + 28, 1.1, C_DIM, 2)

            # ── CTA button ──
            btn_w, btn_h = 440, 68
            bx = (W-btn_w)//2; by = H//2 + 82
            rr(canvas, bx, by, btn_w, btn_h, C_GREEN, r=34)
            btxt = "PRESS ANY BUTTON"
            bw2  = tsz(btxt, 1.1, 2)[0]
            put(canvas, btxt, (W-bw2)//2, by+47, 1.1, C_BG, 2)

            cv2.imshow("Smart Bin", canvas)
            if flags["any"] or key==ord(' '):
                clr(); cam_on(); last_act=now; state=IDLE
            continue

        # ── capture ─────────────────────────────
        raw_last = camera.capture_array()
        canvas   = cam_frame(raw_last)
        frame_n += 1

        # ═══════════════════════════════════════
        #  IDLE  —  camera + reticle
        # ═══════════════════════════════════════
        if state == IDLE:
            canvas = (canvas.astype(np.float32)*0.45).astype(np.uint8)

            # top bar
            canvas[:80,:] = (canvas[:80].astype(np.float32)*0.25).astype(np.uint8)
            cv2.rectangle(canvas,(0,0),(W,80),(15,22,15),-1)
            putc(canvas,"SMART BIN", 56, 2.0, C_GREEN, 3, FB)

            # viewfinder reticle (static — no animation for speed)
            draw_reticle(canvas, W//2, H//2+20, 145, C_GREEN, now)

            # instruction card — sized exactly to text
            L1 = "Hold item up to camera"
            L2 = "then press any button to scan"
            w1 = tsz(L1, 1.15, 2)[0]
            w2_ = tsz(L2, 0.85, 2)[0]
            cw2 = max(w1, w2_) + 56
            ch2 = 108
            cx2 = (W-cw2)//2; cy2 = H-190
            panel_box(canvas, cx2, cy2, cw2, ch2, r=16)
            put(canvas, L1, cx2+28, cy2+48,  1.15, C_WHITE, 2)
            put(canvas, L2, cx2+28, cy2+88,  0.85, C_DIM,   2)

            if flags["any"] or key==ord(' '):
                clr(); last_act=now
                predictions=[]; conf_accum=defaultdict(float)
                count_accum=defaultdict(int); frame_n=0
                scan_start=now; state=SCAN
                print("→ Scanning...")

        # ═══════════════════════════════════════
        #  SCAN  —  camera tracking view
        # ═══════════════════════════════════════
        elif state == SCAN:
            elapsed  = now - scan_start
            last_act = now

            # camera slightly dimmed but readable
            canvas = (canvas.astype(np.float32)*0.70).astype(np.uint8)

            # reticle
            draw_reticle(canvas, W//2, H//2, 210, C_GREEN, now)

            # top-left status chip
            pct = int(min(elapsed/SCAN_DUR,1.0)*100)
            status = f"Analysing...  {pct}%"
            sw  = tsz(status, 0.9, 2)[0] + 44
            panel_box(canvas, 22, 22, sw, 52, r=10)
            put(canvas, status, 44, 58, 0.9, C_GREEN, 2)

            # top-right REC dot
            cv2.circle(canvas,(W-50,40),10,(55,55,220),-1,cv2.LINE_AA)
            put(canvas,"REC", W-84, 52, 0.75, C_WHITE, 1)

            # bottom progress bar
            pb_w=840; pb_h=16
            pbar(canvas, elapsed, SCAN_DUR, (W-pb_w)//2, H-50, pb_w, pb_h, C_GREEN)

            # inference
            if frame_n%10==0:
                img=cv2.resize(to_bgr(raw_last),(224,224))
                img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img=np.expand_dims(img.astype(np.float32)/255.0,0)
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

            if elapsed>=SCAN_DUR:
                if conf_accum:
                    best=max(conf_accum,key=lambda x:conf_accum[x]/count_accum[x])
                    f_conf=conf_accum[best]/count_accum[best]
                else:
                    p2=interp.get_tensor(out_d[0]["index"])[0]
                    best=int(np.argmax(p2)); f_conf=float(p2[best])*100
                f_label=labels[best]; f_cat=get_bin(f_label)
                print(f"\n→ '{f_label}' → {f_cat} ({f_conf:.1f}%)\n")
                state=QUIZ

        # ═══════════════════════════════════════
        #  QUIZ  —  solid dark bg, 3 circles
        # ═══════════════════════════════════════
        elif state == QUIZ:
            last_act = now
            canvas[:] = C_BG   # solid dark, no camera — clean & readable

            # header
            putc(canvas,"Where does it go?", 96, 2.6, C_WHITE, 4, FB)

            # item chip
            chip = f"{f_label}   {f_conf:.0f}% confident"
            cw3  = tsz(chip, 0.9, 2)[0] + 50
            cx3  = (W-cw3)//2
            panel_box(canvas, cx3, 118, cw3, 48, r=24)
            put(canvas, chip, cx3+25, 150, 0.9, C_DIM, 2)

            # circles — evenly spaced
            rad  = 132
            gap  = 72
            tot  = 3*2*rad + 2*gap
            bx0  = (W-tot)//2 + rad
            by0  = H//2 + 52

            quiz_circle(canvas, bx0,              by0, rad, C_GREEN,  "COMPOST", "Compost", press_time["c"])
            quiz_circle(canvas, bx0+2*rad+gap,    by0, rad, C_BLUE,   "RECYCLE", "Recycle", press_time["r"])
            quiz_circle(canvas, bx0+2*(2*rad+gap),by0, rad, C_GREY,   "TRASH",   "Trash",   press_time["t"])

            # hint
            putc(canvas,"Press the matching coloured button", H-34, 0.88, C_DIM, 2)

            guess=None
            if   flags["c"] or key==ord('c'): guess="COMPOST"
            elif flags["r"] or key==ord('r'): guess="RECYCLE"
            elif flags["t"] or key==ord('t'): guess="TRASH"

            if guess:
                clr(); last_act=now
                print(f"  Guess:{guess}  Correct:{f_cat}")
                if guess==f_cat:
                    r_msg="Amazing! You got it!"; r_sub=f"It goes in {BIN_LABEL[f_cat]}!"; r_happy=True
                else:
                    r_msg="Good try!"; r_sub=f"It actually goes in {BIN_LABEL[f_cat]}"; r_happy=False
                led_show(f_cat); res_at=now; state=RESULT

        # ═══════════════════════════════════════
        #  RESULT
        # ═══════════════════════════════════════
        elif state == RESULT:
            last_act = now
            elapsed_r = now - res_at
            col_r = BIN_COL[f_cat]

            # tinted dark bg
            tint = tuple(max(0,c//9) for c in col_r)
            canvas[:] = tint

            # accent bars top/bottom
            cv2.rectangle(canvas,(0,0),(W,8),col_r,-1)
            cv2.rectangle(canvas,(0,H-8),(W,H),col_r,-1)

            # face
            draw_face(canvas, W//2, 188, 92, r_happy)

            # result card
            crd_w=1000; crd_h=195
            crd_x=(W-crd_w)//2; crd_y=318
            panel_box(canvas, crd_x, crd_y, crd_w, crd_h, r=24)
            # coloured left accent stripe on card
            rr(canvas, crd_x, crd_y, 10, crd_h, col_r, r=5)

            msg_col=(90,235,105) if r_happy else (115,150,255)
            putc(canvas, r_msg, crd_y+90,  2.2, msg_col, 3, FB)
            putc(canvas, r_sub, crd_y+148, 1.1, C_WHITE, 2)

            # bin badge
            badge = BIN_LABEL[f_cat].upper()
            bw2   = tsz(badge, 1.8, 3, FB)[0] + 60
            bh2   = 66
            bx2   = (W-bw2)//2; by2 = crd_y+crd_h+20
            rr(canvas, bx2, by2, bw2, bh2, col_r, r=33)
            dark_txt = tuple(max(0,c//4) for c in col_r)
            putc(canvas, badge, by2+50, 1.8, dark_txt, 3, FB)

            # confidence line
            info = f"{f_label}  —  {f_conf:.0f}% confidence"
            putc(canvas, info, by2+bh2+34, 0.85, C_DIM, 1)

            # countdown bar
            pbar(canvas, elapsed_r, RES_DUR, (W-700)//2, H-52, 700, 16, col_r)
            putc(canvas,"Returning to scan...", H-18, 0.72, C_DIM, 1)

            if elapsed_r >= RES_DUR:
                led_clear(); clr(); state=IDLE

        # LED auto-off
        if led_off_at and now>led_off_at: led_clear(); led_off_at=0.0

        cv2.imshow("Smart Bin", canvas)
        cv2.setWindowProperty("Smart Bin",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        clr()

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    led_clear(); cam_off(); cv2.destroyAllWindows(); print("Done!")
