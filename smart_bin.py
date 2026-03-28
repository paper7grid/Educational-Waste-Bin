# smart_bin.py  —  RUN WITH: sudo python3 smart_bin.py

import cv2, numpy as np, time
from ai_edge_litert.interpreter import Interpreter
from picamera2 import Picamera2
from collections import defaultdict
from rpi_ws281x import PixelStrip, Color
from gpiozero import Button

# ═══════════════════════════════════════════════════════
#  SCREEN  (change W/H to match your display if needed)
# ═══════════════════════════════════════════════════════
W, H = 1280, 720

# ═══════════════════════════════════════════════════════
#  LEDS
# ═══════════════════════════════════════════════════════
strip = PixelStrip(180, 18, brightness=190)
strip.begin()
ZONES  = {"COMPOST":(0,60), "RECYCLE":(60,120), "TRASH":(120,180)}
LCOLORS= {"COMPOST":Color(0,255,0), "RECYCLE":Color(0,0,255), "TRASH":Color(255,255,255)}
led_off_at = 0

def led_clear():
    for i in range(180): strip.setPixelColor(i, Color(0,0,0))
    strip.show()

def led_show(cat, dur=15):
    global led_off_at
    led_clear()
    s,e = ZONES[cat]
    for i in range(s,e): strip.setPixelColor(i, LCOLORS[cat])
    strip.show(); led_off_at = time.time()+dur

# ═══════════════════════════════════════════════════════
#  BUTTONS   GPIO 2=Compost  3=Recycle  4=Trash
# ═══════════════════════════════════════════════════════
B_COMPOST = Button(2, bounce_time=0.1)
B_RECYCLE = Button(3, bounce_time=0.1)
B_TRASH   = Button(4, bounce_time=0.1)

flags = {"any":False,"c":False,"r":False,"t":False}
press_anim = {"c":0,"r":0,"t":0}   # timestamp of last press for animation

def on_c(): flags["c"]=flags["any"]=True; press_anim["c"]=time.time()
def on_r(): flags["r"]=flags["any"]=True; press_anim["r"]=time.time()
def on_t(): flags["t"]=flags["any"]=True; press_anim["t"]=time.time()

B_COMPOST.when_pressed=on_c
B_RECYCLE.when_pressed=on_r
B_TRASH.when_pressed  =on_t

def clr(): flags.update({"any":False,"c":False,"r":False,"t":False})

# ═══════════════════════════════════════════════════════
#  BIN MAPPING
# ═══════════════════════════════════════════════════════
COMPOST_SET = {'orange peels','banana','used fiber bowl'}
RECYCLE_SET = {'cardboard','paper','paper bag','glass bottle',
               'glass jar','milk carton','water bottle','soda can'}

def get_bin(label):
    l = label.strip().lower()
    if l in COMPOST_SET: return "COMPOST"
    if l in RECYCLE_SET: return "RECYCLE"
    return "TRASH"

BIN_COL = {                          # BGR
    "COMPOST": (70,  200, 80 ),
    "RECYCLE": (200, 180, 50 ),
    "TRASH":   (180, 160, 220),
}
BIN_LABEL = {"COMPOST":"Compost","RECYCLE":"Recycle","TRASH":"Trash"}

# ═══════════════════════════════════════════════════════
#  FONT / COLOUR HELPERS
# ═══════════════════════════════════════════════════════
F = cv2.FONT_HERSHEY_SIMPLEX

def ctext(canvas, text, y, scale, col, thick=2, alpha_bg=False):
    """Draw horizontally-centred text with optional dark backing."""
    sz  = cv2.getTextSize(text, F, scale, thick)[0]
    x   = (canvas.shape[1]-sz[0])//2
    if alpha_bg:
        pad=14
        ov=canvas.copy()
        cv2.rectangle(ov,(x-pad,y-sz[1]-pad),(x+sz[0]+pad,y+pad),(0,0,0),-1)
        cv2.addWeighted(ov,0.55,canvas,0.45,0,canvas)
    # soft shadow
    cv2.putText(canvas,text,(x+2,y+2),F,scale,(0,0,0),thick+2,cv2.LINE_AA)
    cv2.putText(canvas,text,(x,y),    F,scale,col,    thick,  cv2.LINE_AA)

def rr(canvas,x,y,w,h,col,r=20,t=-1):
    if r*2>w: r=w//2
    if r*2>h: r=h//2
    cv2.rectangle(canvas,(x+r,y),(x+w-r,y+h),col,t)
    cv2.rectangle(canvas,(x,y+r),(x+w,y+h-r),col,t)
    for px,py in[(x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)]:
        cv2.circle(canvas,(px,py),r,col,t)

def dark_overlay(canvas, alpha=0.55):
    ov=canvas.copy(); canvas[:]=0
    cv2.addWeighted(ov,1-alpha,canvas,alpha,0,canvas)

def corners(canvas,col,m=60,l=50,t=5):
    for cx,cy in[(m,m),(W-m,m),(m,H-m),(W-m,H-m)]:
        dx=1 if cx<W//2 else -1; dy=1 if cy<H//2 else -1
        cv2.line(canvas,(cx,cy),(cx+dx*l,cy),col,t,cv2.LINE_AA)
        cv2.line(canvas,(cx,cy),(cx,cy+dy*l),col,t,cv2.LINE_AA)

def pbar(canvas,elapsed,total,x,y,bw,bh,col):
    rr(canvas,x,y,bw,bh,(30,32,40),r=bh//2)
    fill=max(0,int(bw*min(elapsed/total,1.0)))
    if fill>bh: rr(canvas,x,y,fill,bh,col,r=bh//2)

def scan_line(canvas,t):
    sy=int((t%2.0)/2.0*H)
    ov=canvas.copy()
    cv2.line(ov,(0,sy),(W,sy),(120,180,255),4)
    for d in range(1,60):
        if sy-d>=0:
            a=max(0,130-d*2)
            cv2.line(ov,(0,sy-d),(W,sy-d),(60,100,200),1)
    cv2.addWeighted(ov,0.5,canvas,0.5,0,canvas)

def to_bgr(raw):
    if raw.ndim==3 and raw.shape[2]==4:
        return np.ascontiguousarray(raw[:,:,1:4])
    return cv2.cvtColor(raw,cv2.COLOR_RGB2BGR)

def cam_frame(raw):
    """Return camera image scaled & cropped to fill W×H."""
    bgr = to_bgr(raw)
    ch,cw = bgr.shape[:2]
    scale = max(W/cw, H/ch)
    nw,nh = int(cw*scale), int(ch*scale)
    big   = cv2.resize(bgr,(nw,nh),interpolation=cv2.INTER_LINEAR)
    x0    = (nw-W)//2; y0=(nh-H)//2
    return np.ascontiguousarray(big[y0:y0+H, x0:x0+W])

# ═══════════════════════════════════════════════════════
#  BIN CIRCLE  (kid-friendly large coloured circle)
# ═══════════════════════════════════════════════════════
def bin_circle(canvas, cx, cy, rad, col, label, pressed_ts):
    now = time.time()
    age = now - pressed_ts
    # squish animation for 0.3 s after press
    if age < 0.3:
        shrink = 1.0 - 0.18*np.sin(age/0.3*np.pi)
        rad = int(rad*shrink)

    dark = tuple(max(0,c//4) for c in col)
    # glow ring
    cv2.circle(canvas,(cx,cy),rad+8,tuple(c//2 for c in col),4,cv2.LINE_AA)
    # fill
    cv2.circle(canvas,(cx,cy),rad,dark,-1,cv2.LINE_AA)
    # border
    cv2.circle(canvas,(cx,cy),rad,col,5,cv2.LINE_AA)
    # inner highlight
    cv2.circle(canvas,(cx-rad//5,cy-rad//5),rad//5,
               tuple(min(255,c+80) for c in col),-1,cv2.LINE_AA)

    # label below circle
    ls=cv2.getTextSize(label,F,1.0,2)[0]
    cv2.putText(canvas,label,
                (cx-ls[0]//2, cy+rad+44),
                F,1.0,col,2,cv2.LINE_AA)

# ═══════════════════════════════════════════════════════
#  HAPPY / SAD FACE
# ═══════════════════════════════════════════════════════
def draw_face(canvas, cx, cy, r, happy=True, col=(240,220,60)):
    dark=tuple(max(0,c//3) for c in col)
    cv2.circle(canvas,(cx,cy),r,dark,-1,cv2.LINE_AA)
    cv2.circle(canvas,(cx,cy),r,col,4,cv2.LINE_AA)
    # eyes
    ex=r//3
    cv2.circle(canvas,(cx-ex,cy-r//5),r//7,col,-1,cv2.LINE_AA)
    cv2.circle(canvas,(cx+ex,cy-r//5),r//7,col,-1,cv2.LINE_AA)
    # mouth
    if happy:
        cv2.ellipse(canvas,(cx,cy+r//8),(r//2,r//3),0,10,170,col,4,cv2.LINE_AA)
    else:
        cv2.ellipse(canvas,(cx,cy+r//2),(r//2,r//3),0,190,350,col,4,cv2.LINE_AA)

# ═══════════════════════════════════════════════════════
#  MODEL + LABELS
# ═══════════════════════════════════════════════════════
print("\n[1] Loading model...")
interp=Interpreter(model_path='model_unquant2.tflite')
interp.allocate_tensors()
inp=interp.get_input_details(); outp=interp.get_output_details()
print("✓ Model ready")

print("[2] Loading labels...")
with open('labels3.txt') as f:
    labels=[]
    for line in f:
        p=line.strip().split(' ',1)
        labels.append(p[1].strip() if len(p)>1 else p[0])
print(f"✓ {len(labels)} labels")

# ═══════════════════════════════════════════════════════
#  CAMERA
# ═══════════════════════════════════════════════════════
camera=None

def cam_on():
    global camera
    camera=Picamera2()
    camera.configure(camera.create_preview_configuration(
        main={"size":(820,616),"format":"RGB888"}))
    camera.start(); time.sleep(2); print("✓ Camera on")

def cam_off():
    global camera
    if camera: camera.stop(); camera=None; print("✓ Camera off")

# ═══════════════════════════════════════════════════════
#  STATE MACHINE
# ═══════════════════════════════════════════════════════
SLEEP,IDLE,SCAN,QUIZ,RESULT=0,1,2,3,4
state=SLEEP; last_act=time.time(); SLEEP_AFTER=600

col_start=0.0; COL_DUR=5
res_at=0.0;    RES_DUR=15

ph=[]; ct=defaultdict(float); pcount=defaultdict(int)
f_label=f_cat=""; f_conf=0.0
r_msg=r_sub=""; r_happy=True
frame_n=0; THRESH=40.0
raw_last=None   # last camera frame

# ═══════════════════════════════════════════════════════
#  WINDOW
# ═══════════════════════════════════════════════════════
cv2.namedWindow("Smart Bin",cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Smart Bin",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.imshow("Smart Bin",np.zeros((H,W,3),np.uint8)); cv2.waitKey(1)
print("\nReady — waiting for button...\n")

try:
  while True:
    key=cv2.waitKey(1)&0xFF
    if key==27: break
    now=time.time()

    # ══════════════════════════════════════════════════
    #  SLEEP SCREEN
    # ══════════════════════════════════════════════════
    if state==SLEEP:
        t=now
        canvas=np.zeros((H,W,3),np.uint8)

        # Subtle radial gradient feel — concentric dark circles
        for r2 in range(500,0,-80):
            v=max(0,18-int((500-r2)/40))
            cv2.circle(canvas,(W//2,H//2),r2,(v,v,v+4),-1,cv2.LINE_AA)

        # Pulsing ring
        pulse=int(10*np.sin(t*2)+10)
        cv2.circle(canvas,(W//2,H//2),200+pulse,(40,60,40),2,cv2.LINE_AA)
        cv2.circle(canvas,(W//2,H//2),220+pulse,(30,50,30),1,cv2.LINE_AA)

        # Title — big, white, bold
        ctext(canvas,"SMART BIN",  H//2-70, 4.0,(220,235,220),6)
        ctext(canvas,"Educational Waste Sorter",
              H//2+10, 1.0,(120,140,120),2)

        # Press prompt with backing pill
        pill_w,pill_h=520,65
        px2=(W-pill_w)//2; py2=H//2+90
        rr(canvas,px2,py2,pill_w,pill_h,(30,45,30),r=32)
        rr(canvas,px2,py2,pill_w,pill_h,(60,130,70),r=32,t=2)
        ctext(canvas,"Press any button to begin",py2+44,0.85,(180,230,180),2)

        cv2.imshow("Smart Bin",canvas)
        if flags["any"] or key==ord(' '):
            clr(); cam_on(); last_act=now; state=IDLE
        continue

    # ══════════════════════════════════════════════════
    #  AUTO-SLEEP
    # ══════════════════════════════════════════════════
    if now-last_act>SLEEP_AFTER:
        cam_off(); led_clear(); clr(); state=SLEEP; continue

    # ══════════════════════════════════════════════════
    #  CAPTURE
    # ══════════════════════════════════════════════════
    raw_last=camera.capture_array()
    canvas=cam_frame(raw_last)   # full-screen camera
    frame_n+=1

    # ══════════════════════════════════════════════════
    #  IDLE
    # ══════════════════════════════════════════════════
    if state==IDLE:
        # light vignette
        ov=canvas.copy(); canvas[:]=0
        cv2.addWeighted(ov,0.82,canvas,0.18,0,canvas)
        corners(canvas,(80,200,90))

        # Top bar
        rr(canvas,0,0,W,90,(0,0,0),r=0)
        cv2.addWeighted(canvas[:90].copy(),0.0,canvas[:90],1.0,0,canvas[:90])
        rr(canvas,0,0,W,90,(10,18,10),r=0)
        ctext(canvas,"SMART BIN",62,2.2,(100,220,110),3)

        # Centre card
        cw2=680; cy2=H//2-60
        rr(canvas,(W-cw2)//2,cy2,cw2,130,(0,0,0),r=30)
        ov2=canvas.copy()
        rr(ov2,(W-cw2)//2,cy2,cw2,130,(8,20,8),r=30)
        cv2.addWeighted(ov2,0.75,canvas,0.25,0,canvas)
        ctext(canvas,"Hold item up to the camera",cy2+55,1.05,(220,235,220),2)
        ctext(canvas,"then press any button",      cy2+100,0.8,(130,170,130),1)

        # Bottom pill
        pw3=380; ph3=70; py3=H-110
        rr(canvas,(W-pw3)//2,py3,pw3,ph3,(50,140,60),r=35)
        ctext(canvas,"PRESS TO SCAN",py3+47,1.0,(10,20,10),2)

        if flags["any"] or key==ord(' '):
            clr(); last_act=now
            ph=[]; ct=defaultdict(float); pcount=defaultdict(int)
            frame_n=0; col_start=now; state=SCAN
            print("Scanning...")

    # ══════════════════════════════════════════════════
    #  SCAN
    # ══════════════════════════════════════════════════
    elif state==SCAN:
        elapsed=now-col_start; last_act=now

        # camera slightly dimmed
        ov=canvas.copy(); canvas[:]=0
        cv2.addWeighted(ov,0.78,canvas,0.22,0,canvas)
        scan_line(canvas,elapsed)
        corners(canvas,(80,150,255))

        # Top bar
        rr(canvas,0,0,W,90,(0,0,8),r=0)
        ctext(canvas,"SCANNING...",62,2.2,(100,160,255),3)

        # Progress bar bottom
        pb_w=900; pb_h=18
        pbar(canvas,elapsed,COL_DUR,(W-pb_w)//2,H-70,pb_w,pb_h,(80,150,255))
        pct=int(min(elapsed/COL_DUR,1.0)*100)
        ctext(canvas,f"{pct}%",H-30,0.85,(100,140,200),1)

        # Run model
        if frame_n%10==0:
            img=cv2.resize(to_bgr(raw_last),(224,224))
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=np.expand_dims(img.astype(np.float32)/255.0,0)
            interp.set_tensor(inp[0]['index'],img)
            interp.invoke()
            preds=interp.get_tensor(outp[0]['index'])[0]
            idx=int(np.argmax(preds)); conf=preds[idx]*100
            t3=np.argsort(preds)[::-1][:3]
            print(f"  {labels[t3[0]]} {preds[t3[0]]*100:.1f}%  |  "
                  f"{labels[t3[1]]} {preds[t3[1]]*100:.1f}%  |  "
                  f"{labels[t3[2]]} {preds[t3[2]]*100:.1f}%")
            if conf>=THRESH:
                ph.append((idx,conf)); ct[idx]+=conf; pcount[idx]+=1
            else:
                print(f"  (skipped {conf:.1f}%)")

        if elapsed>=COL_DUR:
            if ct:
                best=max(ct,key=lambda x:ct[x]/pcount[x])
                f_conf=ct[best]/pcount[best]
            else:
                p2=interp.get_tensor(outp[0]['index'])[0]
                best=int(np.argmax(p2)); f_conf=float(p2[best]*100)
            f_label=labels[best]; f_cat=get_bin(f_label)
            print(f"\nResult: {f_label} → {f_cat}")
            r_msg=r_sub=""; state=QUIZ

    # ══════════════════════════════════════════════════
    #  QUIZ
    # ══════════════════════════════════════════════════
    elif state==QUIZ:
        last_act=now

        # heavy dark overlay on camera
        ov=canvas.copy(); canvas[:]=0
        cv2.addWeighted(ov,0.35,canvas,0.65,0,canvas)

        # Header
        rr(canvas,0,0,W,100,(0,0,0),r=0)
        ctext(canvas,"Where does this go?",68,2.0,(235,225,255),3)

        # Three circles — evenly spaced
        rad    = 110
        gap5   = 80
        total5 = 3*2*rad + 2*gap5
        x0     = (W-total5)//2 + rad
        y0     = H//2 + 20

        bin_circle(canvas, x0,              y0, rad,
                   (70,210,85),  "Compost", press_anim["c"])
        bin_circle(canvas, x0+2*rad+gap5,   y0, rad,
                   (200,185,50), "Recycle", press_anim["r"])
        bin_circle(canvas, x0+2*(2*rad+gap5),y0, rad,
                   (190,160,230),"Trash",   press_anim["t"])

        # Hint strip
        ctext(canvas,"Press the coloured button for your answer",
              H-35,0.85,(150,140,170),1)

        guess=None
        if flags["c"] or key==ord('c'): guess="COMPOST"
        elif flags["r"] or key==ord('r'): guess="RECYCLE"
        elif flags["t"] or key==ord('t'): guess="TRASH"

        if guess:
            clr(); last_act=now
            print(f"Guess:{guess}  AI:{f_cat}")
            if guess==f_cat:
                r_msg="Amazing! You got it!  🎉"
                r_sub=f"That's right — it goes in {BIN_LABEL[f_cat]}!"
                r_happy=True
            else:
                r_msg="Good try — almost!"
                r_sub=f"It actually goes in {BIN_LABEL[f_cat]}"
                r_happy=False
            led_show(f_cat); res_at=now; state=RESULT

    # ══════════════════════════════════════════════════
    #  RESULT
    # ══════════════════════════════════════════════════
    elif state==RESULT:
        last_act=now; elapsed_r=now-res_at
        col5=BIN_COL[f_cat]

        # Dark bg with faint colour tint
        tint=tuple(max(0,c//6) for c in col5)
        canvas[:]=tint

        # Top celebration strip
        rr(canvas,0,0,W,12,col5,r=0)
        rr(canvas,0,H-12,W,12,col5,r=0)

        # Face
        face_cx=W//2; face_cy=180; face_r=100
        face_col=(240,220,60) if r_happy else (180,180,220)
        draw_face(canvas,face_cx,face_cy,face_r,r_happy,face_col)

        # Result card
        cw6=1000; ch6=180; cx6=(W-cw6)//2; cy6=310
        bg6=tuple(max(0,c//4) for c in col5)
        rr(canvas,cx6,cy6,cw6,ch6,bg6,r=30)
        rr(canvas,cx6,cy6,cw6,ch6,col5,r=30,t=4)

        msg_col=(120,240,120) if r_happy else (100,160,255)
        ctext(canvas,r_msg,cy6+85, 2.0,msg_col,3)
        ctext(canvas,r_sub,cy6+145,1.0,  (220,215,230),2)

        # Bin badge
        badge=f"  {BIN_LABEL[f_cat].upper()}  "
        bs6=cv2.getTextSize(badge,F,1.4,2)[0]
        bx6=(W-bs6[0]-40)//2; by6=cy6+ch6+25
        rr(canvas,bx6,by6,bs6[0]+40,58,col5,r=22)
        ctext(canvas,badge,by6+44,1.4,(15,15,15),2)

        # Item name small
        chip=f"{f_label}  {f_conf:.0f}% confident"
        ctext(canvas,chip,by6+115,0.75,(150,145,165),1,alpha_bg=False)

        # Countdown bar
        pb_w2=700
        pbar(canvas,elapsed_r,RES_DUR,(W-pb_w2)//2,H-55,pb_w2,16,col5)
        ctext(canvas,"Returning to scan...",H-18,0.65,(130,125,145),1)

        if elapsed_r>=RES_DUR:
            led_clear(); clr(); state=IDLE

    # ── LED auto-off ──────────────────────────────────
    if led_off_at and now>led_off_at: led_clear(); led_off_at=0

    cv2.imshow("Smart Bin",canvas)
    cv2.setWindowProperty("Smart Bin",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    clr()

except KeyboardInterrupt:
    print("\nStopped")
finally:
    led_clear(); cam_off(); cv2.destroyAllWindows(); print("Done!")
