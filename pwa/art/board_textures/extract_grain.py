import cv2, numpy as np, sys
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
src, tag = sys.argv[1], sys.argv[2]
img = cv2.imread(src)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
t,_ = cv2.threshold(hsv[...,1],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
mask = ((hsv[...,0] > 8) & (hsv[...,0] < 40) & (hsv[...,1] > t) & (hsv[...,2] > 80)).astype(np.uint8)*255
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25,25),np.uint8))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((25,25),np.uint8))
cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = max(cnts, key=cv2.contourArea)
quad = cv2.approxPolyDP(c, 0.02*cv2.arcLength(c,True), True).reshape(-1,2)
if len(quad) != 4:
    quad = cv2.boxPoints(cv2.minAreaRect(c)).astype(int)
s = quad.sum(1); d = np.diff(quad,axis=1).ravel()
tl,br,tr,bl = quad[s.argmin()],quad[s.argmax()],quad[d.argmin()],quad[d.argmax()]
w = int(max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl))); h = int(max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr)))
M = cv2.getPerspectiveTransform(np.float32([tl,tr,br,bl]), np.float32([[0,0],[w,0],[w,h],[0,h]]))
rect = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_CUBIC).astype(np.float64)
gray = rect.mean(2)
bg = cv2.GaussianBlur(gray,(0,0),6); dark = np.clip(bg-gray,0,None)
col = dark.mean(0); row = dark.mean(1)
pitch = min(w,h)/20
pc,_ = find_peaks(col, prominence=col.max()*0.25, distance=int(pitch*0.6))
pr,_ = find_peaks(row, prominence=row.max()*0.25, distance=int(pitch*0.6))
print(f'{tag}: board {w}x{h}px, lines {len(pc)}x{len(pr)}')
sp = np.median(np.diff(pc))
R = max(2, int(round(sp*0.09))); S = max(4, int(round(sp*0.2)))
out = rect.copy()
def vfill(x, y0, y1):
    y0=max(0,y0); y1=min(h-1,y1)
    for yy in range(y0+1,y1):
        t=(yy-y0)/(y1-y0); out[yy,x]=(1-t)*out[y0,x]+t*out[y1,x]
for y in pr:
    lo,hi=max(0,y-S),min(h,y+S+1)
    band=uniform_filter1d(dark[lo:hi],9,axis=1); ys=lo+band.argmax(0)
    ys=np.round(uniform_filter1d(ys.astype(float),15)).astype(int)
    for x in range(w): vfill(x, ys[x]-R-1, ys[x]+R+1)
g2=out.mean(2); d2=np.clip(cv2.GaussianBlur(g2,(0,0),6)-g2,0,None)
for x in pc:
    lo,hi=max(0,x-S),min(w,x+S+1)
    band=uniform_filter1d(d2[:,lo:hi],9,axis=0); xs=lo+band.argmax(1)
    xs=np.round(uniform_filter1d(xs.astype(float),15)).astype(int)
    for y in range(h):
        x0,x1=xs[y]-R-1,xs[y]+R+1
        if x0<0 or x1>=w: continue
        for xx in range(x0+1,x1):
            t=(xx-x0)/(x1-x0); out[y,xx]=(1-t)*out[y,max(0,x0-(xx-x0))]+t*out[y,min(w-1,x1+(x1-xx))]
if len(pc)==19 and len(pr)==19:
    r=int(round(sp*0.28))
    for i in (3,9,15):
        for j in (3,9,15):
            cx,cy=int(pc[i]),int(pr[j])
            for xx in range(cx-r,cx+r+1):
                dy=int(np.sqrt(r*r-(xx-cx)**2)); vfill(xx, cy-dy-1, cy+dy+1)
for it in range(2):
    g=out.mean(2); d=cv2.GaussianBlur(g,(0,0),6)-g
    resid=cv2.dilate((d>12).astype(np.uint8),np.ones((3,3),np.uint8))
    for x in range(w):
        m=resid[:,x]; y=0
        while y<h:
            if m[y]:
                y0=y
                while y<h and m[y]: y+=1
                vfill(x, y0-1, y)
            else: y+=1
big=cv2.GaussianBlur(out,(0,0),max(40,int(sp*2)))
flat=np.clip(out/big*out.reshape(-1,3).mean(0),0,255)
# crop to the grid extent plus half a spacing (inside the bevelled edge), then square
x0,x1=int(pc[0]-sp/2),int(pc[-1]+sp/2); y0,y1=int(pr[0]-sp/2),int(pr[-1]+sp/2)
sq=flat[max(0,y0):y1, max(0,x0):x1]; side=min(sq.shape[:2]); sq=sq[:side,:side]
cv2.imwrite(f'{tag}_native.png', sq.astype(np.uint8))
up=cv2.resize(sq,(1024,1024),interpolation=cv2.INTER_LANCZOS4 if side<1024 else cv2.INTER_AREA)
cv2.imwrite(f'{tag}_1024.png', np.clip(up,0,255).astype(np.uint8))
print(f'{tag}: native square {side}px')
