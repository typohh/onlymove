"""Population calibration of the 3PL success model vs (z = slope*(R-x)) and the
user's ABSOLUTE rating. Working entirely in raw-rating space (no M needed):
    P = c + (1-c)*sigmoid(a*(R-x) + kappa(c)),  kappa makes P=0.8 at R=x.
Per-(user, calendar-month) rating R is fit by 3PL MLE so 8-month drift doesn't
collapse into one number. Then per attempt: residual r = y - P, plus z and R.
"""
import sqlite3, numpy as np, pandas as pd, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CALIB_P = 0.8
LOGIT_80 = np.log(CALIB_P/(1-CALIB_P))
FLOOR_MAX = 0.78

def kappa(c):
    c = np.clip(c, 0, FLOOR_MAX)
    q = np.clip((CALIB_P - c)/(1 - c), 1e-6, 1-1e-6)
    return np.log(q/(1-q))

def sig(x): return 1/(1+np.exp(-np.clip(x,-40,40)))

# ratings dir is overridable (--ratings-dir) so a candidate retrain in a temp dir can be
# A/B'd against production ratings_v2/ without copying files around.
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--ratings-dir', default=str(ROOT/'ratings_v2'))
_ap.add_argument('--tag', default='')   # appended to output png names for A/B side-by-side
_args, _ = _ap.parse_known_args()
RATINGS = Path(_args.ratings_dir)
TAG = ('_'+_args.tag) if _args.tag else ''
print(f"ratings dir: {RATINGS}")

# ---- per-problem maps (per size) -------------------------------------------
maps = {}
for size in (9,13):
    d = RATINGS/f'{size}'
    r = pd.read_csv(d/'tsumego_ratings_final.csv').set_index('tsumego_id')['rating']
    a = pd.read_csv(d/'tsumego_quality_final.csv').set_index('tsumego_id')['quality']
    c = pd.read_csv(d/'tsumego_floor_final.csv').set_index('tsumego_id')['floor']
    m = pd.DataFrame({'x':r,'a':a,'c':c}).dropna()
    m = m[(m['a']>0)]
    maps[size] = m

# ---- attempts --------------------------------------------------------------
con = sqlite3.connect(ROOT/'attempts/database.sqlite')
df = pd.read_sql("SELECT user_id,problem,success,created_at,size FROM tsumego_reports WHERE is_repeat=0", con)
con.close()
df['size'] = df['size'].astype(int)
df['y'] = df['success'].astype(float)
df['ym'] = df['created_at'].str.slice(0,7)

# join problem params per size
parts=[]
for size,m in maps.items():
    s = df[df['size']==size].join(m, on='problem', how='inner')
    parts.append(s)
df = pd.concat(parts, ignore_index=True)
df['kap'] = kappa(df['c'].values)
print(f"usable attempts: {len(df)}  users: {df.user_id.nunique()}")

# ---- per (user, month) 3PL MLE for R ---------------------------------------
def fit_R(x,a,c,kap,y, R0=None):
    R = float(np.average(x)) if R0 is None else R0
    for _ in range(40):
        eta = a*(R-x)+kap
        praw = sig(eta); P = c+(1-c)*praw
        P = np.clip(P,1e-9,1-1e-9)
        g = (1-c)*a*praw*(1-praw)          # dP/dR
        score = np.sum((y-P)*g/(P*(1-P)))
        info  = np.sum(g*g/(P*(1-P)))
        if info < 1e-9: break
        step = np.clip(score/info, -2, 2)
        R += step
        R = np.clip(R, -6, 10)
        if abs(step) < 1e-7: break
    return R

MIN_N = 15
rows = []      # per-attempt records with R attached
dR = []        # per user-month: (R_all, R_near, absR, n)
for (uid,ym), g in df.groupby(['user_id','ym']):
    if len(g) < MIN_N: continue
    ym_mean = g['y'].mean()
    if ym_mean<=0 or ym_mean>=1: continue   # all-correct/all-wrong: R unidentified
    x=g['x'].values; a=g['a'].values; c=g['c'].values; kap=g['kap'].values; y=g['y'].values
    R = fit_R(x,a,c,kap,y)
    z = a*(R-x)
    P = c+(1-c)*sig(a*(R-x)+kap)
    sub = g[['size']].copy()
    sub['z']=z; sub['P']=P; sub['y']=y; sub['R']=R
    rows.append(sub)
    # seeded near-rank refit (|z|<=L), L from pHi=0.95 mirrored in logit
    L = np.log(0.95/0.05) - LOGIT_80
    near = np.abs(z)<=L
    if near.sum()>=10 and 0<y[near].mean()<1:
        Rn = fit_R(x[near],a[near],c[near],kap[near],y[near], R0=R)
        dR.append((R, Rn, R, near.sum(), len(g)))

att = pd.concat(rows, ignore_index=True)
print(f"user-months fit: {att.R.notna().groupby(att.R).ngroups if False else len(rows)}  attempts in calib: {len(att)}")

# ---- marginal calibration curve in z ---------------------------------------
def binstats(d, col, edges):
    out=[]
    idx = np.digitize(d[col].values, edges)-1
    for b in range(len(edges)-1):
        sel = idx==b
        n = sel.sum()
        if n<30: continue
        yb=d['y'].values[sel]; Pb=d['P'].values[sel]
        obs=yb.mean(); pred=Pb.mean()
        se=np.sqrt(obs*(1-obs)/n)
        out.append(((edges[b]+edges[b+1])/2, obs, pred, obs-pred, se, n))
    return pd.DataFrame(out, columns=[col,'obs','pred','rho','se','n'])

zedges = np.arange(-5,5.01,0.5)
marg = binstats(att, 'z', zedges)
print("\n=== MARGINAL calibration vs z (obs - pred) ===")
print(marg.to_string(index=False, float_format=lambda v:f'{v:7.3f}'))

# ---- conditioned on absolute rating ----------------------------------------
# rank label via M for readability
M = json.load(open(ROOT/'pwa/tsumegos/rating_to_rank.json'))['9']
def to_rank(r): return float(np.interp(r, M['ratings'], M['rank_indices']))
qs = att['R'].quantile([0,.25,.5,.75,1.0]).values
att['Rbin'] = pd.cut(att['R'], bins=np.unique(qs), include_lowest=True)
print("\n=== calibration vs z, STRATIFIED by absolute rating R ===")
strata=[]
for rb, d in att.groupby('Rbin', observed=True):
    rmid = d['R'].mean()
    bs = binstats(d, 'z', zedges)
    hard = bs[bs['z']<-1]; easy = bs[bs['z']>1]; mid = bs[bs['z'].abs()<=1]
    wm = lambda t: np.average(t['rho'], weights=t['n']) if len(t) else np.nan
    print(f"R~{rmid:6.2f} (~rank {to_rank(rmid):4.1f}, n={len(d):6d})  "
          f"rho_hard(z<-1)={wm(hard):+.3f}  rho_mid={wm(mid):+.3f}  rho_easy(z>1)={wm(easy):+.3f}")
    strata.append((rmid, to_rank(rmid), bs))

# ---- practical: does near-rank refit move R, vs absolute rating? ------------
dR = pd.DataFrame(dR, columns=['R_all','R_near','absR','n_near','n_all'])
dR['delta'] = dR['R_near']-dR['R_all']
dR['rank_all'] = dR['R_all'].map(to_rank)
dR['rank_near']= dR['R_near'].map(to_rank)
dR['drank'] = dR['rank_near']-dR['rank_all']
print(f"\n=== near-rank refit shift (user-months n={len(dR)}) ===")
print(f"  median |delta rank| = {dR['drank'].abs().median():.3f}   "
      f"90th pct = {dR['drank'].abs().quantile(.9):.3f}")
dR['absRbin']=pd.cut(dR['R_all'], bins=np.unique(dR['R_all'].quantile([0,.5,.8,.95,1]).values), include_lowest=True)
for rb,d in dR.groupby('absRbin', observed=True):
    print(f"  R_all~{d['R_all'].mean():6.2f} (rank {to_rank(d['R_all'].mean()):4.1f}, n={len(d):5d})  "
          f"median drank={d['drank'].median():+.3f}  mean drank={d['drank'].mean():+.3f}  "
          f"90pct|drank|={d['drank'].abs().quantile(.9):.3f}")

# ---- plots -----------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(18,5.2))

# A: marginal obs vs pred
a0=ax[0]
a0.plot(marg['z'], marg['pred'], 'o-', color='#2563eb', label='model P (predicted)')
a0.plot(marg['z'], marg['obs'],  's-', color='#dc2626', label='observed success')
a0.axvline(0, color='gray', lw=.7, ls=':'); a0.axhline(0.8, color='gray', lw=.7, ls=':')
a0.set_xlabel('z = slope·(R − x)   [hard ← 0 → easy]'); a0.set_ylabel('P(clear)')
a0.set_title('A. Marginal calibration\n(model over-predicts the hard side → floor too high)')
a0.legend(); a0.grid(alpha=.2)

# B: rho(z) per absolute-rating stratum  (the money plot)
a1=ax[1]
cols=['#1a9850','#91cf60','#fdae61','#d73027']
for (rmid,rk,bs),col in zip(strata,cols):
    a1.plot(bs['z'], bs['rho'], 'o-', color=col, label=f'R≈{rmid:.1f} (~rank {rk:.0f})')
a1.axhline(0, color='k', lw=.8); a1.axvline(0, color='gray', lw=.7, ls=':')
a1.set_xlabel('z = slope·(R − x)'); a1.set_ylabel('ρ = observed − predicted')
a1.set_title('B. Calibration shifts UP with absolute rating\n(weak under-perform, strong over-perform, esp. mid)')
a1.legend(fontsize=9); a1.grid(alpha=.2)

# C: near-rank refit shift vs absolute rating
a2=ax[2]
g = dR.groupby('absRbin', observed=True)
xs = g['R_all'].mean().map(to_rank).values
md = g['drank'].median().values
a2.bar(range(len(xs)), md, color=['#d73027' if v<0 else '#1a9850' for v in md])
a2.set_xticks(range(len(xs))); a2.set_xticklabels([f'rank\n{x:.0f}' for x in xs])
a2.axhline(0, color='k', lw=.8)
a2.set_ylabel('median Δrank (near-rank − all-data)')
a2.set_title('C. Outlier removal: rating-dependent\n(weak pulled DOWN, strong pushed UP)')
a2.grid(alpha=.2, axis='y')

plt.tight_layout()
fig.savefig(f'/tmp/calibration{TAG}.png', dpi=110)
print(f"\nsaved /tmp/calibration{TAG}.png")

# ---- implied better floor at the hard tail ---------------------------------
deep = att[att['z']<-2.5]
print(f"\nHard tail (z<-2.5, n={len(deep)}): observed clear = {deep['y'].mean():.3f}, "
      f"model predicts = {deep['P'].mean():.3f}  → floor over-stated by ~{deep['P'].mean()-deep['y'].mean():.2f}")
