from pathlib import Path
import pickle
import numpy as np
from scipy.spatial import cKDTree
import fcntl
import tifffile
from scipy.interpolate import RBFInterpolator
import wtrack as W, wgen, prog, baseline, vertices, wallpts, seedverify, vgrow
from vfuse import vfuse
MODELS=Path(__file__).resolve().parents[1]/"models"
DATA=OUTPUT=None


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp");tmp.write_bytes(pickle.dumps(obj,protocol=4));tmp.replace(path)


def trained(seq,t,fam):
    import bm,em,generalize
    from setmatch import decode_sets
    def bounded(y,d,**kw):
        if len(y)>1500:
            take=np.linspace(0,len(y)-1,1500).astype(int);y,d=y[take],d[take]
        return RBFInterpolator(y,d,**kw)
    bm.RBFInterpolator=em.RBFInterpolator=bounded
    generalize.log=lambda *args:None
    A,B=[pickle.loads((DATA/seq/f"T{k}/props.pkl").read_bytes()) for k in (t,t+1)]
    anchors={m:ds for m,(ds,q) in fam.items() if q>=0.5}
    if len(anchors)<4:return {},'insufficient anchors'
    X=A["cen"][list(anchors)]
    if np.linalg.matrix_rank(X-X.mean(0))<3:return {},'coplanar anchors'
    warp=em.fit_warp(A,B,anchors,smoothing=100)
    models=pickle.loads((MODELS/"set_ranker.pkl").read_bytes())
    models["final"]=pickle.loads((MODELS/"final_ranker.pkl").read_bytes())
    segs=[tifffile.imread(DATA/seq/f"T{k}/seg.tif") for k in (t,t+1)]
    saved=generalize.safe_set_em(*segs,A,B,warp,models)
    if not saved['acc']:return {},'no candidates'
    mothers=[int(m) for m in A["labels"] if A["lay"].get(m,99)<=2]
    dec=decode_sets(list(saved['acc']),np.array(list(saved['acc'].values())),mothers)
    return {m:list(v['daughters']) for m,v in dec.items()},'ran'


def junctions(seq,t,A,B,fam):
    ia=dict(zip(A['labs'],range(len(A['labs']))));ib=dict(zip(B['labs'],range(len(B['labs']))))
    seeds={m:ds for m,(ds,q) in fam.items() if q>=0.7 and A['lay'][ia[m]]==1}
    if len(seeds)<4 or min((A['lay']==1).sum(),(B['lay']==1).sum())<4:return {},'insufficient surface anchors'
    for k in (t,t+1):
        with (wgen.OUT/f"{seq}_{k}.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            V=vertices.vertices(seq,k)
            wallpts.wall_points(seq,k)
            seedverify.snapped(seq,k,6.0)
            if not V['of']:return {},'no surface junctions'
    ms=sorted(seeds);xa=np.array([A['cen'][ia[m]] for m in ms]);xb=np.array([vgrow.clone_cen(B,ib,seeds[m]) for m in ms])
    if np.linalg.matrix_rank(xa-xa.mean(0))<2:return {},'degenerate surface anchors'
    result,_=vgrow.run(seq,t,xa,xb,iters=400,workers=1,log=lambda *x:None,Gt=None,seed_fam=seeds)
    return result['clones'],'ran'


def run(job):
    seq,t=job;out=OUTPUT/seq/f"T{t}"; out.mkdir(parents=True,exist_ok=True)
    A,B=wgen.load(seq,t),wgen.load(seq,t+1)
    regfile=out/"registration.pkl"
    if not regfile.exists():
        T,info=wgen.register(A,B)
        for sigma in (8.,6.,5.):
            tab,*_=W.overlap(seq,t,t+1,T)
            if not np.any(tab[:,0]>0):break
            T,_=W.refine_from_overlap(T,tab,A,B,sigma*info['unit'])
        u=np.cbrt(np.median(A['vol']))
        schedule=[(None,.35)]*4+[(s,.45) for s in (4.,3.)]*2+[(s,.5) for s in (2.2,1.6,1.2)]*3+[(1.,.55)]*10
        best=-np.inf
        for sg,thr in schedule+[(1.,.55)]:
            tab,vol,_,fam,xb,xa,score=prog.step(seq,t,T,A,B,thr)
            if score>best:
                best=score;bestfam=fam; ov=baseline.predict(dict(tab=tab,volB=vol),.4,.05)
            if len(xa)<15 or np.linalg.matrix_rank(xa-xa.mean(0))<3:break
            T=prog.Affine(xb,xa) if sg is None else prog.MLS(xb,xa,sg*u)
        save(regfile,(bestfam,ov,best))
    fam,ov,best=pickle.loads(regfile.read_bytes())
    save(out/"overlap.pkl",ov)
    if not (out/"matcher.pkl").exists():save(out/"matcher.pkl",trained(seq,t,fam))
    if not (out/"junction.pkl").exists():save(out/"junction.pkl",junctions(seq,t,A,B,fam))
    mt,_=pickle.loads((out/"matcher.pkl").read_bytes());vg,_=pickle.loads((out/"junction.pkl").read_bytes())
    result=vfuse(vg,fam,ov,mt,.7,cthr=.9)
    assert sum(map(len,result.values()))==len(set(d for ds in result.values() for d in ds))
    save(out/"full.pkl",result)
    return seq,t
