from pathlib import Path
import pickle
import numpy as np
def log(*args): pass


def safe_set_em(seg_a, seg_b, A, B, W, models):
    """Frozen-model set-EM with explicit handling of unsupported/empty frames.

    Keeps the last usable candidate set if a learned warp produces no candidates;
    stops with fewer than four anchors instead of attempting a degenerate TPS.
    """
    from setmatch import build_sets,decode_sets,competition_features
    from em import fit_warp
    mothers=[int(m) for m in A['labels'] if A['lay'].get(m,99)<=2]
    rows,X,_,_,growth=build_sets(0,seg_a,seg_b,A,B,W)
    previous=None;history=[];diagnostic=[]
    if not rows:
        return dict(rows=[],X=np.empty((0,19),np.float32),acc={},W=W,sel=0,diagnostics=['no_initial_candidates'])
    for r in range(11):
        s=models['stage1'].predict_proba(X)[:,1]
        pred=decode_sets(rows,s,mothers)
        anchors={m:v['daughters'] for m,v in pred.items() if v['daughters'] and v['conf']>=.5}
        current={m:tuple(v['daughters']) for m,v in pred.items()}
        stable=float(np.mean([current[m]==previous[m] for m in current])) if previous else 0.
        history.append(len(anchors))
        log('set-EM round',r,'anchors',len(anchors),'stable',round(stable,4))
        if r==10 or stable>=.98 or (len(history)>=3 and max(history[-2:])<=max(history[:-2])):break
        if len(anchors)<4:
            diagnostic.append('fewer_than_four_anchors');break
        proposed=fit_warp(A,B,anchors,smoothing=100.)
        new_rows,new_X,_,_,_=build_sets(0,seg_a,seg_b,A,B,proposed,anchors=anchors,growth=growth)
        if not new_rows:
            diagnostic.append('empty_proposed_warp_reverted');break
        previous=current;W=proposed;rows,X=new_rows,new_X
    s2=models['stage2'].predict_proba(competition_features(rows,X,s,A,B))[:,1]
    pred=decode_sets(rows,s2,mothers)
    anchors={m:v['daughters'] for m,v in pred.items() if v['daughters'] and v['conf']>=.5}
    final_rows,final_X,_,_,_=build_sets(0,seg_a,seg_b,A,B,W,anchors=anchors,growth=growth)
    if final_rows:rows,X=final_rows,final_X
    else:diagnostic.append('empty_final_rebuild_reverted')
    final=models['final'];lr=final['lambdarank'];p1=lr.p1(rows,X)
    scores=.5*(lr.p2(rows,X,p1,A,B)+final['classifier'].predict_proba(X)[:,1])
    return dict(rows=rows,X=X.astype(np.float32),acc=dict(zip(rows,scores)),W=W,sel=len(anchors),diagnostics=diagnostic)
