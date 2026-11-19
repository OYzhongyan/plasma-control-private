"""Compare replay traces on their common input prefix and aligned solver output."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np


def compare(d, left, right, case):
    changed = {}
    first = None
    identical = 0
    steps = 0
    inp = hashlib.sha256()
    with gzip.open(d/left/(case+'.jsonl.gz'),'rt') as a, gzip.open(d/right/(case+'.jsonl.gz'),'rt') as b:
        for xs,ys in zip(a,b):
            x,y=json.loads(xs),json.loads(ys)
            assert x['step']==y['step'] and x['input']==y['input']
            inp.update(x['input'].encode())
            steps+=1
            xx,yy=x['obs'],y['obs']
            assert xx.keys()==yy.keys()
            dif={}
            for k in xx:
                if xx[k]==yy[k]:
                    continue
                p,q=np.asarray(xx[k],dtype=float),np.asarray(yy[k],dtype=float)
                if p.shape!=q.shape:
                    dif[k]=dict(shape=[p.shape,q.shape])
                    continue
                eq=(p==q)|(np.isnan(p)&np.isnan(q))
                if eq.all(): continue
                diff=np.abs(p[np.isfinite(p)&np.isfinite(q)]-q[np.isfinite(p)&np.isfinite(q)])
                dif[k]=dict(n=int((~eq).sum()),max_abs=float(diff.max()) if diff.size else None)
            if dif:
                if first is None: first=dict(step=x['step'],fields=dif)
                for k in dif: changed[k]=changed.get(k,0)+1
            else:
                identical+=1
    return dict(common_frames=steps,identical_frames=identical,first=first,differing_frame_counts=changed,
                common_input_digest=inp.hexdigest())


def solver_rows(path):
    cases=[]
    waiting=False
    for line in path.read_text(errors='replace').splitlines():
        if 'Received command: RESET' in line:
            cases.append([])
            waiting=False
        elif 'Received command: STEP' in line and cases:
            if waiting: raise ValueError('Missing solver row for STEP')
            waiting=True
        if waiting and re.match(r'^\s*\d+\.\d+\s*\|',line):
            cases[-1].append(line.strip())
            waiting=False
    return cases


def report(d):
    states={s:json.loads((d/s/'status.json').read_text()) for s in 'ABCDE'}
    assert all(s.get('completed') for s in states.values()), states
    out=dict(runs=states, comparisons={})
    for a,b in [('A','C'),('A','B'),('B','C'),('A','D'),('A','E')]:
        logs1,logs2=solver_rows(d/a/'hfm.log'),solver_rows(d/b/'hfm.log')
        for i,case in enumerate(['zero','varying']):
            result=compare(d,a,b,case)
            x,y=logs1[i],logs2[i + int(b=='D')]
            assert len(x)==states[a]['cases'][case]['steps'] and len(y)==states[b]['cases'][case]['steps']
            result['solver_row_counts']=[len(x),len(y)]
            result['first_solver_difference']=next((dict(step=j+1,left=m,right=n) for j,(m,n) in enumerate(zip(x,y)) if m!=n),None)
            out['comparisons'][a+b+'_'+case]=result
    return out


if __name__=='__main__':
    print(json.dumps(report(Path(sys.argv[1])),indent=2))
