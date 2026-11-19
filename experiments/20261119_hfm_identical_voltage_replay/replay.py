"""Replay identical socket bytes into a fresh HFM process. Run with a separate output directory per process."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time

import numpy as np


def run(runtime, out, cpus, warmup=False):
    defaults = dict(signeo=6e6, bp=0.19, q0=1.44)
    init = dict(defaults, L_addr="TOK_EXL50U/XL50U_fge/Data/FGE_INI/ini_L_v2.2.mat",
                LX_addr="TOK_EXL50U/XL50U_fge/Data/FGE_INI/ini_15892_500_v2.mat")
    scales = np.random.default_rng(15).uniform(0.95, 1.05, (4, 3))[-1]
    perturbed = {k: float(v*s) for (k,v), s in zip(defaults.items(), scales)}
    summary = dict(host=socket.gethostname(), affinity=cpus, cases={})
    def save():
        p = out/'status.tmp'
        p.write_text(json.dumps(summary, indent=2))
        p.replace(out/'status.json')
    save()
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix='hfm-replay-') as cache, (out/'hfm.log').open('w') as log:
        env = dict(os.environ, LISTEN_PORT=str(port), MCR_CACHE_ROOT=cache, MATLAB_SOCKET_TIMEOUT='120', MATLAB_IDLE_TIMEOUT='300')
        server = subprocess.Popen(['taskset','-c', cpus,str(runtime/'run_hfm_server.sh')], env=env, cwd=runtime,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic()+180
            while True:
                try:
                    conn = socket.create_connection(('127.0.0.1',port), timeout=2)
                    break
                except ConnectionRefusedError:
                    if server.poll() is not None or time.monotonic()>deadline:
                        raise
                    time.sleep(1)
            with conn, conn.makefile('rb') as reader:
                conn.settimeout(120)
                def request(payload):
                    conn.sendall(payload)
                    response = reader.readline()
                    if not response:
                        raise IOError('HFM disconnected')
                    return response
                init_bytes=('INIT\n'+json.dumps(init)+'\n').encode()
                if request(init_bytes).strip()!=b'INIT_OK':
                    raise RuntimeError('INIT rejected')
                cases = [('zero', defaults), ('varying', perturbed)]
                if warmup:
                    cases.insert(0, ('warmup', perturbed))
                for case, params in cases:
                    digest=hashlib.sha256(init_bytes)
                    with gzip.open(out/(case+'.jsonl.gz'),'wt',encoding='utf8') as trace:
                        for step in range(501):
                            if step==0:
                                payload=('RESET\n'+json.dumps(params)+'\n').encode()
                            else:
                                u=[0]*12 if case=='zero' else [((step*(j+7)+31*j)%161)-80 for j in range(12)]
                                payload=('STEP\n'+' '.join(map(str,u))+'\n').encode()
                            digest.update(payload)
                            raw=json.loads(request(payload))
                            trace.write(json.dumps(dict(step=step,input=payload.decode(),obs=raw),separators=(',',':'))+'\n')
                            failure=bool(raw['is_failure'])
                            if failure or step==500:
                                break
                    summary['cases'][case]=dict(steps=step,failure=failure,input_sha256=digest.hexdigest())
                    save()
            summary['completed']=True
        finally:
            try:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
            except ProcessLookupError:
                pass
            save()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('runtime',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('cpus')
    parser.add_argument('--warmup',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    run(args.runtime.resolve(),args.output,args.cpus,args.warmup)
