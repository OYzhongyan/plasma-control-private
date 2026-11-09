from pathlib import Path
import json,os,time,datetime
root=Path('/inspire/qb-ilm/project/ai-for-fusion/czxs25220173/plasma-control-baseline-20261109')
status=root/'experiments/20261109_f2b_history_smoke_1024_seed0/short_verification_status.json'
record=root/'experiments/20261109_f2b_history_baseline_10m_seed0/autostart_status.json'
def state(phase,**fields):
    record.write_text(json.dumps(dict(phase=phase,pid=os.getpid(),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields),indent=2))
state('WAITING_FOR_SHORT_VERIFICATION')
try:
    deadline=time.monotonic()+1800
    while time.monotonic()<deadline:
        if status.exists():
            phase=json.loads(status.read_text()).get('phase')
            if phase=='COMPLETED': break
            if phase=='FAILED': raise RuntimeError('Short scoring verification failed; full run not started')
        time.sleep(2)
    else: raise TimeoutError('Short verification did not finish within 30 minutes')
    exec("from pathlib import Path\nimport os,subprocess,json,datetime\nroot=Path('/inspire/qb-ilm/project/ai-for-fusion/czxs25220173/plasma-control-baseline-20261109')\nshort=root/'experiments/20261109_f2b_history_smoke_1024_seed0/short_verification_status.json'\nif not short.exists() or json.loads(short.read_text()).get('phase')!='COMPLETED':\n    raise SystemExit('Short real-model scoring verification is not complete')\nexperiment='20261109_f2b_history_baseline_10m_seed0'\nexp=root/'experiments'/experiment\nif (exp/'status.json').exists(): raise SystemExit('Full experiment already has a status; refusing duplicate launch')\nlisten=set();active=[]\nfor name in ['/proc/net/tcp','/proc/net/tcp6']:\n    for line in Path(name).read_text().splitlines()[1:]:\n        f=line.split();port=int(f[1].split(':')[1],16)\n        if f[3]=='0A':listen.add(port)\n        if f[3]=='01' and 2223<=port<=2254: active.append(port)\nmissing=sorted(set(range(2223,2255))-listen)\nif missing or active: raise SystemExit('HFM pool must be listening and idle; missing='+str(missing)+' active='+str(active))\nenv=os.environ.copy();env.update(NO_PROXY='*',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',MPLBACKEND='Agg',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')\nfor key in list(env):\n    if key.lower() in ['all_proxy','http_proxy','https_proxy']: env.pop(key,None)\ncommand=['/root/miniforge3/envs/competition_env/bin/python','-B','tools/run_baseline_experiment.py','--experiment-id',experiment,'--total-timesteps','10000000','--num-envs','32','--base-port','2223','--wandb-mode','offline']\nwith (exp/'launcher.log').open('w') as stream: child=subprocess.Popen(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)\nrecord={'pid':child.pid,'command':command,'cwd':str(root),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'expected_requested_timesteps':10000000,'expected_actual_timesteps':10002432,'reuse_hfm_ports':[2223,2254],'no_hfm_restart':True,'transport':'jupyter','proxy':'none','preflight_status':str(short)}\n(exp/'launch.json').write_text(json.dumps(record,indent=2)+'\\n')\nprint(json.dumps(record))\n",{'__name__':'__main__'})
    state('LAUNCHED')
except BaseException as exc:
    state('FAILED',error=str(exc));raise
