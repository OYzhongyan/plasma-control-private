from pathlib import Path
import os,json,subprocess,time,datetime
exp=Path('/inspire/qb-ilm/project/ai-for-fusion/czxs25220173/plasma-control-cnn-critic/experiments/20261110_cnn_flux_critic_10m_seed0');out=exp/'training/resume_20261112_01/hfm';out.mkdir(parents=True,exist_ok=True)
record=exp/'resumes/20261112_01/hfm_launch.json'
runtime=Path('/inspire/qb-ilm/project/ai-for-fusion/czxs25220173/hfm-runtime-release-sealed-latest')
cpus=sorted(os.sched_getaffinity(0));assert len(cpus)>=32
per=len(cpus)//32
d={'hostname':os.uname().nodename,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'servers':[],'phase':'STARTING','cpu_partition':'same logical contiguous partition as default tasksetstart.sh','cache_isolation':'container-local /tmp','runtime':str(runtime)}
for i,port in enumerate(range(2223,2255)):
 cache=Path('/tmp')/('plasma-recovery-hfm-'+exp.name)/str(port);cache.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env.update(LISTEN_PORT=str(port),MATLAB_SOCKET_TIMEOUT='600',MATLAB_IDLE_TIMEOUT='600',MCR_CACHE_ROOT=str(cache),NO_PROXY='*')
 for k in list(env):
  if k.lower() in ['http_proxy','https_proxy','all_proxy']:env.pop(k,None)
 part=cpus[i*per:] if i==31 else cpus[i*per:(i+1)*per]
 command=['taskset','-c',','.join(str(c) for c in part),str(runtime/'run_hfm_server.sh')]
 with (out/('hfm_'+str(port)+'.log')).open('w') as log:
  p=subprocess.Popen(command,env=env,cwd=runtime,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 d['servers'].append({'port':port,'pid':p.pid,'cpus':part});record.write_text(json.dumps(d,indent=2))
 time.sleep(2)
d['phase']='STARTED';record.write_text(json.dumps(d,indent=2))
