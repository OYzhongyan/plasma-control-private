from pathlib import Path
import os,subprocess,json,datetime,shutil
exp=Path('/inspire/qb-ilm/project/ai-for-fusion/czxs25220173/plasma-control-seed1/experiments/20261115_f2b_history_baseline_10m_seed1');status=exp/'setup/system_tools.json'
state={'phase':'INSTALLING','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};status.write_text(json.dumps(state,indent=2))
env=os.environ.copy();env.update(DEBIAN_FRONTEND='noninteractive',NO_PROXY='*')
for key in list(env):
 if key.lower() in ['http_proxy','https_proxy','all_proxy']:env.pop(key,None)
with (exp/'setup/system_tools.log').open('w') as log:
 commands=[] if shutil.which('git') else [['apt-get','-o','Acquire::Retries=2','update'],['apt-get','install','-y','--no-install-recommends','git']]
 for command in commands:
  result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:
   state.update(phase='FAILED',command=command,returncode=result.returncode);status.write_text(json.dumps(state,indent=2));raise SystemExit(result.returncode)
state.update(phase='READY',git=subprocess.check_output(['git','--version'],text=True).strip());status.write_text(json.dumps(state,indent=2))
