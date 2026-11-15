from pathlib import Path
import subprocess,json,datetime,os
exp=Path('/inspire/qb-ilm/project/ai-for-fusion/czxs25220173/plasma-control-seed1/experiments/20261115_f2b_history_baseline_10m_seed1');archive=exp/'setup/competition_env.tar'
state={'phase':'EXPORTING','hostname':os.uname().nodename,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive':str(archive)}
status=exp/'setup/environment_export.json';status.write_text(json.dumps(state,indent=2))
r=subprocess.run(['tar','--exclude=__pycache__','--exclude=*.pyc','-cf',str(archive),'-C','/root/miniforge3/envs/competition_env','.'],capture_output=True,text=True)
state.update(phase='EXPORTED' if r.returncode==0 else 'FAILED',returncode=r.returncode,archive_bytes=archive.stat().st_size,stderr=r.stderr[-2000:],finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
status.write_text(json.dumps(state,indent=2))
