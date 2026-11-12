"""Recorded recovery of an existing interrupted experiment; no HFM lifecycle changes."""
from pathlib import Path
import datetime,json,os,sys,subprocess,time,hashlib,zipfile,pickle,shutil,tempfile,importlib.util

SEG=Path(__file__).resolve().parent
EXP=SEG.parents[1]
ROOT=EXP.parents[1]
TRAIN=EXP/'training'/('resume_'+SEG.name)
sys.path.insert(0,str(ROOT))
from tools.run_baseline_experiment import summarize_score,save_metrics,environment_record
from train.train_ppo import model_architecture_record
from stable_baselines3 import PPO
import numpy as np

request=json.loads((SEG/'resume_request.json').read_text())
state={'pid':os.getpid(),'child_pid':None,'phase':'CHECKING','segment':SEG.name,'training_directory':str(TRAIN),'starting_timesteps':request['starting_timesteps'],'target_total_timesteps':request['target_total_timesteps']}
def write(path,data):
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def update(phase,**kw):
 state.update(phase=phase,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw)
 write(SEG/'status.json',state);write(EXP/'status.json',state)
def execute(command,log,phase):
 with log.open('w') as out:
  child=subprocess.Popen(command,cwd=ROOT,env=ENV,stdout=out,stderr=subprocess.STDOUT)
  update(phase,child_pid=child.pid)
  code=child.wait()
 if code:raise RuntimeError(phase+' child failed code='+str(code)+'; see '+str(log))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
ENV=os.environ.copy();ENV.update(NO_PROXY='*',PATH=str(Path(sys.executable).parent)+':'+ENV.get('PATH',''),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg',WANDB_MODE='offline',WANDB_DIR=str(EXP/'wandb'/('resume_'+SEG.name)))
for key in list(ENV):
 if key.lower() in ['http_proxy','https_proxy','all_proxy'] or key in ['WANDB_RUN_ID','WANDB_RESUME']:ENV.pop(key,None)
Path(ENV['WANDB_DIR']).mkdir(parents=True,exist_ok=True)
lock=SEG/'.continuation.lock'
try:
 with lock.open('x') as f:f.write(str(os.getpid()))
 update('CHECKING')
 model_path=Path(request['model_path']);norm_path=Path(request['vecnormalize_path'])
 model=PPO.load(str(model_path),device='cpu')
 assert model.num_timesteps==request['starting_timesteps']
 assert len(model.policy.optimizer.state)>0,'Optimizer state was not restored'
 critic='flux_cnn' if isinstance(model.observation_space, __import__('gymnasium').spaces.Dict) else 'mlp'
 arch=model_architecture_record(model.policy,critic)
 assert arch['actor_parameters']==275726
 assert critic==request['critic_type']
 with norm_path.open('rb') as f:norm=pickle.load(f)
 rms=vars(norm)['obs_rms'];rms=rms['actor_obs'] if isinstance(rms,dict) else rms
 assert rms.mean.shape==(812,) and np.all(np.isfinite(rms.mean)) and rms.count>0
 validation={'model':str(model_path),'vecnormalize':str(norm_path),'model_sha256':sha(model_path),'vecnormalize_sha256':sha(norm_path),'starting_timesteps':model.num_timesteps,'n_updates':model._n_updates,'optimizer_state_entries':len(model.policy.optimizer.state),'actor_normalization_count':float(rms.count),'architecture':arch}
 if critic=='flux_cnn':
  flux=vars(norm)['flux_rms'];assert np.isfinite(flux.mean) and flux.count>0
  validation['flux_normalization_count']=float(flux.count)
 write(SEG/'checkpoint_validation.json',validation)
 del model,norm
 # Runtime tests are temporary; no test model, score or log is retained.
 if importlib.util.find_spec('pytest'):
  with tempfile.TemporaryDirectory(prefix='plasma-resume-validation-') as tmp:
   r=subprocess.run([sys.executable,'-B','-m','pytest','test/test_resume_budget.py','-q','-p','no:cacheprovider','--basetemp',tmp],cwd=ROOT,env=ENV,capture_output=True,text=True)
   assert r.returncode==0,r.stdout[-1800:]+r.stderr[-600:]
  validation['runtime_budget_tests_passed']=True;write(SEG/'checkpoint_validation.json',validation)
 commands=json.loads((SEG/'commands.json').read_text())
 write(SEG/'environment.json',environment_record())
 deadline=time.monotonic()+600
 while True:
  listening=set();active=[]
  for name in ['/proc/net/tcp','/proc/net/tcp6']:
   for line in Path(name).read_text().splitlines()[1:]:
    f=line.split();port=int(f[1].split(':')[1],16)
    if f[3]=='0A':listening.add(port)
    if f[3]=='01' and 2223<=port<=2254:active.append(port)
  if set(range(2223,2255))<=listening and not active:break
  assert time.monotonic()<deadline,'HFM pool did not become idle/listening'
  update('WAITING_HFM');time.sleep(5)
 execute(commands['training'],TRAIN/'train.log','TRAINING')
 summary=json.loads((TRAIN/'training_summary.json').read_text());actual=summary['actual_timesteps']
 assert actual>=request['target_total_timesteps'],summary
 shutil.copy2(TRAIN/'.hydra/config.yaml',SEG/'resolved_config.yaml')
 metrics={'requested_timesteps':request['target_total_timesteps'],'actual_timesteps':actual,'actual_timesteps_evidence':str(TRAIN/'training_summary.json'),'training_reward':'upstream F2 dense reward','resume_segment':SEG.name,'checkpoint_starting_timesteps':request['starting_timesteps'],'evaluation':{}}
 for split,spec in commands['evaluation'].items():
  destination=EXP/'evaluation'/split
  assert not (destination/'score_summary.json').exists(),'Refusing to overwrite formal evaluation'
  destination.mkdir(parents=True,exist_ok=True)
  execute(spec['command'],TRAIN/('evaluate_'+split+'.log'),'EVALUATING_'+split.upper())
  output=TRAIN/'eval'
  metrics['evaluation'][split]=summarize_score(output/'score_summary.json',spec)
  for p in output.iterdir():
   assert not (destination/p.name).exists(),str(destination/p.name)
   shutil.move(str(p),str(destination/p.name))
  output.rmdir();save_metrics(EXP,metrics);write(SEG/'metrics.json',metrics)
 update('COMPLETED',child_pid=None,actual_timesteps=actual)
except BaseException as e:
 update('FAILED',error=type(e).__name__+': '+str(e));raise
finally:
 lock.unlink(missing_ok=True)
