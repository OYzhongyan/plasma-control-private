from pathlib import Path
import csv,json,os,subprocess,sys,shutil,datetime
exp=Path(__file__).resolve().parent
root=exp.parents[1]
sys.path.insert(0,str(root))
from tools.run_baseline_experiment import summarize_score,write_json,make_plots
status=exp/'short_verification_status.json'
record={'phase':'RUNNING','pid':os.getpid(),'horizon':16,'episodes_per_split':1,'purpose':'pipeline smoke check only; not baseline performance','evaluation_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()}
def update(phase,**fields):
    record.update(phase=phase,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields);write_json(status,record)
try:
    original=exp/'status.json'
    if original.exists() and not (exp/'first_failure.json').exists(): shutil.copyfile(original,exp/'first_failure.json')
    command_record=json.loads((exp/'commands.json').read_text())
    metrics={}
    for split,specification in command_record['evaluation'].items():
        dest=exp/'verification'/split;dest.mkdir(parents=True,exist_ok=True)
        command=[arg for arg in specification['command'] if not arg.startswith(('eval.episodes_per_shot=','max_steps=','hydra.run.dir='))]
        command+=['eval.episodes_per_shot=1','max_steps=16','hydra.run.dir='+json.dumps(str(dest/'hydra'))]
        write_json(dest/'command.json',command)
        update('EVALUATING',split=split)
        with (dest/'evaluate.log').open('w') as log:
            result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(split+' short evaluation failed; see '+str(dest/'evaluate.log'))
        output=exp/'training/eval'
        metrics[split]=summarize_score(output/'score_summary.json',{**specification,'episodes':1})
        for p in output.iterdir(): shutil.move(str(p),str(dest/p.name))
        output.rmdir()
    write_json(exp/'short_verification_metrics.json',{'horizon':16,'episodes_per_split':1,'scores_not_baseline_performance':True,'evaluation':metrics})
    make_plots(exp)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for split in metrics:
        with (exp/'verification'/split/'step_reward_components.csv').open() as stream: rows=list(csv.DictReader(stream))
        fig,ax=plt.subplots(figsize=(7,4))
        ax.plot([float(r['step']) for r in rows],[float(r['ip']) for r in rows],label='actual Ip')
        ax.plot([float(r['step']) for r in rows],[float(r['reference_ip']) for r in rows],label='reference Ip')
        ax.set(xlabel='step (smoke horizon 16)',ylabel='Ip (A)',title=split+' pipeline smoke only');ax.legend();fig.tight_layout();fig.savefig(exp/'picture'/('smoke_'+split+'_ip.png'),dpi=140);plt.close(fig)
    with (exp/'README.md').open('a') as stream: stream.write('\n\n## Runtime preflight follow-up\nTraining completed 1024 steps. Initial evaluation failed while restoring Hydra save_dir; see first_failure.json and evaluation/seen/evaluate.log. After code correction, reused that model for 16-step, one-episode scoring probes on all three shots. See short_verification_metrics.json, verification/ and picture/. These short probes verify data flow only and are not full benchmark results.\n')
    update('COMPLETED',metrics=metrics,original_failure_preserved=True)
except BaseException as exc:
    update('FAILED',error=str(exc));raise
