"""Prepare the seed-1 deployment using the already validated seed-0 harness."""
from pathlib import Path
import ast
import base64
import json
import zlib

EID = '20261115_f2b_history_baseline_10m_seed1'
REMOTE_ROOT = '/inspire/qb-ilm/project/ai-for-fusion/czxs25220173/plasma-control-seed1'


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Unexpected harness template: '+old)
    return source.replace(old, new)


def main():
    root = Path(__file__).resolve().parents[2]
    (root/'results/baseline_restore').mkdir(parents=True, exist_ok=True)
    exp = root/'experiments'/EID
    exp.mkdir(parents=True, exist_ok=True)
    templates = root/'scripts/seed_repro'
    bootstrap = (templates/'bootstrap_remote.py').read_text(encoding='utf-8')
    replacements = [
        ('plasma-control-seed-repro', 'plasma-control-seed1'),
        ('20261115_f2b_history_baseline_10m_seed0_repro', EID),
        ('competition_repro_20261115', 'competition_seed1_20261115'),
        ("startswith('plasma-control3--')", "startswith('plasma-control4--')"),
        ('plasma-seed-repro-20261115', 'plasma-seed1-20261115'),
        ("'2223', '--seed', '0',", "'2223', '--seed', '1',"),
        ("assert candidate == reference, 'Training hyperparameters differ from original baseline'",
         "assert reference['ppo']['seed'] == 0\n    reference['ppo']['seed'] = 1\n    assert candidate == reference, 'A non-seed training setting differs from baseline'"),
        ("'all_training_settings_identical': True, 'seed': 0, 'num_envs': 32,",
         "'all_training_settings_except_seed_identical': True, 'reference_seed': 0, 'seed': 1, 'num_envs': 32, 'only_training_setting_changed': 'ppo.seed',"),
    ]
    for old, new in replacements:
        bootstrap = replace_once(bootstrap, old, new)
    bootstrap = bootstrap.replace('baseline replay on plasma-control3', 'seed-1 baseline run on plasma-control4')
    bootstrap = replace_once(bootstrap, "manifest = json.loads((EXP/'reference_manifest.json').read_text())",
        "deadline = time.monotonic()+600\n    while True:\n        tools = EXP/'setup/system_tools.json'\n        if tools.is_file():\n            tool_state = json.loads(tools.read_text())\n            assert tool_state['phase'] != 'FAILED', tool_state\n            if tool_state['phase'] == 'READY': break\n        assert time.monotonic() < deadline, 'Git setup did not finish'\n        time.sleep(5)\n    manifest = json.loads((EXP/'reference_manifest.json').read_text())")
    launch = replace_once((templates/'launch_recorded.py').read_text(encoding='utf-8'),
                          "'2223', '--seed', '0',", "'2223', '--seed', '1',")
    inspect = (templates/'inspect_remote.py').read_text(encoding='utf-8').replace(
        'plasma-control-seed-repro', 'plasma-control-seed1').replace(
        '20261115_f2b_history_baseline_10m_seed0_repro', EID)
    sources = {'bootstrap_remote.py': bootstrap, 'launch_recorded.py': launch,
               'inspect_remote.py': inspect,
               'capture_initial_train.py': (templates/'capture_initial_train.py').read_text(encoding='utf-8'),
               'compare_training.py': (templates/'compare_training.py').read_text(encoding='utf-8')}
    payload = {}
    for name, content in sources.items():
        ast.parse(content)
        file = exp/'scripts'/name
        file.parent.mkdir(exist_ok=True)
        file.write_text(content, encoding='utf-8')
        payload['scripts/'+name] = file.read_bytes()
    reference = root/'experiments/20261109_f2b_history_baseline_10m_seed0/remote_snapshot'
    for dest, original in [('reference_manifest.json','manifest.json'),('reference_config.json','requested_config.json'),
                           ('reference_environment.json','environment.json'),('reference_runtime.json','runtime_provenance.json'),
                           ('reference_resolved_config.yaml','training/.hydra/config.yaml')]:
        content = (reference/original).read_bytes()
        (exp/dest).write_bytes(content)
        payload[dest] = content
    packed = base64.b64encode(zlib.compress(json.dumps({n:base64.b64encode(b).decode() for n,b in payload.items()}).encode())).decode()
    source = '''from pathlib import Path
import os,json,base64,zlib,subprocess
assert os.uname().nodename.startswith('plasma-control4--')
exp=Path(EXPERIMENT)
data=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
for name,content in data.items():
 p=exp/name;assert p.resolve().is_relative_to(exp.resolve());p.parent.mkdir(parents=True,exist_ok=True)
 assert not p.exists(),name
 p.write_bytes(base64.b64decode(content))
with (exp/'setup/bootstrap.log').open('w') as log:
 p=subprocess.Popen(['/root/miniforge3/bin/python','-B',str(exp/'scripts/bootstrap_remote.py')],cwd=exp.parent.parent,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'bootstrap_pid':p.pid,'experiment':str(exp),'files_staged':len(data)}))
'''.replace('EXPERIMENT',repr(REMOTE_ROOT+'/experiments/'+EID)).replace('PAYLOAD',repr(packed))
    output = root/'results/baseline_restore/stage_seed1_remote.py'
    output.write_text(source, encoding='utf-8')
    exporter = '''from pathlib import Path
import os,json,subprocess
assert os.uname().nodename.startswith('plasma-control--')
exp=Path(EXPERIMENT);archive=exp/'setup/competition_env.tar'
assert exp.is_dir() and not archive.exists()
code="""from pathlib import Path
import subprocess,json,datetime,os
exp=Path(EXPERIMENT);archive=exp/'setup/competition_env.tar'
state={'phase':'EXPORTING','hostname':os.uname().nodename,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive':str(archive)}
status=exp/'setup/environment_export.json';status.write_text(json.dumps(state,indent=2))
r=subprocess.run(['tar','--exclude=__pycache__','--exclude=*.pyc','-cf',str(archive),'-C','/root/miniforge3/envs/competition_env','.'],capture_output=True,text=True)
state.update(phase='EXPORTED' if r.returncode==0 else 'FAILED',returncode=r.returncode,archive_bytes=archive.stat().st_size,stderr=r.stderr[-2000:],finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
status.write_text(json.dumps(state,indent=2))
"""
driver=exp/'setup/export_environment.py';driver.write_text(code)
with (exp/'setup/environment_export.log').open('w') as log:
 child=subprocess.Popen(['/root/miniforge3/bin/python','-B',str(driver)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'export_pid':child.pid,'archive':str(archive)}))
'''.replace('EXPERIMENT',repr(REMOTE_ROOT+'/experiments/'+EID))
    ast.parse(exporter)
    (root/'results/baseline_restore/export_seed1_environment.py').write_text(exporter, encoding='utf-8')
    print(json.dumps({'experiment_id':EID,'training_seed':1,'prepared_assets':len(payload)}))


if __name__ == '__main__':
    main()
