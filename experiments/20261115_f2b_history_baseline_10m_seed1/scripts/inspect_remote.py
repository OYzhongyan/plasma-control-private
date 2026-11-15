"""Read-only live evidence for the independent seed reproduction."""
from pathlib import Path
import csv
import datetime
import json
import os

ROOT=Path('/inspire/qb-ilm/project/ai-for-fusion/czxs25220173/plasma-control-seed1')
EXP=ROOT/'experiments/20261115_f2b_history_baseline_10m_seed1'
record={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,
        'listening_ports':[],'active_ports':[],'records':{},'logs':{},'processes':[],'hfm':{}}
for path in ('/proc/net/tcp','/proc/net/tcp6'):
    for line in Path(path).read_text().splitlines()[1:]:
        fields=line.split();port=int(fields[1].split(':')[1],16)
        if 2223<=port<=2254:
            if fields[3]=='0A':record['listening_ports'].append(port)
            if fields[3]=='01':record['active_ports'].append(port)
for name in ['launch.json','setup/launch_status.json','setup/environment_import.json','source_parity.json',
             'runtime_parity.json','config_parity.json','status.json','initial_state.json','comparison.json']:
    path=EXP/name
    if path.is_file():record['records'][name]=json.loads(path.read_text())
for name in ['train.log','launcher.log','setup/bootstrap.log','comparison.log']:
    path=EXP/name
    if path.is_file():
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-2500));record['logs'][name]=stream.read().decode(errors='replace')
progress=EXP/'training/logs/ppo/progress.csv'
if progress.is_file():
    with progress.open() as stream:rows=list(csv.DictReader(stream))
    record['progress']={'rows':len(rows),'first':rows[0] if rows else {},'last':rows[-1] if rows else {}}
for port in range(2223,2255):
    path=EXP/f'training/hfm/hfm_{port}.log'
    if path.is_file():
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-1500));tail=stream.read().decode(errors='replace')
        record['hfm'][str(port)]={'bytes':path.stat().st_size,'recent_step':'Received command: STEP' in tail,
                                  'tail':tail if port in (2223,2254) else None}
for path in Path('/proc').iterdir():
    if not path.name.isdigit():continue
    try:
        command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        if '-c ' not in command and str(ROOT) in command and any(name in command for name in ['capture_initial_train','launch_recorded','compare_training']):
            fields=(path/'stat').read_text().split()
            record['processes'].append({'pid':int(path.name),'state':fields[2], 'cpu_ticks':int(fields[13])+int(fields[14])})
    except OSError:pass
print(json.dumps(record))
