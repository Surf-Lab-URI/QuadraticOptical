"""Sequential, logged production of the two fresh image-only analyses."""
from pathlib import Path
import subprocess,sys,time,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
events=[]
for pair in [80,100]:
    steps=[('tracking',['tracking.py','--pair',str(pair),'--workers','4']),
           ('fitting',['parallel_fit_fields.py','--pairs',str(pair),'--workers','4','--checkpoint-every','1024']),
           ('finalization',['finalize_fields.py','--pairs',str(pair)]),
           ('integration',['integrate_profile.py','--pair',str(pair)])]
    for name,args in steps:
        start=time.time();print('START',pair,name,flush=True)
        with (HERE/str(pair)/(name+'.log')).open('w') as log:
            proc=subprocess.Popen([sys.executable,'-u',str(HERE/args[0])]+args[1:],cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in proc.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            code=proc.wait()
        events.append(dict(pair=pair,stage=name,exit_code=code,seconds=time.time()-start))
        (HERE/'production_progress.json').write_text(json.dumps(events,indent=2))
        if code:raise SystemExit(code)
        print('FINISH',pair,name,round(time.time()-start,1),'seconds',flush=True)
print('Both image-only pairs complete',flush=True)
