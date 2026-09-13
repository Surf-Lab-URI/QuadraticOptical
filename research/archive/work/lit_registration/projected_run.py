from experiments import *
f=np.load(WORK/'final_ptv13.npz');pts=f['points'];p0=f['params'];A,B,va,vb=inputs();t=time.time();samplers=(Sampler(A,True),Sampler(B,True))
def task(i):return solve(A,B,va,vb,pts[i],p0[i],r=13,cubic=True,project_photometric=True,samplers=samplers)
with ThreadPoolExecutor(max_workers=4) as ex:results=list(ex.map(task,range(len(pts))))
p=np.array([v[0] for v in results]);stats={k:np.array([v[1].get(k,np.nan) for v in results]) for k in set().union(*(v[1] for v in results))}
path=HERE/'forward_cubic_projected.npz';np.savez(path,points=pts,params=p,radius=13,**stats)
metrics=evaluate(path);metrics['diagnostics']=dict(seconds=time.time()-t,failed_fallbacks=int(stats['failed'].sum()),median_ncc=float(np.median(stats['ncc'])))
(HERE/'projected_metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
