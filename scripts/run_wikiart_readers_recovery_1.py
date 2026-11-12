"""Recovery: reuse completed fits/sanity images, score sanity, then finish frozen edits."""
import argparse,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(1024**2):h.update(b)
    return h.hexdigest()

def main(output,repo):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Allocated job required')
    repo=Path(repo).resolve();output=Path(output).resolve()
    assert output.is_dir() and not (output/'run').exists()
    prep=json.loads((output/'preparation.json').read_text());data=Path(prep['data']);assets=Path(prep['assets'])
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==prep['code_commit'], 'Code commit changed'
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip(), 'Tracked worktree changes'
    assert prep['ready'] is True
    for path,sha in prep['required_sha256'].items():assert digest(path)==sha,path
    run=output/'run';run.mkdir();commands=[]
    provenance=dict(preparation=prep,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),seed=42,null_seed=123,commands=commands)
    def stage(module,*args):
        code="import runpy,sys,random; import numpy as np,torch; random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);m=sys.argv.pop(1);runpy.run_module(m,run_name='__main__')"
        cmd=[sys.executable,'-c',code,module,*map(str,args)];commands.append(cmd)
        (run/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        subprocess.run(cmd,cwd=repo,check=True)
    def config(arm,dest,edit_manifest=None):
        c=json.loads((repo/f'config/steering/wikiart_reader_{arm}.json').read_text())
        if edit_manifest:
            c['source_manifest']=str(edit_manifest)
            c['edit']={'layers':'up_blocks.1,up_blocks.2,up_blocks.3','start_call':1,'stop_call':50,'representations':['SA'],'strengths':[1.,2.],'controls':['true'],'renorm':False}
        path=dest/('edit_config.json' if edit_manifest else 'extract_config.json');path.write_text(json.dumps(c,indent=2)+'\n');return path
    upstream=Path(prep['upstream']).resolve()
    # Only immutable completed directories are linked; no stage writes into them.
    for arm in ('empty','different'):
        dest=run/arm;dest.mkdir()
        (dest/'fit').symlink_to(upstream/'run'/arm/'fit',target_is_directory=True)
    sanity=output/'sanity';sanity.mkdir();sanrun=sanity/'run';sanrun.mkdir()
    for arm in ('empty','different'):
        dest=sanrun/arm;dest.mkdir()
        (dest/'edit').symlink_to(upstream/'sanity/run'/arm/'edit',target_is_directory=True)
    stage('scripts.score_wikiart_edits','--run',sanrun,'--data',data,'--assets',assets,'--expected',2)
    stage('scripts.summarize_wikiart','--run',sanrun,'--expected',2)
    for arm in ('empty','different'):
        dest=run/arm;cfg=config(arm,dest,data/'edit.jsonl')
        stage('diff_inversion.steering','edit','--config',cfg,'--vectors',dest/'fit/directions.pt','--output',dest/'edit')
    stage('scripts.score_wikiart_edits','--run',run,'--data',data,'--assets',assets,'--expected',96)
    stage('scripts.summarize_wikiart','--run',run,'--expected',96)
    # Secondary vector diagnostics use the established per-call then per-layer aggregation.
    stage('scripts.wikiart_vector_stats','--run',run,'--output',output/'comparison/vectors.json')
    q=json.loads((run/'quality.json').read_text());assert len(q['rows'])==1152
    for name in ['quality_summary.json','per_image.csv','summary.md','overview_DDIM.png','overview_MW.png','vectors.json']:
        assert (output/'comparison'/name).is_file()
    (output/'SUCCESS.json').write_text(json.dumps(dict(status='complete',rows=1152,paired_images=96,independent_seeds=1,reconstructions='recomputed and SHA256 equality checked',sanity_rows=24),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('repo');a=p.parse_args()
    try:main(a.output,a.repo)
    except Exception:
        path=Path(a.output)/'FAILURE.txt'
        if not path.exists():path.write_text(traceback.format_exc())
        raise
