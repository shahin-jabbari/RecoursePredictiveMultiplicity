"""Build once per study/dataset/seed, exclusively from this distribution."""
from pathlib import Path
import fcntl, hashlib, json, os, re, shutil, subprocess, sys, tempfile
from io_tools import ROOT

def obtain(kind, ds, seed):
    if not all(re.fullmatch(r'[A-Za-z0-9_-]+', str(x)) for x in (kind,ds,seed)):raise ValueError('Invalid input identity')
    base=Path(os.environ.get('RECOURSE_INPUT_CACHE',str(ROOT.parent/'recourse-model-cache'))).expanduser().resolve()
    manifest=ROOT/'SHA256.json'
    version=hashlib.sha256(manifest.read_bytes()).hexdigest()[:16] if manifest.exists() else 'development'
    if manifest.is_file():
        import platform
        from importlib.metadata import version as package_version
        machine=[sys.version.split()[0],platform.system(),platform.machine()]+[package_version(k) for k in ('numpy','scipy','scikit-learn','torch')]
        version+='-'+hashlib.sha256(json.dumps(machine).encode()).hexdigest()[:10]
    path=base/version/kind/ds/f'seed_{seed}'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (path/'READY.json').is_file():
            if path.exists():raise RuntimeError('Incomplete or modified cache: '+str(path))
            with tempfile.TemporaryDirectory(dir=path.parent,prefix='.preparing-') as tmp:
                dest=Path(tmp)/'inputs';dest.mkdir()
                subprocess.run([sys.executable,'-B',str(ROOT/'replication/regenerate.py'),kind,ds,str(seed),str(dest)],check=True)
                (dest/'READY.json').write_text(json.dumps(dict(study=kind,dataset=ds,seed=seed,source='raw datasets and packaged training code'))+'\n')
                dest.rename(path)
    return path

def copy_part(study, directory, prefix):
    parts=Path(prefix.rstrip('/')).parts
    if len(parts)==3 and parts[0] in ('banks','blocks') and parts[2].startswith('seed_'):
        ds,seed=parts[1],int(parts[2][5:])
    elif len(parts)==4 and parts[0]=='panels':
        ds,seed=parts[2],int(Path(parts[3]).stem[5:])
    else:raise ValueError('Expected a single model bank or panel: '+prefix)
    source=obtain(study,ds,seed)/Path(*parts)
    dest=Path(directory)/Path(*parts)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if source.is_dir():shutil.copytree(source,dest)
    else:shutil.copyfile(source,dest)
    return dest
