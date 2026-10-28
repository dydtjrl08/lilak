"""Compile helpers against the user's ROOT; no project-library dependency."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent


def toolchain(root_executable=None):
    root = root_executable or shutil.which('root')
    if not root:
        raise FileNotFoundError('Set root_executable to the ROOT executable used for this dataset')
    root = Path(root).expanduser().resolve(strict=True)
    config = root.with_name('root-config')
    if not config.is_file():
        raise FileNotFoundError('Matching root-config missing next to '+str(root))
    def get(flag):
        return subprocess.check_output([str(config),flag],text=True,timeout=30).strip()
    return dict(root=str(root),config=str(config),version=get('--version'),
                cxx=get('--cxx'),cflags=get('--cflags'),libs=get('--libs'),libdir=get('--libdir'))


def build(name, root_executable=None):
    if name not in ('observed_reader','truth_reader','step_reader'):
        raise ValueError('Unknown reconstruction native helper: '+str(name))
    source = HERE/(name+'.cpp')
    if not source.is_file():
        raise FileNotFoundError(source)
    tc = toolchain(root_executable)
    digest = hashlib.sha256(source.read_bytes()+json.dumps(tc,sort_keys=True).encode()).hexdigest()[:20]
    folder = Path(os.environ.get('ROOT_RECO_BUILD_DIR',str(HERE/'_native_build'))).expanduser()/digest
    folder.mkdir(parents=True,exist_ok=True)
    target,receipt = folder/name,folder/(name+'.json')
    if target.is_file() and receipt.is_file():
        saved=json.loads(receipt.read_text())
        if saved['binary_sha256']==hashlib.sha256(target.read_bytes()).hexdigest():
            return target.resolve()
        raise RuntimeError('Native binary changed from build receipt: '+str(target))
    print('Compiling',name,'| ROOT',tc['version'],flush=True)
    with tempfile.TemporaryDirectory(prefix='build_',dir=folder) as temp:
        temporary=Path(temp)/name
        command=shlex.split(tc['cxx'])+shlex.split(tc['cflags'])+[
            '-O2',str(source),'-o',str(temporary)]+shlex.split(tc['libs'])+[
            '-lTreePlayer','-Wl,-rpath,'+tc['libdir']]
        try:
            done=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
        except FileNotFoundError as exc:
            raise RuntimeError('Activate the C++ compiler used by this ROOT installation') from exc
        (folder/(name+'_build.log')).write_text(done.stdout)
        if done.returncode:
            raise RuntimeError('Native compilation failed:\n'+done.stdout[-16000:])
        os.replace(temporary,target)
    receipt.write_text(json.dumps(dict(toolchain=tc,command=command,
        binary_sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2))
    return target.resolve()
