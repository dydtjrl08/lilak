"""Bounded, observed-only ROOT helper IPC; no PyROOT/MC/Si dependencies."""
from collections import deque
import os
from pathlib import Path
import select
import subprocess
import tempfile
import threading
import time
import numpy as np
try:
    from .native import build
except ImportError:
    from native import build


class ObservedReader:
    program_name='observed_reader'
    ready_header='RRREADY'
    def __init__(self,path,root_executable=None,timeout_seconds=120):
        self.path=Path(path).expanduser().resolve(strict=True)
        self.timeout_seconds=float(timeout_seconds)
        if not np.isfinite(self.timeout_seconds) or self.timeout_seconds<=0:
            raise ValueError('Reader timeout must be finite and positive')
        self.proc=None;self._temp=None;self._stderr_thread=None
        self.messages=deque(maxlen=100)
        executable=build(self.program_name,root_executable)
        try:
            self._temp=tempfile.TemporaryDirectory(prefix='root_reco_observed_')
            self.proc=subprocess.Popen([str(executable),str(self.path)],stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0,cwd=self._temp.name)
            self._stderr_thread=threading.Thread(target=self._stderr,daemon=True)
            self._stderr_thread.start()
            p=self._header(self.ready_header)
            self.n=int(p[1]);self.root_version=p[2]
        except BaseException:
            self.close();raise

    def _stderr(self):
        for line in iter(self.proc.stderr.readline,b''):
            self.messages.append(line.decode('utf-8',errors='replace').rstrip())

    def _read(self,size,deadline=None):
        if deadline is None:deadline=time.monotonic()+self.timeout_seconds
        data=bytearray()
        while len(data)<size:
            remaining=deadline-time.monotonic()
            if remaining<=0:
                self.close()
                raise TimeoutError('Observed ROOT reader timed out.\n'+'\n'.join(self.messages))
            ready=select.select([self.proc.stdout],[],[],min(remaining,.25))[0]
            if not ready:
                if self.proc.poll() is not None:
                    raise RuntimeError('Observed ROOT reader exited.\n'+'\n'.join(self.messages))
                continue
            block=os.read(self.proc.stdout.fileno(),min(size-len(data),1048576))
            if not block:
                if self._stderr_thread:self._stderr_thread.join(timeout=.1)
                raise RuntimeError('Observed ROOT reader stopped during read.\n'+'\n'.join(self.messages))
            data.extend(block)
        return data

    def _header(self,expected):
        deadline=time.monotonic()+self.timeout_seconds
        for _ in range(200):
            line=bytearray()
            while not line.endswith(b'\n'):
                line.extend(self._read(1,deadline))
                if len(line)>65536:raise RuntimeError('Invalid observed reader header')
            value=line.decode('utf-8',errors='replace').strip()
            if value.startswith(('RRERROR ', 'CNNERROR ')):
                raise RuntimeError(value+'\n'+'\n'.join(self.messages))
            if value.startswith(expected+' '):return value.split()
            self.messages.append(value)
        raise RuntimeError('Missing observed reader header')

    def read(self,entry):
        if isinstance(entry,bool) or int(entry)!=entry or entry<0 or entry>=self.n:
            raise IndexError('Entry outside observed ROOT tree')
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError('Observed ROOT reader is closed')
        self.proc.stdin.write(('W %d\n'%int(entry)).encode('ascii'));self.proc.stdin.flush()
        p=self._header('RRW')
        if int(p[1])!=entry or int(p[2])!=12*64*512:
            raise RuntimeError('Observed ROOT response entry/shape mismatch')
        return np.frombuffer(self._read(12*64*512*4),dtype=np.float32).copy().reshape(12,64,512)

    def close(self):
        p=self.proc
        if p is not None:
            if p.poll() is None:
                try:p.stdin.write(b'Q\n');p.stdin.flush();p.wait(timeout=1)
                except (BrokenPipeError,OSError,subprocess.TimeoutExpired):
                    p.terminate()
                    try:p.wait(timeout=1)
                    except subprocess.TimeoutExpired:p.kill();p.wait(timeout=2)
            if self._stderr_thread:self._stderr_thread.join(timeout=.5)
            for pipe in (p.stdin,p.stdout,p.stderr):
                if pipe:pipe.close()
            self.proc=None
        if self._temp:self._temp.cleanup();self._temp=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
