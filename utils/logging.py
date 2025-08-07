from typing import Optional
import os
from warnings import warn
import shutil
import numpy as np
from numbers import Number
import torch


class Averager:
    def __init__(self):
        self.data: dict[str, list[torch.Tensor]] = dict()
        self.subdata: dict[str, list[torch.Tensor]] = dict()
        self.count: dict[str, int] = dict()
    
    def push(self, subdata: dict[str, torch.Tensor], flush: bool=False):
        for key, val in subdata.items():
            if isinstance(val, torch.Tensor):
                val = val.clone().detach().cpu()
            elif isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            else:
                val = torch.tensor(val)
                
            if key in self.subdata:
                self.subdata[key].append(val)
                self.count[key] += 1
            else:
                self.subdata[key] = [val]
                self.count[key] = 1
        
        if flush:
            return self.flush()
                
    def flush(self) -> dict[str, torch.Tensor]:
        result = dict()
        for key in self.subdata.keys():
            val = sum(self.subdata[key]) / self.count[key]
            if key in self.data:
                self.data[key].append(val)
            else:
                self.data[key] = [val]
            result[key] = val
        self.subdata.clear()
        self.count.clear()
        return result
    
    def take(self, index: int) -> dict[str, torch.Tensor]:
        return {
            key: val_list[index]
            for key, val_list in self.data.items()
        }
    
    def gather(self) -> dict[str, torch.Tensor]:
        return {
            key: torch.stack(val_list)
            for key, val_list in self.data.items()
        }
    
    def clear(self):
        self.data.clear()
        self.subdata.clear()
        self.count.clear()

    def __iter__(self):
        return iter(self.gather())


class CheckpointWriter:
    def __init__(
        self,
        dir_name: str='./output',
        save_first: bool=False,
        save_every: int=0,
        save_best: bool=True,
        save_last: bool=True,
        larger_better: bool=False,
        overwrite: bool=False,
    ):
        self.dir_name = dir_name
        self.save_first = save_first
        self.save_every = save_every
        self.save_best = save_best
        self.save_last = save_last
        self.larger_better = larger_better

        if os.path.exists(dir_name):
            if overwrite:
                shutil.rmtree(dir_name)
                warn(
                    f"'{dir_name}' already exists, but it has been removed for overwriting.",
                    RuntimeWarning,
                )
            else:
                raise FileExistsError(f"'{dir_name}' already exists.")
        os.makedirs(dir_name, exist_ok=False)
        
        self.count = 0
        self.best = \
            -float('inf') if larger_better else \
            +float('inf')
            
    def write(self, checkpoint: dict, score: Number):
        self.count += 1

        if self.save_first and (self.count == 1):
            torch.save(checkpoint, os.path.join(self.dir_name, 'first.pt'))
        
        if (self.save_every > 0) and ((self.count % self.save_every) == 0):
            torch.save(checkpoint, os.path.join(self.dir_name, f'{self.count:06d}.pt'))
        
        if self.save_best:
            if self.larger_better:
                save = self.best <= score
            else:
                save = self.best >= score
            
            if save:
                self.best = score
                torch.save(checkpoint, os.path.join(self.dir_name, 'best.pt'))
        
        if self.save_last:
            torch.save(checkpoint, os.path.join(self.dir_name, 'last.pt'))
    
    def copy_code(self, src_fname: os.PathLike, dst_fname: Optional[os.PathLike]=None):
        if not os.path.exists(src_fname):
            raise FileNotFoundError(f"Source file '{src_fname}' does not exist.")
        
        code_dname = os.path.join(self.dir_name, 'code')
        if not os.path.exists(code_dname):
            os.makedirs(code_dname, exist_ok=True)
        
        if dst_fname is None:
            dst_fname = os.path.basename(src_fname)
        dst_fname = os.path.join(code_dname, dst_fname)
        
        return shutil.copy(src_fname, dst_fname)
