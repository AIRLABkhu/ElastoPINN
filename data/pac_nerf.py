import os
from typing import Iterator, Optional
from glob import glob
import json
import torch
from torch.utils.data import Dataset
import torchvision.io
import pytorch3d.io


class PACNeRFDataset(Dataset):
    INSTANCES = [
        'cream',
        *[f'newtonian/{i}' for i in range(10)],
        *[f'plasticine/{i}' for i in range(10)],
        *[f'sand/{i}' for i in range(5)],
        'rope2',
        'playdoh',
        'letter',
        'cat',
        *[f'elastic/{i}' for i in range(10)],
        'torus',
        'droplet',
        'trophy',
        *[f'non_newtonian/{i}' for i in range(10)],
        'rope',
        'bird',
        'toothpaste',
    ]
    
    def __init__(
        self,
        dataroot: str,
        instance: str,
        verbose: bool=False,
        max_frames: Optional[int] = None,
    ):
        self._dataroot = dataroot
        self._instance = instance
        
        if instance.startswith('plasticine/'):
            dname_data = os.path.join(dataroot, 'data', instance.replace('plasticine', 'plasticine_batch'))
        elif instance.startswith('sand/'):
            dname_data = os.path.join(dataroot, 'data', instance.replace('sand', 'sand_batch'))
        else:
            dname_data = os.path.join(dataroot, 'data', instance)
            
        dname_png = os.path.join(dname_data, 'data')
        dname_ply = os.path.join(dataroot, 'simulation_data', instance)
        num_views = len(glob(os.path.join(dname_png, 'r_*_-1.png')))
        num_frames = len(glob(os.path.join(dname_png, 'r_0_*.png'))) - 1

        # 내가 임의으로 frame 설정할 수 있게. 
        if max_frames is not None:
            num_frames = min(num_frames, max_frames)
        print(f'selected num_frames: {num_frames}')

        if verbose:
            print('PACNeRFDataset:', 'collecting images and point clouds.')
        self.images = [[None for _ in range(num_frames)] for _ in range(num_views)]
        self.pointclouds = [None for _ in range(num_frames)]
        for frame in range(num_frames):
            for view in range(num_views):
                png_fname = os.path.join(dname_png, f'r_{view}_{frame}.png')
                img = torchvision.io.read_image(png_fname)
                self.images[view][frame] = img
            ply_fname = os.path.join(dname_ply, f'{frame}.ply')
            pointcloud, _ = pytorch3d.io.load_ply(ply_fname)
            self.pointclouds[frame] = pointcloud
        self.images = torch.stack([torch.stack(imgs) for imgs in self.images])
        self.images = self.images.float() / 255.0
        num_points = torch.tensor([*map(len, self.pointclouds)], dtype=torch.long)
        max_n_points = num_points.max()
        self.padding_sizes = max_n_points - num_points
        self.pointclouds = torch.stack([
            torch.cat([pc, torch.zeros(pad, 3, dtype=pc.dtype, device=pc.device)], dim=0)
            for pc, pad in zip(self.pointclouds, self.padding_sizes)
        ], dim=0)
        
        if verbose:
            print('PACNeRFDataset:', 'gathering metadata.')
        fname_json = os.path.join(dname_data, 'all_data.json')
        with open(fname_json, 'r') as file:
            metadata = json.load(file)
        self.metadata = {
            'time': torch.empty(num_views, num_frames, dtype=torch.float),
            'c2w': torch.empty(num_views, num_frames, 3, 4, dtype=torch.float),
            'intrinsic': torch.empty(num_views, num_frames, 3, 3, dtype=torch.float),
        }
        for sample in metadata:
            parts = os.path.splitext(sample['file_path'])[0].split('_')
            view, frame = int(parts[1]), int(parts[2])
            
            # max frame 보다 크면 집어 넣지 마라 
            
            if (frame < 0) or (frame >= num_frames):
                continue
            self.metadata['time'][view, frame] = sample['time'] * 2.0  # map to [0, 1]
            self.metadata['c2w'][view, frame] = torch.tensor(sample['c2w'])
            self.metadata['intrinsic'][view, frame] = torch.tensor(sample['intrinsic'])
            
        self.metadata['validity'] = torch.logical_and(  # filter for abnormal timestamp
            0.0 <= self.metadata['time'],
            self.metadata['time'] <= 1.0,
        )
        
    @property
    def dataroot(self) -> str:
        return self._dataroot
    
    @property
    def instance(self) -> str:
        return self._instance
    
    @property
    def num_views(self) -> int:
        return self.images.size(0)
    
    @property
    def num_frames(self) -> int:
        return self.images.size(1)
    
    @property
    def image_size(self) -> tuple[int, int]:
        return tuple(self.images.shape[-2:])
    
    @property
    def num_points(self) -> int:
        return self.pointclouds.size(0)
    
    @property
    def ground_pos(self) -> float:
        return self.pointclouds[..., self.up_index].quantile(0.05).item()
        # return self.pointclouds[:, self.up_index].min().item()
        # return 0
    
    @property
    def up_index(self) -> int:
        return 1
    
    def __len__(self) -> int:
        return self.num_frames
    
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            'projection': self.images[:, index],
            'geometry': self.pointclouds[0],
            'displacement': self.pointclouds[index] - self.pointclouds[0],
            'time': self.metadata['time'][:, index],
            'c2w': self.metadata['c2w'][:, index],
            'intrinsic': self.metadata['intrinsic'][:, index],
            'validity': self.metadata['validity'][:, index],
        }
    
    def __str__(self) -> str:
        return f"{type(self).__name__}(dataroot='{self.dataroot}', instance='{self.instance}')"
    
    def __iter__(self) -> Iterator:
        return iter([
            ('projection', self.images),
            ('geometry', self.pointclouds),
            ('displacement', self.pointclouds - self.pointclouds[0:1]),
            ('time', self.metadata['time']),
            ('c2w', self.metadata['c2w']),
            ('intrinsic', self.metadata['intrinsic']),
            ('validity', self.metadata['validity']),
        ])
