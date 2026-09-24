"""Pinned, frozen DINOv2 with RGB preprocessing and auditable disk caches."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import SCENES, write_json
from .data import frames, videos

DINO_COMMIT='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
DINO_MODEL='dinov2_vitb14'


def install_source():
    target=Path('cache/dinov2_source')/DINO_COMMIT
    if (target/'hubconf.py').exists():return target
    archive=urllib.request.urlopen(f'https://codeload.github.com/facebookresearch/dinov2/zip/{DINO_COMMIT}').read()
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for item in z.infolist():
            relative=Path(*Path(item.filename).parts[1:])
            if not relative.parts or '..' in relative.parts:continue
            dest=target/relative
            if item.is_dir():dest.mkdir(parents=True,exist_ok=True)
            else:
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(z.read(item))
    write_json(target/'source.json',{'repository':'https://github.com/facebookresearch/dinov2',
               'commit':DINO_COMMIT,'archive_sha256':hashlib.sha256(archive).hexdigest()})
    return target


class DinoFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        source=install_source()
        torch.hub.set_dir(str(Path('cache/torch_hub').resolve()))
        self.backbone=torch.hub.load(str(source.resolve()),DINO_MODEL,source='local',pretrained=True).eval()
        self.backbone.requires_grad_(False)
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])

    def forward(self,bgr_normalized):
        rgb=(bgr_normalized[:,[2,1,0]].float()+1)/2
        rgb=F.interpolate(rgb,size=(252,252),mode='bilinear',align_corners=False)
        rgb=(rgb-self.mean)/self.std
        values=self.backbone.get_intermediate_layers(rgb,n=[5,11],return_class_token=True,norm=True)
        return {'patch6':F.normalize(values[0][0].float(),dim=-1),
                'patch12':F.normalize(values[1][0].float(),dim=-1),
                'cls':F.normalize(values[1][1].float(),dim=-1)}


def reconstruction_distances(extractor,inputs,reconstructions):
    both=extractor(torch.cat([inputs,reconstructions]))
    b=inputs.shape[0]
    result={}
    for name,value in both.items():
        d=(1-(value[:b]*value[b:]).sum(-1)).clamp_min(0)
        if d.ndim>1:d=d.mean(-1)
        result['dino_'+name]=d.cpu().numpy()
    result['dino_multilevel']=(result['dino_patch6']+result['dino_patch12'])/2
    return result


def frame_signature(fs):
    return hashlib.sha256('\n'.join(f'{f.name}:{f.stat().st_size}:{f.stat().st_mtime_ns}' for f in fs).encode()).hexdigest()


def cache_features(scene,root='IPAD_dataset',batch_size=64,limit_videos=0):
    torch.set_num_threads(8)
    extractor=DinoFeatures().cuda().eval()
    out=Path('cache/dino')/scene
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    with torch.inference_mode():
        for split in ['training','testing']:
            for vi,video in enumerate(videos(root,scene,split)):
                if limit_videos and vi>=limit_videos:break
                dest=out/split/video.name;dest.mkdir(parents=True,exist_ok=True)
                fs=frames(video); signature=frame_signature(fs)
                meta=dest/'meta.json'
                if meta.exists():
                    info=json.loads(meta.read_text())
                    if info['signature']!=signature or info['commit']!=DINO_COMMIT:
                        raise ValueError(f'Stale feature cache: {dest}')
                    for key in ['cls','patch12']:
                        array=np.load(dest/f'{key}.npy',mmap_mode='r')
                        expected=(len(fs),768) if key=='cls' else (len(fs),324,768)
                        if array.shape!=expected:raise ValueError(f'Invalid feature cache {dest}/{key}')
                    records.append(info);continue
                cls=np.lib.format.open_memmap(dest/'cls.tmp.npy',mode='w+',dtype=np.float16,shape=(len(fs),768))
                patch=np.lib.format.open_memmap(dest/'patch12.tmp.npy',mode='w+',dtype=np.float16,shape=(len(fs),324,768))
                image_cache=Path('cache/frames')/scene/split/f'{video.name}.npy'
                decoded=np.load(image_cache,mmap_mode='r') if image_cache.exists() else None
                gpu_seconds=0.; wall=time.perf_counter()
                for start in range(0,len(fs),batch_size):
                    end=min(len(fs),start+batch_size)
                    images=np.array(decoded[start:end],copy=True) if decoded is not None else np.stack([cv2.resize(cv2.imread(str(p)),(256,256)) for p in fs[start:end]])
                    x=torch.from_numpy(images).cuda().permute(0,3,1,2).float().div_(127.5).sub_(1)
                    torch.cuda.synchronize();tick=time.perf_counter()
                    result=extractor(x)
                    torch.cuda.synchronize();gpu_seconds+=time.perf_counter()-tick
                    cls[start:end]=result['cls'].cpu().numpy().astype(np.float16)
                    patch[start:end]=result['patch12'].cpu().numpy().astype(np.float16)
                cls.flush();patch.flush();del cls,patch,decoded
                (dest/'cls.tmp.npy').replace(dest/'cls.npy')
                (dest/'patch12.tmp.npy').replace(dest/'patch12.npy')
                info={'scene':scene,'split':split,'video':video.name,'frames':len(fs),'signature':signature,
                      'commit':DINO_COMMIT,'model':DINO_MODEL,'input_size':252,'cache_dtype':'float16',
                      'compute_dtype':'float32','gpu_seconds':gpu_seconds,'wall_seconds':time.perf_counter()-wall}
                write_json(meta,info);records.append(info)
                print(json.dumps(info),flush=True)
    summary={'records':records,'seconds_this_run':time.perf_counter()-started,
             'peak_gb':torch.cuda.max_memory_allocated()/1e9,'limited':bool(limit_videos)}
    write_json(out/('partial.json' if limit_videos else 'complete.json'),summary)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--scene',choices=SCENES,required=True)
    p.add_argument('--root',default='IPAD_dataset')
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--limit-videos',type=int,default=0)
    cache_features(**vars(p.parse_args()))
