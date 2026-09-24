"""Frame-ID preserving video loaders. Test labels never enter the model."""
import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from .common import SCENES, write_json


def videos(root, scene, split):
    return sorted((Path(root) / scene / split / 'frames').iterdir(), key=lambda p: int(p.name))


def frames(video):
    return sorted(video.glob('*.jpg'), key=lambda p: int(p.stem))


def validation_videos(root, scene):
    names = [p.name for p in videos(root, scene, 'training')]
    shuffled = np.random.default_rng(0).permutation(names)
    return sorted(shuffled[:max(1, int(np.ceil(len(names) * .2)))].tolist())


def label_path(root, scene, video):
    return Path(root) / scene / 'test_label' / f'{int(video):03d}.npy'


def audit(root, output, decode=True):
    report = {'root': str(Path(root).resolve()), 'scenes': {}, 'fatal_errors': []}
    for scene in SCENES:
        records = []
        for split in ('training', 'testing'):
            for video in videos(root, scene, split):
                fs = frames(video)
                ids = [int(p.stem) for p in fs]
                row = {'split': split, 'video': video.name, 'frames': len(fs),
                       'first_id': min(ids), 'last_id': max(ids),
                       'missing_ids': sorted(set(range(min(ids), max(ids)+1)) - set(ids))}
                if row['missing_ids'] or ids[0] != 0 or len(fs) < 16:
                    report['fatal_errors'].append(f'{scene}/{split}/{video.name}: invalid frame sequence')
                if decode:
                    shapes = set()
                    for p in fs:
                        image = cv2.imread(str(p))
                        if image is None:
                            report['fatal_errors'].append(f'Cannot decode {p}')
                        else:
                            shapes.add(tuple(image.shape))
                    row['image_shapes'] = [list(s) for s in sorted(shapes)]
                if split == 'testing':
                    lp = label_path(root, scene, video.name)
                    y = np.load(lp, allow_pickle=False).reshape(-1)
                    row.update(labels=len(y), label_values=np.unique(y).tolist(),
                               label_sha256=hashlib.sha256(lp.read_bytes()).hexdigest(),
                               strict_eligible=len(y) == len(fs))
                    if not set(np.unique(y)).issubset({0, 1}):
                        report['fatal_errors'].append(f'Nonbinary label: {lp}')
                records.append(row)
        report['scenes'][scene] = {'videos': records, 'validation_videos': validation_videos(root, scene)}
    write_json(output, report)
    return report


def cache_video(video, destination):
    fs = frames(video)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix('.tmp.npy')
    array = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.uint8, shape=(len(fs), 256, 256, 3))
    for i, f in enumerate(fs):
        image = cv2.imread(str(f))
        if image is None:
            raise ValueError(f'Unreadable image: {f}')
        array[i] = cv2.resize(image, (256, 256))
    array.flush()
    del array
    tmp.replace(destination)


class Clips(Dataset):
    def __init__(self, root, scene, split='training', subset='all', cache='cache/frames',
                 label_policy='strict', stride=1):
        if scene not in SCENES:
            raise ValueError('Only R01-R04 are authorized for this experiment')
        self.scene, self.split = scene, split
        self.records, self.samples, self.maps = [], [], {}
        heldout = validation_videos(root, scene)
        for video in videos(root, scene, split):
            if split == 'training' and subset != 'all' and ((video.name in heldout) != (subset == 'val')):
                continue
            fs = frames(video)
            ids = [int(f.stem) for f in fs]
            if ids != list(range(len(fs))):
                raise ValueError(f'Non-contiguous zero-based frames: {video}')
            y = np.load(label_path(root, scene, video.name)).reshape(-1) if split == 'testing' else None
            if y is not None and len(y) != len(fs) and label_policy == 'strict':
                continue
            rec = {'video': video.name, 'files': fs, 'length': len(fs), 'labels': y,
                   'cache': Path(cache)/scene/split/f'{video.name}.npy' if cache else None}
            index = len(self.records)
            self.records.append(rec)
            self.samples.extend((index, s) for s in range(0, len(fs)-15, stride))
        if not self.samples:
            raise ValueError('No usable clips')

    def __len__(self):
        return len(self.samples)

    @property
    def median_period(self):
        return float(np.median([r['length'] for r in self.records]))

    def __getitem__(self, index):
        ri, start = self.samples[index]
        rec = self.records[ri]
        cached = rec['cache']
        if cached and cached.exists():
            if ri not in self.maps:
                self.maps[ri] = np.load(cached, mmap_mode='r')
            clip = np.array(self.maps[ri][start:start+16], copy=True)
        else:
            clip = np.stack([cv2.resize(cv2.imread(str(p)), (256,256)) for p in rec['files'][start:start+16]])
        result = {'clip': clip, 'video': rec['video'], 'start': start, 'frame': start+8}
        # Match official supervision: phase of the clip START, reconstruction of index 8.
        if self.split == 'training':
            result['phase'] = min(199, start*200//rec['length'])
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='IPAD_dataset')
    p.add_argument('--output', default='reports/data_audit.json')
    p.add_argument('--cache', action='store_true')
    p.add_argument('--scenes', nargs='+', default=list(SCENES), choices=SCENES)
    a = p.parse_args()
    report = audit(a.root, a.output)
    print({s: {'train': sum(r['frames'] for r in v['videos'] if r['split']=='training'),
               'test': sum(r['frames'] for r in v['videos'] if r['split']=='testing'),
               'excluded': [r['video'] for r in v['videos'] if r.get('strict_eligible') is False]}
           for s, v in report['scenes'].items()}, flush=True)
    if report['fatal_errors']:
        raise RuntimeError(report['fatal_errors'])
    if a.cache:
        for scene in a.scenes:
            for split in ('training', 'testing'):
                for video in videos(a.root, scene, split):
                    dest = Path('cache/frames')/scene/split/f'{video.name}.npy'
                    if not dest.exists():
                        cache_video(video, dest)
                print(f'Cached {scene}/{split}', flush=True)


if __name__ == '__main__':
    main()
