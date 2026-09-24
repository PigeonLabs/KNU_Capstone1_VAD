"""Small real-image DINOv2 verification; never a benchmark performance claim."""
import hashlib
import json
from pathlib import Path
import torch
import numpy as np
from ipad.features import DinoFeatures, reconstruction_distances, DINO_COMMIT
from ipad.common import write_json

torch.set_num_threads(4)
model=DinoFeatures().cuda().eval()
frames=np.load('cache/frames/R01/training/01.npy',mmap_mode='r')
x=torch.from_numpy(np.array(frames[:2],copy=True)).cuda().permute(0,3,1,2).float()/127.5-1
with torch.inference_mode():
    features=model(x)
    identity=reconstruction_distances(model,x,x)
    changed=reconstruction_distances(model,x,-x)
assert not any(p.requires_grad for p in model.parameters())
assert features['patch12'].shape==(2,324,768)
assert features['patch6'].shape==(2,324,768)
assert features['cls'].shape==(2,768)
assert max(float(v.max()) for v in identity.values())<1e-5
assert all(np.isfinite(v).all() for v in changed.values())
assert changed['dino_patch12'].mean()>1e-3
checkpoint=Path('cache/torch_hub/checkpoints/dinov2_vitb14_pretrain.pth')
report={'frozen':True,'input_reconstruction_identity_max':max(float(v.max()) for v in identity.values()),
        'changed_image_distances':{k:v.tolist() for k,v in changed.items()},
        'shapes':{k:list(v.shape) for k,v in features.items()},'commit':DINO_COMMIT,
        'weights_sha256':hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest()}
write_json('reports/dino_verification.json',report)
print(json.dumps(report))
