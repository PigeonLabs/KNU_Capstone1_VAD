import json
import torch
from ipad.common import environment, write_json

torch.manual_seed(0)
x=torch.randn(2,3,4,8,8,device='cuda',requires_grad=True)
m=torch.nn.Conv3d(3,8,3,padding=1).cuda()
q=torch.randn(2,4,16,32,device='cuda',requires_grad=True)
loss=m(x).square().mean()+torch.nn.functional.scaled_dot_product_attention(q,q,q).square().mean()
loss.backward()
assert torch.isfinite(x.grad).all() and torch.isfinite(q.grad).all()
report={**environment(),'conv3d_backward':True,'attention_backward':True,'loss':loss.item()}
write_json('reports/cuda_check.json',report)
print(json.dumps(report))
