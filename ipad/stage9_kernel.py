"""INT4-storage/BF16-arithmetic fused GEMM for bounded stage-9 diagnostics."""
import torch
import triton
import triton.language as tl

CONFIGS=[(16,64,64,4),(32,64,64,4),(32,128,64,4),(64,64,64,4),(64,128,64,8),(32,64,128,4)]


@triton.jit
def packed_gemm(X,Q,S,Z,B,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,HAS_BIAS:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    accum=tl.full((BM,BN),0,tl.float32)
    for block in range(tl.cdiv(K,BK)):
        ks=block*BK+kk
        a=tl.load(X+rows[:,None]*K+ks[None,:],(rows[:,None]<M)&(ks[None,:]<K),other=0)
        packed=tl.load(Q+cols[None,:]*(K//2)+ks[:,None]//2,(cols[None,:]<N)&(ks[:,None]<K),other=0)
        q=tl.where(ks[:,None]%2==0,packed>>4,packed&15).to(tl.float32)
        scale=tl.load(S+cols[None,:]*(K//128)+ks[:,None]//128,(cols[None,:]<N)&(ks[:,None]<K),other=0).to(tl.float32)
        zero=tl.load(Z+cols[None,:]*(K//128)+ks[:,None]//128,(cols[None,:]<N)&(ks[:,None]<K),other=0).to(tl.float32)
        weight=((q-8)*scale+zero).to(tl.bfloat16)
        accum=tl.dot(a,weight,accum)
    # Match PackedInt4Linear matmul rounding before bias addition.
    result=accum.to(tl.bfloat16)
    if HAS_BIAS:
        bias=tl.load(B+cols,cols<N,other=0)
        result=(result.to(tl.float32)+bias[None,:].to(tl.float32)).to(tl.bfloat16)
    tl.store(Y+rows[:,None]*N+cols[None,:],result,(rows[:,None]<M)&(cols[None,:]<N))


def fused_linear(x,layer,config):
    assert x.is_contiguous() and x.dtype==torch.bfloat16
    shape=x.shape;matrix=x.reshape(-1,shape[-1]);m,k=matrix.shape
    assert k==layer.k and layer.group_size==128
    y=torch.empty((m,layer.n),device=x.device,dtype=torch.bfloat16)
    bm,bn,bk,warps=config
    packed_gemm[(triton.cdiv(m,bm),triton.cdiv(layer.n,bn))](matrix,layer.packed,layer.scale,layer.zero,layer.bias if layer.bias is not None else layer.scale,y,m,layer.n,k,bm,bn,bk,layer.bias is not None,num_warps=warps,num_stages=3,enable_fp_fusion=False)
    return y.reshape(*shape[:-1],layer.n)
