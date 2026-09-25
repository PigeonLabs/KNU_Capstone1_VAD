from scripts.summarize_stage9_1 import kernel_audit


def test_weight_only_and_compiler_graph_are_not_integer_gemm_or_cuda_graph():
    result=kernel_audit(['aten::_weight_int8pack_mm','## Call CompiledFxGraph abc ##','Pregraph bytecode'])
    assert result['weight_only_int8_kernel']
    assert not result['integer_gemm'] and not result['cuda_graph_api']


def test_real_kernel_and_cuda_launch_evidence():
    assert kernel_audit(['aten::_int_mm'])['integer_gemm']
    assert kernel_audit(['void cutlass_i16832gemm_s8_tn()'])['integer_gemm']
    assert kernel_audit(['cudaGraphLaunch'])['cuda_graph_api']
    assert kernel_audit(['aten::_weight_int4pack_mm'])['packed_int4']
