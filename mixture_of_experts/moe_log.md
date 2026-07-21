# Optimizing MOE kernel

This is a work log where I start with by going through the torch profile of a reference moe kernel from a GPUMODE Challenge and build on that going ahead. There are a lot of kernels available publicly on the gpumode platform, so will start by going through some of the kernels there first. I will focus on triton kernels first and then move on to HIP. I will be using MI300X for this worklog (also because MI355X are not available to me.)

- look into this "Expert Choice Routing took a different approach: instead of having tokens choose experts, each expert chooses its top-K tokens, guaranteeing perfect load balance by construction."
- Auxilary-loss-free load balancing

## Profiling the reference pytorch code

I had been reading about torch lately, so this should be aligned to that. Also a first glance at the profile trace made me realize that with this I can read through the functions in pytorch.

- profiler initialization, how torch/profile/profiler.py is connected to torch/autograd/profiler.py, how Profiler.start() calls _KinetoProfile.start_trace(), which finally invokes _start_trace() in torch/autograd/profiler.py

- `hipExtModuleLaunchKernel` is used to launch kernels from dynamically loaded modules that were either JIT compiled or loaded from an external binary file at runtime using `hipModuleLoad`, while `hipLaunchKernel` is used to launch Ahead-Of-Time(AOT) compiled kernels like `aten::silu`.

- Normalization by input_dimesion vs output_dimension:
 - sqrt(fan_in) : keeps forward activations stable
 - sqrt(fan_out) : keeps the backward gradients stable

### Observations

- In one Expert forward pass 5 kernels are getting executed, specifically
    1. aten::linear (W_gate masking) (uses a external module kernel, hipBLAS)
    2. aten::silu  (activation) (uses aten native silu kernel)
    3. aten::linear (up projection) (same as W_gate)
    4. aten::mul (vectorized_elementwise_kernel) (uses at::native::BinaryFunctor)
    5. aten::linear (down projection) (same as W_gate)

    maybe fusing them is one option to optimize

- The routing gate is launching 3 kernels
    - aten::linear  (hipBLAS kernel)
    - softmax (void (anonymous namespace)::softmax_warp_forward)
    - topk (void at::native::sbtopk::gatherTopK)

- the argsort
    some D -> D memcpy kernels
    aten::arange
    void at::native::bitonicSortKVInPlace (with hipOccupancyMaxpotentialBlockSizes)

#### 1st Expert Kernel notes (triton_ref_01.py):

- tl.load builds a 2D grid of addresses, it does not matter which axis you treat as rows/columns.One will be transpose of the other, this may/may not help with coalescing the loads.

- Learnt how to implement multiple gemms in one kernel with the help of reductions as the last gemm was tricy becuase for this op,  the number of output tiles was not equal to the number of triton programs/blocks so it had to be improvised by using atomic operations becuase each block would compute a partial result of the output matrix.

#### 2nd Expert Kernel Notes (triton_ref_02.py, my_triton_01.py):

- *Optimized Weight loads* : instead of loading weights per ecxpert, weights are loaded in batches (one for each: gate, up, down), might lead to optimised loads.
- *flat_expert_indices* : indices of the expert to rout a particular token, `tensor([0, 2, 0, 3, 0, 2, 0, 3, 3, 1, 0, 2, 2, 1, 3, 0], device='cuda:0')`
- *sorted_expert_ids* : sorts `flat_expert_indices` based on experts, `tensor([0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], device='cuda:0')`
- *sort_order_indices / perm_indices* : indices of `flat_expert_indices` that were grouped based on the experts, `tensor([ 6,  4,  0,  2, 10, 15, 13,  9,  5,  1, 11, 12, 14,  8,  3,  7], device='cuda:0')`
- *token_indices/ indices_to_gather* : now that we have the sorted indices in `sorted_flat_expert_indices`, we get the tokens by dividing them by `n_expert_pre_token`, `tensor([3, 2, 0, 1, 5, 7, 6, 4, 2, 0, 5, 6, 7, 4, 1, 3], device='cuda:0')`

- Because teh expert weights are stores as [d_out, d_in], which is consistent for pytorch but for our tl.dot, we had to load the tile in a **transposed** manner so that it is convinent for the next operation.

- Also the partial weights to be written into the out tensor are not aligned (they are aligned with the transposed view of the out tensor)
    - tile shape: [BK, BN]
    - out shape: [num_tokens, d_hidden]
    - BK is along the hidden dimension and BN is along th number of tokens so we had to write to swapped indices.


#### NOTE We have not dived into the hyper params yet!!!


#### 3rd Expert kernel Notes (triton_ref_02):
    
- Here we used batching and launched all the expert forward passes with just one kernel. The grids 0th axis being the experts and the 1st axis handling each experts M*dhid space (which was padded)

- Input (x) had to be perpared as per the kernel's expectations.
