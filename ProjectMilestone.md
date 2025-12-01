# 15618-Project

# **Large-Scale Parallel Traffic Simulation across Multiple GPUs**
**Team Members:** Mingqi Lu & Jiaying Li  
**URL:** https://github.com/PinkCrow007/15618-Project

## **Work Completed**
According to the schedule in our proposal, we have completed most tasks planned for Week 1 and Week 2, and the multi-GPU simulator is now functionally operational on two A100 GPUs with basic correctness and communication validated.

We have set up the development and environment, including verifying multi-GPU availability, NVLink connectivity, and CUDA peer access support. We prepared the traffic simulation datasets and finalized the representation of the road network and vehicle states. We also ran the original single-GPU simulator to establish a performance baseline and collected profiling traces. Based on these findings, we designed the multi-GPU data structures, including spatial partition metadata and per-GPU vehicle arrays, ensuring that these structures are compatible with CUDA kernels under partitioned execution.

We also have implemented the core foundational components required for multi-GPU execution. This includes completing the naive version of the spatial partitioning logic, which divides the road network and assigns vehicles to either GPU according to their positions. We also implemented ghost-zone synchronization between adjacent partitions, allowing each GPU to maintain up-to-date information about nearby cross-boundary vehicles for collision checks and neighbor queries. Furthermore, we implemented the GPU–GPU communication pipeline, which supports building boundary buffers on each timestep, packaging migrating vehicles, performing P2P transfers over NVLink via cudaMemcpyPeer.

## Updated Detailed Schedule (Dec 1 – Dec 8)

### Dec 1 – Dec 4
| Task | Owner |
|------|--------|
| Run full Nsight Systems / Nsight Compute profiling on 1-GPU and 2-GPU versions | Jiaying |
| Analyze communication patterns and bottlenecks | Mingqi |
| Begin optimizing inter-GPU communication based on profiling results | Jiaying |
| Perform roofline analysis to characterize compute vs memory bottlenecks | Mingqi |

### Dec 4 – Dec 8
| Task | Owner |
|------|--------|
| Implement kernel-level and communication optimizations guided by roofline and Nsight profiling | Both |
| Re-run multi-GPU experiments, collect communication vs computation breakdown | Both |
| Write report | Both |


## **Goals and Deliverables**
The goals and deliverables of the project remain aligned with the original proposal, and the progress so far confirms that the project is on schedule. The highest-priority work for the next phase is to optimize both the communication and computational aspects of the multi-GPU simulator.

Our main goals are:
-	Profile the program using NVIDIA Nsight Systems and Nsight Compute to identify bottlenecks in kernel execution, memory access patterns, and GPU–GPU communication.
-	Optimize GPU–GPU communication, focusing on reducing per-step communication overhead so that data exchange consistently remains dominated by computation time
-	Reduce communication cost by exploiting NVLink peer-to-peer bandwidth and minimizing unnecessary boundary transfers.
-	Optimize per-GPU computation, such as improving memory layouts using structure-of-arrays, reducing warp divergence in update kernels, and optionally leveraging shared memory for local vehicle-neighbor interactions.
-	Produce comprehensive performance plots, including:
	-	single-GPU vs two-GPU speedup curves
	-	communication/computation breakdown
	-	scaling characteristics as workload size increases


With the baseline multi-GPU simulator implemented, we expect that the remaining optimization and evaluation work will proceed smoothly and will be able to meet all the deliverables outlined in the proposal.

## **Preliminary Results**
#### Single-GPU Baseline
We ran part of the dataset on a single A100 GPU (1.25M OD pairs, 07:00–08:00 window).
Key results:

- Simulation time: 11.1 s  
- End-to-end time: ~43 s  

#### 2-GPU Prototype
We also implemented an initial 2-GPU version with naive partitioning and GPU-GPU communication.

The first implementation resized per-GPU vehicle buffers every timestep using `cudaMalloc`/`cudaFree`, because vehicle counts change dynamically. This approach was extremely slow — each timestep triggered heavy allocation costs, making the system too slow to measure meaningfully.

We have since switched to using `thrust::device_vector` for dynamic vehicle storage, eliminating per-timestep allocations. With this improvement, the 2-GPU prototype now runs end-to-end:

- Simulation time: 18.9 s  
- 2-GPU end-to-end time: ~56 s  

This is slower than the single-GPU baseline, primarily due to unoptimized partitioning and inter-GPU communication. The multi-GPU pipeline is now functional, and we expect performance gains as we reduce communication overhead.



## **Issues of Concern**

Although the system is functional, several risks remain:
- Partition imbalance could occur in downtown areas with dense vehicles, heavily affecting 2-GPU speedup. Achieving perfectly balanced spatial partitions while minimizing cross-boundary interactions maps directly to a balanced graph-partitioning problem, which is NP-hard. Finding a better partitioning strategy is non-trivial and may require additional experimentation.
- Kernel-level optimization (warp divergence, SOA memory layout) may take longer than expected.
- End-to-end speedup is uncertain until communication optimizations are completed.
  
These issues are engineering-heavy but solvable; none appear fundamentally blocking.
