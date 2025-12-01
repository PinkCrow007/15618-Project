# 15618-Project

# **Large-Scale Parallel Traffic Simulation across Multiple GPUs**
**Team Members:** Mingqi Lu & Jiaying Li  
**URL:** https://github.com/PinkCrow007/15618-Project

## **Work Completed**
According to the schedule in our proposal, we have completed all tasks planned for Week 1 and Week 2, and the multi-GPU simulator is now functionally operational on two A100 GPUs with basic correctness and communication validated.

We have set up the development and profiling environment, including verifying multi-GPU availability, NVLink connectivity, and CUDA peer access support. We prepared the traffic simulation datasets and finalized the representation of the road network and vehicle states. We also ran the original single-GPU simulator to establish a performance baseline and collected profiling traces. Based on these findings, We designed the multi-GPU data structures, including spatial partition metadata and per-GPU vehicle arrays, ensuring that these structures are compatible with CUDA kernels under partitioned execution.

We also have implemented the core foundational components required for multi-GPU execution. This includes completing the initial version of the spatial partitioning logic, which divides the road network and assigns vehicles to either GPU according to their positions. Furthermore, We implemented the GPU–GPU communication pipeline, which supports building boundary buffers on each timestep, packaging migrating vehicles, performing P2P transfers over NVLink via cudaMemcpyPeer.

To ensure correctness, we constructed a series of small synthetic tests to verify that boundary exchanges and inter-GPU synchronization behave as intended. We performed preliminary debugging using CUDA memcheck and timeline analysis to confirm that each timestep’s communication and computation phases are correctly ordered. 

## **Goals and Deliverables**
The goals and deliverables of the project remain aligned with the original proposal, and the progress so far confirms that the project is on schedule. The highest-priority work for the next phase is to optimize both the communication and computational aspects of the multi-GPU simulator.

Our main goals are:
-	Optimize GPU–GPU communication, including overlapping computation and communication using asynchronous CUDA streams, reducing global synchronization by deferring or batching ghost-zone updates.
-	Reduce communication cost by exploiting NVLink peer-to-peer bandwidth and minimizing unnecessary boundary transfers.
-	Optimize per-GPU computation, such as improving memory layouts using structure-of-arrays, reducing warp divergence in update kernels, and optionally leveraging shared memory for local vehicle-neighbor interactions.
-	Produce comprehensive performance plots, including:
	-	single-GPU vs two-GPU speedup curves,
	-	communication/computation breakdown,
	-	scaling characteristics as workload size increases,



With the baseline multi-GPU simulator implemented, we expect that the remaining optimization and evaluation work will proceed smoothly and will be able to meet all the deliverables outlined in the proposal.

## **Preliminary Results**

## **Issues of Concern**