# 15618-Project

# **Large-Scale Parallel Traffic Simulation across Multiple GPUs**
**Team Members:** Mingqi Lu & Jiaying Li  
**URL:** https://github.com/PinkCrow007/15618-Project



## **SUMMARY**
We plan to extend a single-GPU microscopic traffic simulator into a **multi-GPU parallel system** capable of simulating significantly larger road networks. Our work focuses on spatially partitioning the simulation across two GPUs and designing efficient GPU-GPU communication to synchronize vehicle states.



## **BACKGROUND**
Microscopic traffic simulation updates the state of each individual vehicle (position, speed, lane changes, interactions with neighbors) at every timestep. This is computationally expensive because the simulation may involve millions of vehicles, and each update depends on nearby traffic conditions.

A typical single-GPU implementation assigns **one vehicle per CUDA thread**, enabling the simulation to advance in parallel. However, A single GPU may become limited by memory capacity, memory bandwidth and global synchronization overhead as vehicle/road-network scale increases.

To scale further, the simulation domain can be partitioned across multiple GPUs. Each GPU simulates vehicles in its region, but vehicles near region boundaries must be exchanged every timestep. Most interactions are local, making spatial partitioning a natural parallelization strategy.



## **THE CHALLENGE**

### **Workload Characteristics**
- **Local interactions:** each vehicle interacts with only nearby vehicles, creating irregular but spatially bounded dependencies.
- **Dynamic migration:** vehicles frequently cross region boundaries, requiring inter-GPU transfers.
- **Irregular memory access:** vehicles are not neatly arranged in arrays, reducing locality.

### **Mapping to Multi-GPU**
- GPUs must exchange boundary data at every timestep, which can dominate runtime if poorly optimized.
- Traffic conditions change over time, causing load imbalance between GPUs.
- Divergent control flow (e.g., free-flow vs. congestion) can reduce SIMT efficiency.

### **What We Hope to Learn**
- How to efficiently manage GPU-to-GPU communication. 
- How to profile and reason about multi-GPU performance
- How to choose and tune domain partitioning schemes.  
- How workload characteristics influence multi-GPU scaling and what kinds of optimizations are most effective.



## **RESOURCES**
- **Hardware:** Two NVIDIA A100 GPUs connected with NVLink.  
- **Software:** CUDA, Nsight Systems/Compute.  
- **Codebase:** A simple single-GPU traffic simulator.  



## **GOALS AND DELIVERABLES**

### **Plan to Achieve (75%)**
- Implement naive workload partitioning that splits the road network and vehicles across two GPUs.
- Introduce ghost zones to maintain information across partition boundaries.
- Implement GPU-GPU communication of boundary vehicles.  


### **Expected Goal (100%)**
- Optimize communication using techniques such as       
    - asynchronous CUDA streams to overlap boundary exchanges with computation,
    - high-bandwidth peer-to-peer transfers (e.g., cudaMemcpyPeer over NVLink),
    - Optimize memory access patterns (e.g., structure-of-arrays).  
    - lightweight synchronization to update ghost zones with minimal stalling.
- Demonstrate **speedup** on large workloads.  

### **Stretch Goals (125%)**

Further optimize intra-GPU performance using techniques such as shared memory, deeper profiling-driven analysis, and explore alternative parallelization strategies.

### **Demo**
We plan to show 1-GPU vs 2-GPU performance profiling comparison.



## **PLATFORM CHOICE**
A100 GPUs are well suited for this project because:
- NVLink offers high-bandwidth, low-latency GPU-GPU communication.  
- CUDA provides fine-grained control of memory transfers and asynchronous execution.  
- The workload maps naturally to massively parallel GPU threads.



## **SCHEDULE**

### **Week 1 (Nov 17)**
- Set up development environment and verify multi-GPU availability (NVLink, peer access, compute capability).

- Prepare or generate traffic datasets and define road-network representation.

- Run and analyze the single-GPU baseline; collect initial profiling traces (kernel time, memory throughput, occupancy).

- Design data structures for multi-GPU state

### **Week 2 (Nov 24)**
- Implement partitioning across two GPUs.  
- Implement ghost-zone creation, update rules, and consistency conditions.
- Implement GPU-GPU communication (buffer building, atomic writes, P2P transfers, correctness checks).
- Validate correctness through small synthetic cases.
- Profile communication vs. computation and identify bottlenecks.

### **Week 3 (Dec 1)**
- Optimize communication: overlapping with compute via streams, batching, minimizing synchronization, tuning buffer operations.
- Optimize computation: improve memory layout, reduce divergence, optionally use shared memory for local interactions.
- Produce speedup plots and communication/computation breakdowns.
- Finalize demo materials (partition visualization, Nsight timelines, graphs).


