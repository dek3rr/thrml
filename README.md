<h1 align='center'>THRML</h1>

This is a fork of [Extropic's thrml library](https://github.com/Extropic-AI/thrml) with a focus on squeezing more performance out of the JAX implementation. The API is identical — you can drop this in as a replacement.

---

## What is THRML?

THRML is a JAX library for sampling from probabilistic graphical models (PGMs). The core idea is **blocked Gibbs sampling**: rather than updating one node at a time, nodes are grouped into blocks and sampled all at once. THRML compiles these block updates into tight JAX programs that run as fused GPU kernels — so even large, heterogeneous graphs stay fast.

It's a good fit for Ising models, RBMs, discrete EBMs, or anything with a sparse bipartite graph structure.

---

## What is THRML-Boost?

**Parallel tempering runs all chains at once**
The original looped over chains in Python, unrolling N copies of the same computation into the XLA graph. THRML-Boost uses `jax.vmap` instead — one kernel, all chains in parallel. Compile time no longer grows with chain count, and GPU utilization goes up significantly. This is the biggest win.

**No redundant work inside the sampler loop**
Each Gibbs iteration was rebuilding the full global state array from scratch. It's now threaded through as a carry so observers can read it directly without recomputing anything.

**Accumulator dtype is fixed at construction**
The moment accumulator was inferring its dtype on every scan step, which silently triggered float64 emulation on GPU. Now it's set once upfront (float32 by default).

**Energy evaluation skips unnecessary work**
`energy()` was rebuilding a `BlockSpec` every call, including 4 times per swap attempt during parallel tempering. Pre-built specs are now passed through directly.

**Global state layout is now deterministic**
The original used a Python `set` to determine global state ordering — non-deterministic across runs, which could silently break saved states. Replaced with insertion-order-preserving deduplication.

---

## Speedups

| Workload | Speedup |
|---|---|
| Parallel tempering, 8–16 chains | **2–3×** | TBD
| Moment accumulation on GPU | up to **4×** | TBD
| Single-chain sampling | **5–15%** | TBD

---

## Installation

```bash
git clone https://github.com/dek3rr/thrml_boost.git
cd thrml
pip install -e .
```

## Quick example

```python
import jax
import jax.numpy as jnp
from thrml_boost import SpinNode, Block, SamplingSchedule, sample_states
from thrml_boost.models import IsingEBM, IsingSamplingProgram, hinton_init

nodes = [SpinNode() for _ in range(5)]
edges = [(nodes[i], nodes[i+1]) for i in range(4)]
model = IsingEBM(nodes, edges, jnp.zeros(5), jnp.ones(4) * 0.5, jnp.array(1.0))

free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

key = jax.random.key(0)
k_init, k_samp = jax.random.split(key, 2)
init_state = hinton_init(k_init, model, free_blocks, ())
schedule = SamplingSchedule(n_warmup=100, n_samples=1000, steps_per_sample=2)

samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
```

## Citing THRML

```bibtex
@misc{jelinčič2025efficientprobabilistichardwarearchitecture,
      title={An efficient probabilistic hardware architecture for diffusion-like models}, 
      author={Andraž Jelinčič and Owen Lockwood and Akhil Garlapati and Guillaume Verdon and Trevor McCourt},
      year={2025},
      eprint={2510.23972},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.23972}, 
}
```
