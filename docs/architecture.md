# Developer Documentation

## What is `THRML`?

`THRML` is a [JAX](https://docs.jax.dev/en/latest/)‑based Python package for efficient [block Gibbs sampling](https://proceedings.mlr.press/v15/gonzalez11a/gonzalez11a.pdf) of graphical models at scale. It provides the tools to do block Gibbs sampling on any graphical model, and includes built-in support for models such as [Boltzmann Machines](https://www.cs.toronto.edu/~fritz/absps/cogscibm.pdf).


## How does `THRML` work?

From a user perspective, there are three main components: blocks, factors, and programs. For detailed usage examples, see the example notebooks.

**Blocks** are fundamental to `THRML` since it implements block sampling. A `Block` is a collection of nodes of the same type with implicit ordering.

**Factors** organize interactions between variables into a [bipartite graph of factors and variables](https://ocw.mit.edu/courses/6-438-algorithms-for-inference-fall-2014/3e3e9934d12e3537b4e9b46b53cd5bf1_MIT6_438F14_Lec4.pdf). Factors synthesize collections of interactions via `InteractionGroups` and must implement a `to_interaction_groups()` method.

**Programs** are the key orchestrating data structures. `BlockSamplingProgram` handles the mapping and bookkeeping for padded block Gibbs sampling, managing global state representations efficiently for JAX. `FactorSamplingProgram` is a convenient wrapper that converts factors to interaction groups. These programs coordinate free/clamped blocks, samplers, and interactions to execute the sampling algorithm.

From a developer perspective, the core approach is to represent as much as possible as contiguous arrays/pytrees, operate on these structures, then map to and from them for the user. Internally, this is referred to as the "global" state (in opposition to the "block" state). This is similar in spirit to data-driven design (SoA) and to other JAX graphical model packages such as [PGMax](https://github.com/google-deepmind/PGMax). An important distinction from PGMax is that `THRML` supports pytree states and heterogeneous node types. Heterogeneity is handled by splitting nodes according to their pytree structure and organizing the global state as a list of these pytrees, stacked across blocks that share the same structure. The management of these indices and the mapping between block and global representations is constructed and held by the program.

Since JAX does not support ragged arrays, every block must have the same leaf array size. `THRML` handles variable block sizes by constructing the global representation via stacking and padding. There is a tradeoff between padding overhead (unnecessary computation) and the alternative of looping over blocks, which would incur an untenable compile-time cost instead.

Everything else in `THRML` exists to provide convenience for creating and working with a program. With a focused core on block index management and padding, this allows for a lightweight and hackable code base.


## What are the limitations of `THRML`?

Sampling is a fundamentally difficult problem. Generating samples from a high-dimensional distribution can require many steps even with parallelized proposals. `THRML` is also focused on Gibbs sampling, as Extropic seeks to provide hardware that accelerates this algorithm; for general sampling it is unknown when Gibbs sampling is substantially [faster](https://arxiv.org/abs/2007.08200) or [slower](https://arxiv.org/abs/1605.00139) than other MCMC methods. As a concrete example, consider a two-node Ising model with a single edge: if $J=-\infty, h=0$, Gibbs sampling will never mix between the ground states $\{-1,-1\}$ and $\{1,1\}$, whereas uniform Metropolis-Hastings would converge quickly.


## `THRML` Overviews

<img src="../flow.png" alt="A diagram which shows the flow of different components into the FactorSamplingProgram" width="400"/>

#### Factors:

- `AbstractFactor`
    - `WeightedFactor`: Parameterized by weights
    - `EBMFactor`: defines energy functions for Energy-Based Models
        - `DiscreteEBMFactor`: EBMs with discrete states (spin and categorical)
            - `SquareDiscreteEBMFactor`: Optimized for square interaction tensors
                - `SpinEBMFactor`: Spin-only interactions ({-1, 1} variables)
                - `SquareCategoricalEBMFactor`: Square categorical interactions
            - `CategoricalEBMFactor`: Categorical-only interactions  

#### Samplers:

- `AbstractConditionalSampler`
    - `AbstractParametricConditionalSampler`
        - `BernoulliConditional`: Spin-valued Bernoulli sampling
            - `SpinGibbsConditional`: Gibbs updates for spin variables in EBMs
        - `SoftmaxConditional`: Categorical softmax sampling  
            - `CategoricalGibbsConditional`: Gibbs updates for categorical variables in EBMs
