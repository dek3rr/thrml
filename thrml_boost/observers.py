# Modified from the original thrml library (https://github.com/Extropic-AI/thrml)
# Changes: added global_state fast path to avoid redundant reconstruction; added dtype parameter to MomentAccumulatorObserver

import abc
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Optional, Sequence, TypeVar

import equinox as eqx
import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Int, PyTree

from thrml_boost.block_management import Block, block_state_to_global, from_global_state

if TYPE_CHECKING:
    from thrml_boost.block_sampling import _State, BlockSamplingProgram

from thrml_boost.pgm import AbstractNode

ObserveCarry = TypeVar("ObserveCarry", bound=PyTree)


class AbstractObserver(eqx.Module):
    """
    Interface for objects that inspect the sampling program while it is running.

    A concrete Observer is called once per block-sampling iteration and can maintain an
    arbitrary "carry" state across calls (e.g. running averages, histogram
    buffers, log-probs, etc.).
    """

    @abc.abstractmethod
    def __call__(
        self,
        program: "BlockSamplingProgram",
        state_free: list[PyTree[Array]],
        state_clamped: list[PyTree[Array]],
        carry: ObserveCarry,
        iteration: Int[Array, ""],
        global_state: Optional[list[PyTree[Array]]] = None,
    ) -> tuple[ObserveCarry, PyTree]:
        """Make an observation.

        This function is called at the end of a block-sampling iteration and can record information about the
        current state of the sampling program that might be useful for something later.

        **Arguments:**

        - `program`: The sampling program that is running when this function is called.
        - `state_free`: The current state of the free nodes involved in the sampling program.
        - `state_clamped`: The state of the clamped nodes involved in the sampling program.
        - `carry`: The "memory" available to this observer.
        - `iteration`: How many iterations of block sampling have happened before this function was called.
        - `global_state`: The precomputed global state as returned by `_run_blocks`. When provided
            (always the case inside `sample_with_observation`), observers use it directly to avoid
            an extra `block_state_to_global` call. When ``None`` (user code calling an observer
            directly), observers reconstruct it internally.

        **Returns:**

        A tuple, where the first element is the updated carry, and the second is a PyTree that will be
        recorded by the sampler.
        """
        return NotImplemented

    def init(self) -> PyTree:
        """Initialize the memory for the observer. Defaults to None."""
        return None


class StateObserver(AbstractObserver):
    """
    Observer which logs the raw state of some set of nodes.

    This observer is stateless: its carry is always ``None`` and ``iteration``
    is ignored.

    **Attributes:**

    - `blocks_to_sample`: the list of `Block`s which the states are logged for
    """

    blocks_to_sample: list[Block]

    def __call__(
        self,
        program: "BlockSamplingProgram",
        state_free: list["_State"],
        state_clamped: list["_State"],
        carry: None,
        iteration: Int[Array, ""],
        global_state: Optional[list[PyTree[Array]]] = None,
    ) -> tuple[None, PyTree]:
        """Simply returns the state of the blocks that are being logged to be recorded by the sampler."""
        if global_state is None:
            # Fallback when called directly from user code without a precomputed global state.
            global_state = block_state_to_global(state_free + state_clamped, program.gibbs_spec)
        sampled_state = from_global_state(global_state, program.gibbs_spec, self.blocks_to_sample)
        return None, sampled_state


def _f_identity(*x):
    return x[0]


class MomentAccumulatorObserver(AbstractObserver):
    r"""
    Observer that accumulates and updates the provided moments.

    It doesn't log any samples, and will only accumulate moments. Note that this observer does not
    scale the accumulated values by the number of times it was called. It simply records a running sum of a product
    of some state variables,

    $$\sum_i f(x_1^i) f(x_2^i) \dots f(x_N^i)$$

    **Attributes:**

    - `blocks_to_sample`: the blocks to accumulate the moments over. These
        are for constructing the final state, and aren't truly "blocks"
        in the algorithmic sense (they can be connected to each other).
        There is one block per node type.
    - `flat_nodes_list`: a list of all of the nodes in the moments (each
        occurring only once, so len(set(x)) = len(x)).
    - `flat_to_type_slices_list`: a list over node types in which each element
        is an array of indices of the `flat_node_list` which that type
        corresponds to
    - `flat_to_full_moment_slices`: a list over moment types in which each
        element is a 2D array, which matches the shape of the `moment_spec[i]`
        and of which each element is the index in the `flat_node_list`.
    - `f_transform`: the element-wise transformation $f$ to apply to sample values before
        accumulation.
    - `_flat_scatter_index`: precomputed concatenation of all `flat_to_type_slices_list`
        arrays, used to build `flat_state` in a single scatter call.
    - `_flat_scatter_sizes`: number of entries contributed by each node type, used to
        split the concatenated sampled state before scattering.
    - `_accumulate_dtype`: dtype for the accumulator, fixed at construction time.
    """

    blocks_to_sample: list[Block]
    flat_nodes_list: list[AbstractNode]
    flat_to_type_slices_list: list[Int[Array, " nodes_in_slice"]]
    flat_to_full_moment_slices: list[Int[Array, "num_groups nodes_in_moment"]]
    f_transform: Callable
    _flat_scatter_index: Array  # shape: (total_flat_nodes,)
    _flat_scatter_sizes: list[int]  # len == number of node types
    _flat_state_size: int
    _accumulate_dtype: jnp.dtype

    def __init__(
        self,
        moment_spec: Sequence[Sequence[Sequence[AbstractNode]]],
        f_transform: Callable = _f_identity,
        dtype: jnp.dtype = jnp.float32,
    ):
        r"""
        Create a MomentAccumulatorObserver.

        **Arguments:**

        - `moment_spec`: A 3 depth sequence. The first is a sequence over different moment types.
            A given moment type should have the same number of nodes in each moment. Then for each
            moment type, there is a sequence over moments. Each given moment is defined by a certain
            set of nodes.

            For example, to get the first and second moments on a simple o-o graph:

            [
                [(node1,), (node2,)],
                [(node1, node2)]
            ]

        - `f_transform`: A function that takes in (state, blocks) and returns something with the same
            structure as state. Defines a transformation $y=f(x)$ so accumulated moments are
            $\langle f(x_1) f(x_2) \rangle$.

        - `dtype`: Accumulator dtype, fixed at construction. Defaults to `jnp.float32`. Use
            `jnp.float64` for double-precision models. Fixing this here avoids a per-step cast
            inside the scan body.
        """
        self.f_transform = f_transform
        # Canonicalize the dtype immediately so that if JAX truncates float64
        # to float32 (because JAX_ENABLE_X64 is not set), the warning fires
        # once here at construction time rather than on every scan iteration.
        self._accumulate_dtype = jnp.zeros(0, dtype=dtype).dtype

        flat_nodes_list = []
        node_to_flat_idx = {}
        flat_to_full_moment_slices = []
        nodes_by_type = defaultdict(list)
        flat_to_type_slices = defaultdict(list)

        for i, moment in enumerate(moment_spec):
            shape = (len(moment), len(moment[0]))
            moment_slice = np.zeros(shape, dtype=int)

            for j, nodes in enumerate(moment):
                for k, node in enumerate(nodes):
                    idx = node_to_flat_idx.get(node, -1)
                    if idx == -1:
                        idx = len(flat_nodes_list)
                        node_to_flat_idx[node] = idx
                        flat_nodes_list.append(node)
                    moment_slice[j, k] = idx
                    nodes_by_type[node.__class__].append(node)
                    flat_to_type_slices[node.__class__].append(node_to_flat_idx[node])

            flat_to_full_moment_slices.append(moment_slice)

        blocks_to_sample = []
        flat_to_type_slices_list = []

        for node_type, nodes in nodes_by_type.items():
            blocks_to_sample.append(Block(nodes))
            type_slice = jnp.array(flat_to_type_slices[node_type], dtype=int)
            flat_to_type_slices_list.append(type_slice)

        self.flat_nodes_list = flat_nodes_list
        self.flat_to_full_moment_slices = [jnp.array(s, dtype=int) for s in flat_to_full_moment_slices]
        self.blocks_to_sample = blocks_to_sample
        self.flat_to_type_slices_list = flat_to_type_slices_list

        # Precompute a single concatenated scatter index so __call__ can build
        # flat_state with one .at[].set() instead of one per node type.
        self._flat_scatter_index = (
            jnp.concatenate(flat_to_type_slices_list) if flat_to_type_slices_list else jnp.array([], dtype=int)
        )
        self._flat_scatter_sizes = [len(s) for s in flat_to_type_slices_list]
        self._flat_state_size = len(flat_nodes_list)

    def __call__(
        self,
        program: "BlockSamplingProgram",
        state_free: list[PyTree[Array]],
        state_clamped: list[PyTree[Array]],
        carry: list[Array],
        iteration: Int[Array, ""],
        global_state: Optional[list[PyTree[Array]]] = None,
    ) -> tuple[list[Array], PyTree]:
        """Accumulate the moments via `carry`. Does not return anything for the sampler to write down."""
        if global_state is None:
            # Fallback when called directly from user code without a precomputed global state.
            global_state = block_state_to_global(state_free + state_clamped, program.gibbs_spec)

        sampled_state = from_global_state(global_state, program.gibbs_spec, self.blocks_to_sample)
        sampled_state = list(self.f_transform(sampled_state, self.blocks_to_sample))

        # Single scatter instead of one .at[].set() per node type.
        flat_values = jnp.concatenate([jnp.ravel(s) for s in sampled_state])
        flat_state = (
            jnp.zeros(self._flat_state_size, dtype=self._accumulate_dtype)
            .at[self._flat_scatter_index]
            .set(flat_values.astype(self._accumulate_dtype))
        )

        def accumulate_moment(mem_entry, sl):
            # sl: (num_groups, nodes_in_moment) — gather then product over nodes_in_moment.
            # mem_entry and update share _accumulate_dtype, so no cast needed.
            update = jnp.prod(flat_state[sl], axis=1)
            return mem_entry + update

        mem = jax.tree.map(accumulate_moment, carry, self.flat_to_full_moment_slices)
        return mem, None

    def init(self) -> list[Array]:
        """Initialize the moment accumulators with the correct dtype.

        Uses an explicit list comprehension rather than jax.tree.map over a list
        of arrays, and initialises with `_accumulate_dtype` to avoid any per-step cast.
        """
        return [jnp.zeros(x.shape[0], dtype=self._accumulate_dtype) for x in self.flat_to_full_moment_slices]
