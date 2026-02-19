"""
Parallel tempering utilities built on THRML's block Gibbs samplers.

This module orchestrates multiple tempered chains (one per beta/model), runs
blocked Gibbs steps on each, and proposes swaps between adjacent temperatures.

Key design: all chains share the same *structural* program data (block spec,
active flags, gather/scatter indices, samplers). Only `per_block_interactions`
differs across chains, because it carries the beta-scaled weights. We exploit
this by vmapping a single-chain runner over (key, state, per_block_interactions),
replacing the Python for-loop over chains that previously unrolled n_chains
copies of the full Gibbs graph into the XLA computation.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
from jax import lax

from thrml.block_sampling import BlockSamplingProgram, _run_blocks
from thrml.models.ebm import AbstractEBM


def _init_sampler_states(program: BlockSamplingProgram) -> list:
    """Initialize sampler state list for a BlockSamplingProgram."""
    return [s.init() for s in program.samplers]


def _attempt_swap_pair(
    key,
    ebm_i: AbstractEBM,
    ebm_j: AbstractEBM,
    program_i: BlockSamplingProgram,
    program_j: BlockSamplingProgram,
    state_i: list,
    state_j: list,
    clamp_state: list,
):
    """
    Propose a swap between two adjacent temperature chains (i, j).

    Acceptance ratio uses energies of both states under both models:
    log r = (E_i(x_i) + E_j(x_j)) - (E_i(x_j) + E_j(x_i))
    """
    # Pass the pre-built BlockSpec directly so energy() skips rebuilding it.
    spec_i = program_i.gibbs_spec
    spec_j = program_j.gibbs_spec

    Ei_xi = ebm_i.energy(state_i + clamp_state, spec_i)
    Ej_xj = ebm_j.energy(state_j + clamp_state, spec_j)
    Ei_xj = ebm_i.energy(state_j + clamp_state, spec_i)
    Ej_xi = ebm_j.energy(state_i + clamp_state, spec_j)

    log_r = (Ei_xi + Ej_xj) - (Ei_xj + Ej_xi)
    accept_prob = jnp.exp(jnp.minimum(0.0, log_r))
    u = jax.random.uniform(key)

    def do_swap(states):
        s_i, s_j = states
        return s_j, s_i, jnp.int32(1)

    def no_swap(states):
        s_i, s_j = states
        return s_i, s_j, jnp.int32(0)

    return lax.cond(u < accept_prob, do_swap, no_swap, (state_i, state_j))


def _swap_pass_stacked(
    key,
    ebms: Sequence[AbstractEBM],
    programs: Sequence[BlockSamplingProgram],
    stacked_states: list,
    stacked_ss: list,
    all_ss_none: bool,
    clamp_state: list,
    pair_indices: Sequence[int],
    n_free_blocks: int,
):
    """Perform one swap pass over a fixed set of adjacent pairs.

    Operates on the chain-stacked state representation:
    - ``stacked_states[b]`` has shape ``(n_chains, n_nodes_b, *node_shape)``.
    - ``stacked_ss[b]`` has shape ``(n_chains, ...)`` or is ``None``.

    For each pair (i, j) in ``pair_indices``, single-chain states are extracted
    via static int indexing, passed to ``_attempt_swap_pair`` unchanged, then
    scattered back with ``.at[i/j].set()``.
    """
    n_pairs = len(ebms) - 1
    zeros = jnp.zeros(n_pairs, dtype=jnp.int32)

    if len(pair_indices) == 0:
        return stacked_states, stacked_ss, zeros, zeros

    keys = jax.random.split(key, len(pair_indices))
    accept_counts = zeros
    attempt_counts = zeros

    for idx, pair in enumerate(pair_indices):
        i, j = pair, pair + 1

        # Extract single-chain states for the energy computation.
        # Static int indexing into a stacked array — compiles to a gather slice.
        state_i = [stacked_states[b][i] for b in range(n_free_blocks)]
        state_j = [stacked_states[b][j] for b in range(n_free_blocks)]

        new_i, new_j, acc = _attempt_swap_pair(
            keys[idx], ebms[i], ebms[j], programs[i], programs[j],
            state_i, state_j, clamp_state,
        )

        # Scatter updated states back into the stacked arrays.
        # Both .at[i] and .at[j] use static Python ints, so XLA lowers these
        # to a pair of scatter ops rather than a general dynamic scatter.
        stacked_states = [
            stacked_states[b].at[i].set(new_i[b]).at[j].set(new_j[b])
            for b in range(n_free_blocks)
        ]

        if not all_ss_none:
            # Swap sampler-state rows in the same way.
            stacked_ss = [
                stacked_ss[b].at[i].set(stacked_ss[b][j]).at[j].set(stacked_ss[b][i])
                for b in range(n_free_blocks)
            ]

        attempt_counts = attempt_counts.at[pair].set(1)
        accept_counts = accept_counts.at[pair].set(acc)

    return stacked_states, stacked_ss, accept_counts, attempt_counts


def parallel_tempering(
    key,
    ebms: Sequence[AbstractEBM],
    programs: Sequence[BlockSamplingProgram],
    init_states: Sequence[list],
    clamp_state: list,
    n_rounds: int,
    gibbs_steps_per_round: int,
    sampler_states: Sequence[list] | None = None,
):
    """Run parallel tempering across a sequence of tempered chains.

    Each round performs block Gibbs updates in every chain in parallel via
    ``jax.vmap``, then proposes swaps between adjacent temperatures using
    alternating even/odd pair selection.

    All chains must share the same block structure (free + clamped) and clamped
    state. Only the EBM weights (encoded in ``per_block_interactions``) may
    differ across chains. This is the standard parallel tempering setup where
    chains differ only by their inverse temperature beta.

    **Arguments:**

    - `key`: JAX PRNG key.
    - `ebms`: One EBM per temperature, ordered from lowest to highest beta.
    - `programs`: One ``BlockSamplingProgram`` per temperature. All must share
        the same block structure; only ``per_block_interactions`` may differ.
    - `init_states`: Initial free-block states, one list per chain.
    - `clamp_state`: Clamped-block state, shared across all chains.
    - `n_rounds`: Number of Gibbs + swap rounds to run.
    - `gibbs_steps_per_round`: Gibbs sweeps per chain per round.
    - `sampler_states`: Optional initial sampler states; inferred from programs
        if not provided.

    **Returns:**

    Tuple ``(states, sampler_states, stats)`` where ``stats`` is a dict with
    keys ``accepted``, ``attempted``, and ``acceptance_rate``, each an array
    of length ``n_chains - 1`` indexed by adjacent pair.
    """
    if not (len(ebms) == len(programs) == len(init_states)):
        raise ValueError("ebms, programs, and init_states must have the same length.")
    if sampler_states is not None and len(sampler_states) != len(programs):
        raise ValueError("sampler_states must match length of programs if provided.")

    base_spec = programs[0].gibbs_spec
    base_free = len(base_spec.free_blocks)
    base_clamped = len(base_spec.clamped_blocks)
    for prog in programs[1:]:
        if (len(prog.gibbs_spec.free_blocks) != base_free
                or len(prog.gibbs_spec.clamped_blocks) != base_clamped):
            raise ValueError("All programs must share the same block structure (free + clamped blocks).")

    clamp_state = clamp_state or []
    n_chains = len(ebms)
    n_free_blocks = base_free

    states = [list(s) for s in init_states]
    sampler_states = (
        [list(s) for s in sampler_states]
        if sampler_states is not None
        else [_init_sampler_states(p) for p in programs]
    )

    # Detect whether all sampler states are None (covers all built-in samplers:
    # BernoulliConditional, SoftmaxConditional, SpinGibbsConditional).
    # We handle this at Python level so we can exclude None from vmap arguments
    # — jax.vmap cannot batch over None (it has no array axis to strip).
    # If your sampler has real state, add it to the stacked vmap inputs below.
    all_ss_none = all(
        ss is None
        for chain_ss in sampler_states
        for ss in chain_ss
    )

    # -------------------------------------------------------------------------
    # Convert to chain-stacked representation
    # -------------------------------------------------------------------------
    # stacked_states[b]: shape (n_chains, n_nodes_b, *node_shape)
    # All blocks for the same node type are concatenated along axis 0 inside
    # the global state; the stacked format adds a leading chain axis on top of
    # the per-block arrays.
    stacked_states = [
        jnp.stack([states[c][b] for c in range(n_chains)], axis=0)
        for b in range(n_free_blocks)
    ]

    # stacked_ss[b]: shape (n_chains, ...) or None
    if all_ss_none:
        stacked_ss = [None] * n_free_blocks
    else:
        stacked_ss = [
            jax.tree.map(
                lambda *xs: jnp.stack(xs, axis=0),
                *[sampler_states[c][b] for c in range(n_chains)],
            )
            for b in range(n_free_blocks)
        ]

    # stacked_pbi[b][g]: same PyTree as programs[0].per_block_interactions[b][g]
    # but with a leading chain axis on every array leaf.
    # Under jax.vmap the chain axis is stripped, so each vmapped call sees
    # exactly the single-chain interaction weights it expects.
    stacked_pbi = [
        [
            jax.tree.map(
                lambda *xs: jnp.stack(xs, axis=0),
                *[programs[c].per_block_interactions[b][g] for c in range(n_chains)],
            )
            for g in range(len(programs[0].per_block_interactions[b]))
        ]
        for b in range(n_free_blocks)
    ]

    n_pairs = max(n_chains - 1, 0)
    accepted = jnp.zeros(n_pairs, dtype=jnp.int32)
    attempted = jnp.zeros(n_pairs, dtype=jnp.int32)

    even_pair_indices = list(range(0, n_pairs, 2))
    odd_pair_indices = list(range(1, n_pairs, 2))

    # -------------------------------------------------------------------------
    # Build vmapped Gibbs runner
    # -------------------------------------------------------------------------
    # We close over:
    #   programs[0]       — template for all static structural data
    #   clamp_state       — identical across chains
    #   gibbs_steps_per_round — scalar
    #   stacked_pbi       — constant JAX arrays (captured in the closure)
    #
    # We vmap over:
    #   gibbs_key         — one key per chain
    #   state_free        — one block list per chain (stacked_states)
    #   pbi               — one interaction weight set per chain (stacked_pbi)
    #
    # Sampler states are None for all built-in samplers, so we close over
    # [None] * n_free_blocks rather than passing them as vmapped arguments.
    # If you add a stateful sampler, stack its states and add them here.
    if all_ss_none:
        null_ss = [None] * n_free_blocks

        def _run_one_chain(gibbs_key, state_free, pbi):
            """Run one chain for gibbs_steps_per_round steps."""
            new_state, _, _ = _run_blocks(
                gibbs_key, programs[0], state_free, clamp_state,
                gibbs_steps_per_round, null_ss,
                per_block_interactions=pbi,
            )
            return new_state

        _run_all_chains = jax.vmap(_run_one_chain)

    else:
        def _run_one_chain_with_ss(gibbs_key, state_free, pbi, ss):
            """Run one chain with real sampler state."""
            new_state, new_ss, _ = _run_blocks(
                gibbs_key, programs[0], state_free, clamp_state,
                gibbs_steps_per_round, ss,
                per_block_interactions=pbi,
            )
            return new_state, new_ss

        _run_all_chains = jax.vmap(_run_one_chain_with_ss)

    # -------------------------------------------------------------------------
    # Scan over rounds
    # -------------------------------------------------------------------------
    def one_round(carry, round_idx):
        key, stacked_states, stacked_ss, accepted, attempted = carry

        # One key per chain for Gibbs, one for swaps.
        key, key_round = jax.random.split(key)
        gibbs_keys = jax.random.split(key_round, n_chains)
        key, swap_key = jax.random.split(key)

        # --- Gibbs updates: all chains simultaneously via vmap ---
        # stacked_pbi is a constant captured in the closure above; it does not
        # change across rounds, so it is not part of the carry.
        if all_ss_none:
            stacked_states = _run_all_chains(gibbs_keys, stacked_states, stacked_pbi)
        else:
            stacked_states, stacked_ss = _run_all_chains(
                gibbs_keys, stacked_states, stacked_pbi, stacked_ss
            )

        # --- Swap step: alternating even/odd pairs via lax.cond ---
        # Both branches (do_even, do_odd) are traced at compile time to establish
        # output shapes; lax.cond selects between them at runtime based on parity.
        # The Python for-loops inside _swap_pass_stacked execute at trace time,
        # producing a fixed XLA scatter pattern for each parity.
        def do_even(args):
            s_states, s_ss, acc, att, s_key = args
            s_states, s_ss, acc_counts, att_counts = _swap_pass_stacked(
                s_key, ebms, programs, s_states, s_ss,
                all_ss_none, clamp_state, even_pair_indices, n_free_blocks,
            )
            return s_states, s_ss, acc + acc_counts, att + att_counts

        def do_odd(args):
            s_states, s_ss, acc, att, s_key = args
            s_states, s_ss, acc_counts, att_counts = _swap_pass_stacked(
                s_key, ebms, programs, s_states, s_ss,
                all_ss_none, clamp_state, odd_pair_indices, n_free_blocks,
            )
            return s_states, s_ss, acc + acc_counts, att + att_counts

        stacked_states, stacked_ss, accepted, attempted = lax.cond(
            (round_idx & 1) == 0,
            do_even,
            do_odd,
            (stacked_states, stacked_ss, accepted, attempted, swap_key),
        )

        return (key, stacked_states, stacked_ss, accepted, attempted), None

    if n_rounds > 0:
        init_carry = (key, stacked_states, stacked_ss, accepted, attempted)
        (key, stacked_states, stacked_ss, accepted, attempted), _ = lax.scan(
            one_round, init_carry, jnp.arange(n_rounds)
        )

    # -------------------------------------------------------------------------
    # Unstack back to list[list[Array]] format: states[chain][block]
    # -------------------------------------------------------------------------
    states = [
        [stacked_states[b][c] for b in range(n_free_blocks)]
        for c in range(n_chains)
    ]

    if all_ss_none:
        sampler_states = [[None] * n_free_blocks for _ in range(n_chains)]
    else:
        sampler_states = [
            [stacked_ss[b][c] for b in range(n_free_blocks)]
            for c in range(n_chains)
        ]

    acceptance_rate = jnp.where(attempted > 0, accepted / attempted, 0.0)
    stats = {
        "accepted": accepted,
        "attempted": attempted,
        "acceptance_rate": acceptance_rate,
    }

    return states, sampler_states, stats
