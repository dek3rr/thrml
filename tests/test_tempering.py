"""Tests for parallel_tempering.py.

Covers:
- Basic smoke test (output shapes, acceptance with zero energy)
- Three-chain smoke to verify all pairs are counted
- Output format (state lists correctly unstacked, sampler_states format)
- vmap correctness: a single-chain PT run should match _run_blocks directly
- Acceptance statistics accumulate correctly across rounds
"""

import jax
import jax.numpy as jnp
import pytest

from thrml import Block, SpinNode, make_empty_block_state
from thrml.block_management import BlockSpec
from thrml.block_sampling import SamplingSchedule, _run_blocks, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram
from thrml.tempering import parallel_tempering


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _tiny_ising(n_temps: int = 2, betas=None):
    """2×2 torus Ising, two-coloured free blocks, zero weights so all
    configurations are equally likely (uniform distribution)."""
    grid = [[SpinNode() for _ in range(2)] for _ in range(2)]
    nodes = [n for row in grid for n in row]
    edges = []
    for i in range(2):
        for j in range(2):
            n = grid[i][j]
            edges.append((n, grid[i][(j + 1) % 2]))
            edges.append((n, grid[(i + 1) % 2][j]))
    even_nodes = [grid[i][j] for i in range(2) for j in range(2) if (i + j) % 2 == 0]
    odd_nodes = [grid[i][j] for i in range(2) for j in range(2) if (i + j) % 2 == 1]
    free_blocks = [Block(even_nodes), Block(odd_nodes)]

    biases = jnp.zeros(len(nodes))
    weights = jnp.zeros(len(edges))

    if betas is None:
        betas = [float(t) / n_temps for t in range(1, n_temps + 1)]

    ebms = [IsingEBM(nodes, edges, biases, weights, jnp.array(b)) for b in betas]
    programs = [IsingSamplingProgram(e, free_blocks, clamped_blocks=[]) for e in ebms]
    init_state = make_empty_block_state(free_blocks, ebms[0].node_shape_dtypes)

    return nodes, edges, free_blocks, ebms, programs, init_state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_parallel_tempering_smoke():
    """Two chains, zero energy — every swap must be accepted."""
    _, _, free_blocks, ebms, programs, init_state = _tiny_ising(n_temps=2)
    init_states = [init_state, init_state]

    @jax.jit
    def run(key):
        return parallel_tempering(
            key, ebms, programs, init_states, clamp_state=[],
            n_rounds=2, gibbs_steps_per_round=1,
        )

    final_states, sampler_states, stats = run(jax.random.key(0))

    assert len(final_states) == 2
    assert len(sampler_states) == 2
    assert stats["accepted"].shape == (1,)
    assert stats["attempted"].shape == (1,)
    assert stats["acceptance_rate"].shape == (1,)
    assert stats["accepted"][0] == 1
    assert stats["attempted"][0] == 1
    assert stats["acceptance_rate"][0] == 1.0


def test_three_chains_smoke():
    """Three chains → two adjacent pairs.

    With zero weights all swaps are accepted. Even rounds try pair 0 (chains
    0–1), odd rounds try pair 1 (chains 1–2). With n_rounds=4 we get two even
    and two odd rounds, so both pairs are attempted and accepted twice each.
    """
    _, _, free_blocks, ebms, programs, init_state = _tiny_ising(n_temps=3)
    init_states = [init_state] * 3

    @jax.jit
    def run(key):
        return parallel_tempering(
            key, ebms, programs, init_states, clamp_state=[],
            n_rounds=4, gibbs_steps_per_round=1,
        )

    _, _, stats = run(jax.random.key(7))

    assert stats["accepted"].shape == (2,)
    assert stats["attempted"].shape == (2,)
    # All swaps accepted (zero energy → log_r = 0 → accept_prob = 1)
    assert jnp.all(stats["accepted"] == stats["attempted"])
    assert jnp.all(stats["acceptance_rate"] == 1.0)
    # 4 rounds: even (0,2) try pair 0; odd (1,3) try pair 1 → each attempted 2x
    assert stats["attempted"][0] == 2
    assert stats["attempted"][1] == 2


def test_output_state_format():
    """Output states should have the same shapes as the inputs."""
    _, _, free_blocks, ebms, programs, init_state = _tiny_ising(n_temps=2)
    init_states = [init_state, init_state]

    final_states, sampler_states, stats = jax.jit(parallel_tempering)(
        jax.random.key(1), ebms, programs, init_states, [],
        n_rounds=2, gibbs_steps_per_round=2,
    )

    assert len(final_states) == 2
    for chain_state in final_states:
        assert len(chain_state) == len(free_blocks)
        for block_state, block in zip(chain_state, free_blocks):
            assert block_state.shape == (len(block),)
            assert block_state.dtype == jnp.bool_

    # All built-in samplers return None; sampler_states should mirror that
    assert len(sampler_states) == 2
    for chain_ss in sampler_states:
        assert len(chain_ss) == len(free_blocks)
        for ss in chain_ss:
            assert ss is None


def test_four_chains_pair_counting():
    """Four chains → three adjacent pairs.

    Even rounds: pairs 0 and 2. Odd rounds: pair 1.
    n_rounds=6 → 3 even, 3 odd → pairs 0,2 attempted 3x; pair 1 attempted 3x.
    """
    _, _, free_blocks, ebms, programs, init_state = _tiny_ising(n_temps=4)
    init_states = [init_state] * 4

    @jax.jit
    def run(key):
        return parallel_tempering(
            key, ebms, programs, init_states, [],
            n_rounds=6, gibbs_steps_per_round=1,
        )

    _, _, stats = run(jax.random.key(42))

    assert stats["attempted"].shape == (3,)
    assert stats["attempted"][0] == 3  # even pair
    assert stats["attempted"][1] == 3  # odd pair
    assert stats["attempted"][2] == 3  # even pair
    assert jnp.all(stats["acceptance_rate"] == 1.0)


def test_zero_rounds():
    """n_rounds=0 should return the initial states unchanged and zero stats."""
    _, _, free_blocks, ebms, programs, init_state = _tiny_ising(n_temps=2)
    init_a = [jnp.ones((len(b),), dtype=jnp.bool_) for b in free_blocks]
    init_b = [jnp.zeros((len(b),), dtype=jnp.bool_) for b in free_blocks]

    final_states, _, stats = parallel_tempering(
        jax.random.key(0), ebms, programs, [init_a, init_b], [],
        n_rounds=0, gibbs_steps_per_round=1,
    )

    # States should be identical to inputs
    for b in range(len(free_blocks)):
        assert jnp.array_equal(final_states[0][b], init_a[b])
        assert jnp.array_equal(final_states[1][b], init_b[b])

    assert stats["accepted"][0] == 0
    assert stats["attempted"][0] == 0


def test_single_chain_matches_run_blocks():
    """A single-chain PT run should give the same final state as _run_blocks
    with the same key (after accounting for the internal key splitting).

    We run PT with one chain for one round and no swaps, then reproduce the
    same computation with _run_blocks using the matching key.
    """
    _, _, free_blocks, ebms, programs, init_state = _tiny_ising(n_temps=1)
    init_states = [init_state]

    key = jax.random.key(999)

    final_pt, _, _ = jax.jit(parallel_tempering)(
        key, [ebms[0]], [programs[0]], init_states, [],
        n_rounds=1, gibbs_steps_per_round=3,
    )

    # Reproduce the key splitting that parallel_tempering does internally.
    # one_round: key, key_round = split(key); gibbs_keys = split(key_round, 1)
    key_after, key_round = jax.random.split(key)
    gibbs_key = jax.random.split(key_round, 1)[0]

    ss = [None] * len(free_blocks)
    final_rb, _, _ = jax.jit(_run_blocks)(
        gibbs_key, programs[0], init_state, [], 3, ss,
    )

    for b in range(len(free_blocks)):
        assert jnp.array_equal(final_pt[0][b], final_rb[b]), (
            f"Block {b} mismatch: PT={final_pt[0][b]} vs _run_blocks={final_rb[b]}"
        )


def test_mismatched_programs_raises():
    """Programs with different block structures should raise ValueError."""
    _, _, free_blocks, ebms, programs, init_state = _tiny_ising(n_temps=2)

    # Build a program with a different number of free blocks
    nodes_alt = [SpinNode() for _ in range(4)]
    alt_block = [Block(nodes_alt)]
    biases_alt = jnp.zeros(4)
    weights_alt = jnp.zeros(0)
    ebm_alt = IsingEBM(nodes_alt, [], biases_alt, weights_alt, jnp.array(1.0))
    prog_alt = IsingSamplingProgram(ebm_alt, alt_block, clamped_blocks=[])

    with pytest.raises(ValueError, match="block structure"):
        parallel_tempering(
            jax.random.key(0), [ebms[0], ebm_alt], [programs[0], prog_alt],
            [init_state, [jnp.zeros((4,), jnp.bool_)]],
            [], n_rounds=1, gibbs_steps_per_round=1,
        )


def test_nonzero_energy_acceptance():
    """With very high beta the model strongly prefers ground state.

    Even if we start from opposite ends of the configuration space, the low-
    temperature chain should reject most swaps with the high-temperature chain
    because the energy difference is large.

    This is a statistical test — just check that acceptance_rate is < 1.
    """
    nodes = [SpinNode() for _ in range(4)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(3)]
    biases = jnp.zeros(4)
    # Strong ferromagnetic coupling
    weights = jnp.ones(3) * 2.0

    ebm_cold = IsingEBM(nodes, edges, biases, weights, jnp.array(5.0))   # low T
    ebm_hot = IsingEBM(nodes, edges, biases, weights, jnp.array(0.1))    # high T

    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    programs = [
        IsingSamplingProgram(ebm_cold, free_blocks, []),
        IsingSamplingProgram(ebm_hot, free_blocks, []),
    ]

    # Cold chain starts at all-True (low energy), hot chain all-False
    init_cold = [jnp.ones((len(b),), jnp.bool_) for b in free_blocks]
    init_hot = [jnp.zeros((len(b),), jnp.bool_) for b in free_blocks]

    _, _, stats = jax.jit(parallel_tempering)(
        jax.random.key(123),
        [ebm_cold, ebm_hot],
        programs,
        [init_cold, init_hot],
        clamp_state=[],
        n_rounds=20,
        gibbs_steps_per_round=1,
    )

    # With a cold chain at beta=5 and ferromagnetic coupling, swapping a very
    # low-energy state into the hot chain and vice versa is usually rejected
    assert float(stats["acceptance_rate"][0]) < 1.0
