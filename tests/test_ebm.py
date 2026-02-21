"""Tests for models/ebm.py.

Covers:
- FactorizedEBM.energy gives the same result whether called with list[Block]
  or with a pre-built BlockSpec (fast path).
- The fast path avoids rebuilding BlockSpec, verified by checking results match.
"""

import unittest

import jax
import jax.numpy as jnp

from thrml.block_management import Block, BlockSpec
from thrml.block_sampling import BlockGibbsSpec
from thrml.models.discrete_ebm import SpinEBMFactor
from thrml.models.ebm import FactorizedEBM
from thrml.pgm import SpinNode


class TestEnergyBlockSpecFastPath(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        key = jax.random.key(55)

        self.n = 6
        self.nodes = [SpinNode() for _ in range(self.n)]
        self.block = Block(self.nodes)

        key, subkey = jax.random.split(key)
        biases = jax.random.uniform(subkey, (self.n,))
        key, subkey = jax.random.split(key)
        weights = jax.random.uniform(subkey, (self.n - 1,))

        bias_factor = SpinEBMFactor([self.block], biases)
        edge_factor = SpinEBMFactor(
            [Block(self.nodes[:-1]), Block(self.nodes[1:])], weights
        )

        self.ebm = FactorizedEBM([bias_factor, edge_factor])

        key, subkey = jax.random.split(key)
        self.state = [jax.random.bernoulli(subkey, 0.5, (self.n,))]

    def test_list_vs_blockspec_gives_same_energy(self):
        """energy(state, list[Block]) == energy(state, BlockSpec)."""
        node_sd = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
        blocks = [self.block]

        energy_from_list = self.ebm.energy(self.state, blocks)
        block_spec = BlockSpec(blocks, node_sd)
        energy_from_spec = self.ebm.energy(self.state, block_spec)

        self.assertTrue(jnp.allclose(energy_from_list, energy_from_spec))

    def test_gibbs_spec_passes_isinstance_check(self):
        """BlockGibbsSpec is a BlockSpec subclass, so it should hit the fast path."""
        from thrml.block_management import BlockSpec
        node_sd = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
        gibbs_spec = BlockGibbsSpec([self.block], [], node_sd)
        self.assertIsInstance(gibbs_spec, BlockSpec)

        energy_list = self.ebm.energy(self.state, [self.block])
        energy_spec = self.ebm.energy(self.state, gibbs_spec)
        self.assertTrue(jnp.allclose(energy_list, energy_spec))

    def test_energy_is_jit_compatible(self):
        """energy() should be JIT-able whether called with list or BlockSpec."""
        node_sd = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
        block_spec = BlockSpec([self.block], node_sd)

        energy_jit = jax.jit(self.ebm.energy)(self.state, block_spec)
        energy_eager = self.ebm.energy(self.state, block_spec)

        self.assertTrue(jnp.allclose(energy_jit, energy_eager))

    def test_energy_changes_with_state(self):
        """Energy should differ for different states (sanity check)."""
        node_sd = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
        block_spec = BlockSpec([self.block], node_sd)

        state_all_true = [jnp.ones(self.n, dtype=jnp.bool_)]
        state_all_false = [jnp.zeros(self.n, dtype=jnp.bool_)]

        e_true = self.ebm.energy(state_all_true, block_spec)
        e_false = self.ebm.energy(state_all_false, block_spec)

        self.assertFalse(jnp.allclose(e_true, e_false))
