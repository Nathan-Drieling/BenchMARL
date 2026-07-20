#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from benchmarl.algorithms import algorithm_config_registry
from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.hydra_config import load_algorithm_config_from_hydra
from hydra import compose, initialize


@pytest.mark.parametrize("algo_name", algorithm_config_registry.keys())
def test_loading_algorithms(algo_name):
    with initialize(version_base=None, config_path="../benchmarl/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"algorithm={algo_name}",
                "task=vmas/balance",
            ],
        )
        algo_config: AlgorithmConfig = load_algorithm_config_from_hydra(cfg.algorithm)
        assert algo_config == algorithm_config_registry[algo_name].get_from_yaml()


class _DummyAlgorithm(Algorithm):
    def _get_loss(self, group, policy_for_loss, continuous):
        raise NotImplementedError

    def _get_parameters(self, group, loss):
        return {}

    def _get_policy_for_loss(self, group, model_config, continuous):
        raise NotImplementedError

    def _get_policy_for_collection(self, policy_for_loss, group, continuous):
        raise NotImplementedError

    def process_batch(self, group, batch):
        return batch

    def _check_specs(self):
        return None


def test_action_modifier_zeroes_requested_dimensions():
    experiment = SimpleNamespace(
        config=SimpleNamespace(
            train_device="cpu",
            buffer_device="cpu",
            soft_target_update=True,
            polyak_tau=0.01,
            hard_target_update_frequency=1,
            replay_buffer_memory_size=lambda *_args, **_kwargs: 1,
            train_minibatch_size=lambda *_args, **_kwargs: 1,
            collected_frames_per_batch=lambda *_args, **_kwargs: 1,
            n_envs_per_worker=lambda *_args, **_kwargs: 1,
            off_policy_use_prioritized_replay_buffer=False,
            off_policy_prb_alpha=0.6,
            off_policy_prb_beta=0.4,
        ),
        model_config=SimpleNamespace(is_rnn=False),
        critic_model_config=SimpleNamespace(is_rnn=False),
        on_policy=False,
        group_map={"agent": ["agent"]},
        observation_spec=None,
        action_spec={},
        state_spec=None,
        action_mask_spec=None,
        algorithm_config=SimpleNamespace(
            has_independent_critic=lambda: False,
            has_centralized_critic=lambda: False,
            has_critic=False,
        ),
    )

    algorithm = _DummyAlgorithm(experiment)
    modifier = algorithm._get_action_modifier(
        "agent",
        hidden_action_dimensions=(1,),
    )

    action = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    tensordict = TensorDict({("agent", "action"): action})

    modified = modifier(tensordict)
    expected = torch.tensor([[1.0, 0.0, 3.0]], dtype=torch.float32)

    assert torch.equal(modified[("agent", "action")], expected)
