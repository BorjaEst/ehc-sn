from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from ehc_sn.controllers.tem import TEMController
from ehc_sn.modules.mec import MECModel, MECSettings
from ehc_sn.modules.mec.ovc import OVCSettings
from ehc_sn.modules.mec.path import PathIntegrator, PathSettings
from ehc_sn.types import LocationBelief


def test_extract_step_data_adds_episode_start_without_rewriting_actions():
    env_td = TensorDict(
        {
            "inputs": torch.zeros((2, 3), dtype=torch.float32),
            "observation_target": torch.zeros((2, 1), dtype=torch.int64),
            "previous_action": torch.tensor([[0], [2]], dtype=torch.int64),
            "location_id": torch.tensor([[1], [2]], dtype=torch.int64),
            "region_id": torch.tensor([[0], [0]], dtype=torch.int64),
            "landmark_id": torch.tensor([[0], [0]], dtype=torch.int64),
            "valid_action_mask": torch.ones((2, 5), dtype=torch.bool),
            "step_count": torch.tensor([[0], [3]], dtype=torch.int32),
        },
        batch_size=[2],
    )

    payload = TEMController._extract_step_data(object(), env_td)

    assert torch.equal(payload["previous_action"], env_td["previous_action"])
    assert torch.equal(payload["episode_start"], torch.tensor([True, False]))


def test_mec_generative_preserves_episode_start_rows():
    model = MECModel(
        action_count=3,
        n_hippocampal=[2],
        f_initial=[0.9],
        config=MECSettings(grid_shape=[2], ovc=OVCSettings(mode="off")),
    )
    state = model.init_state(batch_size=2)
    original_cells = [cell.clone() for cell in state.cells]

    def fake_forward(action_ids, g_prev, no_direc_mask=None):
        del action_ids, no_direc_mask
        shifted_mean = [g_f + 10.0 for g_f in g_prev]
        shifted_uncertainty = [torch.ones_like(g_f) for g_f in g_prev]
        return LocationBelief(mean=shifted_mean, uncertainty=shifted_uncertainty)

    model.path_integration.forward = fake_forward

    g_gen, next_state = model.generative(
        action=torch.tensor([[0], [1]], dtype=torch.int64),
        episode_start=torch.tensor([True, False]),
        landmark_id=None,
        state=state,
    )

    assert torch.equal(g_gen[0][0], original_cells[0][0])
    assert torch.equal(next_state.cells[0][0], original_cells[0][0])
    assert torch.equal(g_gen[0][1], torch.ones_like(original_cells[0][1]))
    assert torch.equal(next_state.cells[0][1], original_cells[0][1] + 10.0)


def test_path_integrator_rejects_invalid_action_ids():
    integrator = PathIntegrator(n_actions=3, mec_shape=[2], f_initial=[0.9], config=PathSettings())

    with pytest.raises(ValueError, match=r"Action ids must be in \[0, 3\)"):
        integrator(torch.tensor([[3]], dtype=torch.int64), [torch.zeros((1, 2), dtype=torch.float32)])