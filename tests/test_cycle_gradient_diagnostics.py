import math

import pytest
import torch

from diff_inversion.modeling.cycle_gradient_diagnostics import loss_gradient_geometry


def test_loss_gradient_geometry_reports_weighted_norms_and_cosine() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    loss_inversion = parameter[0].square() + parameter[1].square()
    loss_cycle = parameter[0].square() - parameter[1].square()

    metrics = loss_gradient_geometry(
        loss_inversion,
        loss_cycle,
        [parameter],
        recon_lambda=2.0,
    )

    expected_norm = math.sqrt(20.0)
    assert metrics["loss_inversion"] == 5.0
    assert metrics["loss_cycle"] == -3.0
    assert metrics["grad_norm_inversion"] == expected_norm
    assert metrics["grad_norm_cycle"] == expected_norm
    assert metrics["weighted_grad_norm_cycle"] == 2.0 * expected_norm
    assert metrics["weighted_grad_norm_ratio"] == 2.0
    assert metrics["grad_cosine"] == pytest.approx(-0.6)


def test_loss_gradient_geometry_handles_unused_parameters() -> None:
    used = torch.nn.Parameter(torch.tensor(3.0))
    unused = torch.nn.Parameter(torch.tensor(7.0))

    metrics = loss_gradient_geometry(
        used.square(),
        2.0 * used.square(),
        [used, unused],
        recon_lambda=0.5,
    )

    assert metrics["grad_norm_inversion"] == 6.0
    assert metrics["grad_norm_cycle"] == 12.0
    assert metrics["weighted_grad_norm_cycle"] == 6.0
    assert metrics["weighted_grad_norm_ratio"] == 1.0
    assert metrics["grad_cosine"] == 1.0
