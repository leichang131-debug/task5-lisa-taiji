"""Check that the local NESSAI version supports vectorised likelihood calls.

This is a lightweight interface test for the subtask 2 production route:
Triangle-BBH GPU heterodyned likelihood -> vectorised nessai.Model ->
nessai.FlowSampler.

It intentionally does not load TDC data or run nested sampling.
"""

from __future__ import annotations

import tempfile

import numpy as np
from nessai.flowsampler import FlowSampler
from nessai.model import Model


class ToyVectorisedModel(Model):
    def __init__(self) -> None:
        self.names = ["x", "y"]
        self.bounds = {"x": [-5.0, 5.0], "y": [-5.0, 5.0]}
        self.allow_vectorised = True
        self.vectorised_likelihood = True
        self.likelihood_chunksize = 16

    def log_prior(self, x):
        log_p = np.full(x.size, -np.inf)
        log_p[self.in_bounds(x)] = -np.log(10.0) - np.log(10.0)
        return log_p

    def log_likelihood(self, x):
        return -0.5 * (np.atleast_1d(x["x"]) ** 2 + np.atleast_1d(x["y"]) ** 2)


def main() -> None:
    model = ToyVectorisedModel()
    sampler = FlowSampler(
        model,
        output=tempfile.mkdtemp(prefix="nessai-vectorised-check-"),
        nlive=20,
        stopping=0.1,
        seed=1234,
        resume=False,
        likelihood_chunksize=model.likelihood_chunksize,
        pytorch_threads=1,
    )
    points = model.new_point(N=64)
    log_l = model.log_likelihood(points)
    print("FlowSampler:", type(sampler).__name__)
    print("allow_vectorised:", model.allow_vectorised)
    print("vectorised_likelihood:", model.vectorised_likelihood)
    print("likelihood_chunksize:", model.likelihood_chunksize)
    print("batch_shape:", log_l.shape)
    print("finite_fraction:", float(np.mean(np.isfinite(log_l))))


if __name__ == "__main__":
    main()
