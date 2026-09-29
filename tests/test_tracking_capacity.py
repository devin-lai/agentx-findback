"""Check the benchmark's queue accounting independently of GPU timing."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "tracking_capacity",
    Path(__file__).resolve().parents[1] / "scripts/eval/benchmark_tracking_capacity.py",
)
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)


def test_underloaded_queue_does_not_accumulate_lag():
    assert capacity.queue_lags([0.08, 0.1, 0.09], 5) == pytest.approx([0.08, 0.1, 0.09])


def test_overloaded_queue_includes_waiting_and_service_time():
    assert capacity.queue_lags([0.3, 0.3, 0.3], 5) == pytest.approx([0.3, 0.4, 0.5])


def test_queue_recovers_after_a_slow_frame():
    assert capacity.queue_lags([0.5, 0.05, 0.05, 0.05], 5) == pytest.approx([0.5, 0.35, 0.2, 0.05])


def test_image_input_is_identical_across_inventory_sizes():
    one, one_object = capacity.fixture(1)
    ten, ten_objects = capacity.fixture(10)
    assert (one == ten).all()
    assert len(one_object) == 1 and len(ten_objects) == 10
    assert one_object == ten_objects[:1]
