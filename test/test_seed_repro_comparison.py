"""The reproducibility check must ignore timing and expose numerical drift."""
import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[1]/'scripts/seed_repro/compare_training.py'
_spec = importlib.util.spec_from_file_location('seed_repro_comparison', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
compare_rows = _module.compare_rows


def test_wall_clock_differences_do_not_hide_exact_training_match():
    reference = [{'time/total_timesteps': '8192', 'time/fps': '10', 'train/loss': '0.125'}]
    candidate = [{'time/total_timesteps': '8192', 'time/fps': '20', 'train/loss': '0.125'}]
    result = compare_rows(reference, candidate, 'time/total_timesteps')
    assert result['all_compared_values_equal']
    assert result['values_compared'] == 1


def test_detects_first_numeric_difference_and_ignores_unmatched_later_steps():
    reference = [{'timesteps': '8192', 'reward': '1.0'}, {'timesteps': '16384', 'reward': '1.1'}]
    candidate = [{'timesteps': '8192', 'reward': '1.0000001'}, {'timesteps': '16384', 'reward': '1.1'},
                 {'timesteps': '24576', 'reward': '1.2'}]
    result = compare_rows(reference, candidate, 'timesteps')
    assert not result['all_compared_values_equal']
    assert result['common_rows'] == 2
    assert result['different_rows'] == 1
    assert result['first_differences'][0]['step'] == 8192


def test_missing_metric_column_cannot_be_reported_as_a_complete_match():
    result = compare_rows([{'timesteps': '8192', 'reward': '1', 'error': '0'}],
                          [{'timesteps': '8192', 'reward': '1'}], 'timesteps')
    assert not result['all_compared_values_equal']
    assert result['missing_metric_columns'] == ['error']
