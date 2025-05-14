# leduc_solver/utils.py
import os, sys
from typing import Any, List, Dict
from src.infoset import InfosetManager


def smoke_test_infoset_manager() -> None:
    """
    Smoke test for InfosetManager to ensure:
      1) New keys get unique indices
      2) Regret & strategy tables grow correctly
      3) Regret‐matching and averaging behave as intended
    """
    mgr = InfosetManager()

    # -- Test 1: Create a 3‐action infoset --
    key1: Any = ('P0', (1,), (), ())
    idx1 = mgr.get_index(key1, num_actions=3)
    assert idx1 == 0, f"Expected index 0 for first infoset, got {idx1}"

    # Initial strategy should be uniform over 3 actions
    strat1: List[float] = mgr.get_strategy(idx1)
    assert strat1 == [1/3, 1/3, 1/3], f"Uniform strat expected, got {strat1}"

    # Accumulate that strategy with weight 2.0
    mgr.accumulate_strategy(idx1, strat1, weight=2.0)

    # Give positive regret only to action 0
    mgr.update_regret(idx1, regrets=[5.0, 0.0, -1.0], weight=1.0)

    # Now regret‐matching should put all weight on action 0
    strat1_updated: List[float] = mgr.get_strategy(idx1)
    assert strat1_updated == [1.0, 0.0, 0.0], f"Expected [1,0,0], got {strat1_updated}"

    # -- Test 2: Create a 2‐action infoset --
    key2: Any = ('P1', (2,), (3,), (0,1))
    idx2 = mgr.get_index(key2, num_actions=2)
    assert idx2 == 1, f"Expected index 1 for second infoset, got {idx2}"

    # Uniform initial strategy over 2 actions
    strat2: List[float] = mgr.get_strategy(idx2)
    assert strat2 == [0.5, 0.5], f"Uniform 2-action strat expected, got {strat2}"

    # -- Test 3: Average‐strategy extraction --
    avg: Dict[Any, List[float]] = mgr.get_average_strategy()
    assert key1 in avg and key2 in avg, "Both keys should appear in average strategy"
    assert len(avg[key1]) == 3 and len(avg[key2]) == 2, "Wrong action counts in avg strat"

    print("✅ InfosetManager smoke test passed!")


if __name__ == "__main__":
    smoke_test_infoset_manager()
