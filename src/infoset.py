# leduc_solver/src/infoset.py

from typing import Any, Dict, List, Tuple

class InfosetManager:
    """
    Manages the mapping from infoset keys to
    regret and strategy tables for CFR.
    """
    def __init__(self) -> None:
        # infoset_key → unique integer index
        self._key_to_index: Dict[Any, int] = {}
        # per-infoset cumulative regrets: list of lists, one inner list per infoset
        self._regret_sum: List[List[float]] = []
        # per-infoset cumulative strategy weights: likewise
        self._strategy_sum: List[List[float]] = []

    def _init_infoset(self, num_actions: int) -> int:
        """
        Allocate a new infoset slot

        Returns:
            The newly assigned infoset index.
        """
        idx = len(self._regret_sum)
        # initialize regrets and strategy sums to zero
        self._regret_sum.append([0.0] * num_actions)
        self._strategy_sum.append([0.0] * num_actions)
        return idx

    def get_index(self, key: Any, num_actions: int) -> int:
        """
        Returns:
            Integer index into regret/strategy arrays.
        """
        if key not in self._key_to_index:
            self._key_to_index[key] = self._init_infoset(num_actions)
        return self._key_to_index[key]

    def get_strategy(self, idx: int) -> List[float]:
        """
        Compute the current regret-matching strategy for infoset idx.
        Returns:
            A probability distribution over actions (sums to 1).
        """
        regrets = self._regret_sum[idx]
        # only positive regrets
        positive = [r if r > 0 else 0.0 for r in regrets]
        total_pos = sum(positive)
        if total_pos > 0:
            return [r / total_pos for r in positive]
        # fallback to uniform if all regrets ≤ 0
        num_actions = len(regrets)
        return [1.0 / num_actions] * num_actions

    def accumulate_strategy(self, idx: int, strategy: List[float], weight: float) -> None:
        """
        Add weighted strategy to the cumulative sum for averaging.
        Args:
            idx: infoset index.
            strategy: the current mixed strategy over actions.
            weight: the reach probability product for this iteration.
        """
        for a, prob in enumerate(strategy):
            self._strategy_sum[idx][a] += weight * prob

    def update_regret(self, idx: int, regrets: List[float], weight: float) -> None:
        """
        Add weighted instantaneous regrets to the cumulative regrets.
        Args:
            idx: infoset index.
            regrets: instantaneous regret for each action
            weight: the opponent reach-probability
        """
        for a, r in enumerate(regrets):
            self._regret_sum[idx][a] += weight * r

    def get_average_strategy(self) -> Dict[Any, List[float]]:
        """
        After training, extract the average strategy for each infoset
        Returns:
            Mapping from infoset key → averaged action distribution
        """
        avg_strat: Dict[Any, List[float]] = {}
        for key, idx in self._key_to_index.items():
            cum_strat = self._strategy_sum[idx]
            total = sum(cum_strat)
            if total > 0:
                avg_strat[key] = [s / total for s in cum_strat]
            else:
                # uniform if never visited
                num_actions = len(cum_strat)
                avg_strat[key] = [1.0 / num_actions] * num_actions
        return avg_strat
