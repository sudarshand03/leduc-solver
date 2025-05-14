# leduc_solver/game.py

import rlcard
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union


class Action(IntEnum):
    """Possible actions in Leduc Poker as defined by RLCard."""
    FOLD = 0
    CALL = 1
    RAISE = 2


class LeducGame:
    def __init__(self, seed: Optional[int] = None) -> None:
        # Make the RLCard environment
    
        self.env = rlcard.make('leduc-poker', config={'seed': seed})
        assert self.env.action_num == 3, "Expected 3 actions: fold, call/check, raise."

        self.env.set_agents([self.env.agent, self.env.agent])

    def reset(self) -> Tuple[Dict[str, Any], int]:
        """
        Start a new hand.
        """
        state, player_id = self.env.init_game()
        return state, player_id

    def step(self, action: Action) -> Tuple[Dict[str, Any], int, List[float], bool, Dict]:
        """
        Take an action in the environment.
        """
        next_state, next_player, reward, done, info = self.env.step(int(action))
        return next_state, next_player, reward, done, info

    def legal_actions(self) -> List[Action]:
        """
        Query the environment for legal actions at the current state.
        """
        legal_idxs = self.env.game.get_legal_actions()
        return [Action(idx) for idx in legal_idxs]

    def is_terminal(self) -> bool:
        """
        Check whether the current hand is over.
        """
        return self.env.is_over()

    def get_payoffs(self) -> List[float]:
        return self.env.get_payoffs()

    @staticmethod
    def infoset_key(state: Dict[str, Any], player_id: int) -> Tuple[int, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        """
        Build a hashable key representing the information set for CFR.
        Returns:
            A tuple (player_id,
                       private_cards,
                       public_cards_or_empty,
                       action_history)
        """
        obs = state['raw_obs']
        private_cards: Tuple[int, ...] = tuple(obs['hand'])
        public_cards_list: List[int] = obs.get('public_cards', [])
        public_cards: Tuple[int, ...] = tuple(public_cards_list) if public_cards_list else ()
        # action_record is a list of ints representing past actions
        history: Tuple[int, ...] = tuple(obs.get('action_record', []))
        return (player_id, private_cards, public_cards, history)

    def render(self) -> None:
        self.env.render()

