"""Event and strategy recognizers for F1 telemetry data."""

from f1_commentary.recognizers.base import EventRecognizer
from f1_commentary.recognizers.battle import BattleRecognizer
from f1_commentary.recognizers.overtake import OvertakeRecognizer
from f1_commentary.recognizers.pit_strategy import PitStrategyRecognizer
from f1_commentary.recognizers.safety_car import SafetyCarRecognizer
from f1_commentary.recognizers.tire_deg import TireDegRecognizer


def create_all_recognizers() -> list[EventRecognizer]:
    """Create and return one instance of every built-in recognizer."""
    return [
        OvertakeRecognizer(),
        BattleRecognizer(),
        PitStrategyRecognizer(),
        TireDegRecognizer(),
        SafetyCarRecognizer(),
    ]


__all__ = [
    "EventRecognizer",
    "BattleRecognizer",
    "OvertakeRecognizer",
    "PitStrategyRecognizer",
    "SafetyCarRecognizer",
    "TireDegRecognizer",
    "create_all_recognizers",
]
