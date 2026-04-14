"""Battle recognizer for detecting active on-track battles."""

from datetime import datetime, timezone

from f1_commentary.recognizers.base import EventRecognizer
from f1_commentary.schemas.events import (
    CandidateEvent,
    CandidateEventType,
    DriverState,
    RaceStateSnapshot,
)


class BattleRecognizer(EventRecognizer):
    """Detect active battles between groups of drivers."""

    BATTLE_GAP_THRESHOLD = 1.5  # seconds

    @property
    def name(self) -> str:
        return "BattleRecognizer"

    def recognize(self, snapshot: RaceStateSnapshot) -> list[CandidateEvent]:
        events: list[CandidateEvent] = []

        # Use battle_pairs from snapshot if available, else compute
        if snapshot.battle_pairs:
            groups = self._groups_from_pairs(snapshot.battle_pairs)
        else:
            groups = self._compute_battle_groups(snapshot.drivers)

        for group in groups:
            if len(group) < 2:
                continue

            # Determine positions of involved drivers
            positions = self._get_positions(snapshot, group)
            min_pos = min(positions) if positions else 99

            # Classify battle type
            if min_pos <= 3 and max(positions) <= 3:
                candidate_type = CandidateEventType.LEAD_BATTLE
                severity = 0.9
            elif min_pos <= 3:
                candidate_type = CandidateEventType.LEAD_BATTLE
                severity = 0.8
            else:
                candidate_type = CandidateEventType.MIDFIELD_BATTLE
                severity = 0.5

            confidence = round(min(1.0, 0.5 + len(group) * 0.15), 3)

            events.append(
                CandidateEvent(
                    timestamp=datetime.now(tz=timezone.utc),
                    candidate_type=candidate_type,
                    involved_drivers=sorted(group),
                    confidence=confidence,
                    severity=severity,
                    explanation=(
                        f"{'Lead' if candidate_type == CandidateEventType.LEAD_BATTLE else 'Midfield'} "
                        f"battle between {', '.join(sorted(group))} "
                        f"(positions {sorted(positions)})"
                    ),
                    triggering_features={
                        "positions": sorted(positions),
                        "driver_count": len(group),
                    },
                    lap=snapshot.lap,
                    session_key=snapshot.session_key,
                )
            )

            # Also emit a generic ACTIVE_BATTLE for every battle group
            events.append(
                CandidateEvent(
                    timestamp=datetime.now(tz=timezone.utc),
                    candidate_type=CandidateEventType.ACTIVE_BATTLE,
                    involved_drivers=sorted(group),
                    confidence=confidence,
                    severity=severity * 0.8,
                    explanation=(
                        f"Active battle: {', '.join(sorted(group))} "
                        f"within {self.BATTLE_GAP_THRESHOLD}s"
                    ),
                    triggering_features={
                        "positions": sorted(positions),
                        "driver_count": len(group),
                    },
                    lap=snapshot.lap,
                    session_key=snapshot.session_key,
                )
            )

        return events

    def _compute_battle_groups(
        self, drivers: list[DriverState]
    ) -> list[list[str]]:
        """Compute battle groups from gap_ahead data."""
        sorted_drivers = sorted(drivers, key=lambda d: d.position)
        groups: list[list[str]] = []
        current_group: list[str] = []

        for driver in sorted_drivers:
            if not current_group:
                current_group.append(driver.driver_id)
                continue

            gap = driver.gap_ahead
            if gap is not None and gap < self.BATTLE_GAP_THRESHOLD:
                current_group.append(driver.driver_id)
            else:
                if len(current_group) >= 2:
                    groups.append(current_group)
                current_group = [driver.driver_id]

        if len(current_group) >= 2:
            groups.append(current_group)

        return groups

    @staticmethod
    def _groups_from_pairs(
        pairs: list[tuple[str, str]],
    ) -> list[list[str]]:
        """Merge battle pairs into connected groups via union-find."""
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for a, b in pairs:
            parent.setdefault(a, a)
            parent.setdefault(b, b)
            union(a, b)

        groups_map: dict[str, list[str]] = {}
        for node in parent:
            root = find(node)
            groups_map.setdefault(root, []).append(node)

        return [g for g in groups_map.values() if len(g) >= 2]

    @staticmethod
    def _get_positions(
        snapshot: RaceStateSnapshot, group: list[str]
    ) -> list[int]:
        return [
            d.position for d in snapshot.drivers if d.driver_id in group
        ]
