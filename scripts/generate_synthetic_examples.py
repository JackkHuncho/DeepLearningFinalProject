"""Generate additional synthetic F1 commentary training examples.

Programmatically varies driver pairings, circuits, lap numbers, gaps,
tire compounds, and scenario parameters to bulk up the dataset.

Run after build_training_data.py to append to sft_train.jsonl.
"""

import json
import random
from pathlib import Path

random.seed(2024)

INSTRUCTION = (
    "You are an expert Formula 1 commentator. Generate professional, insightful "
    "race commentary based on the following race situation. Your commentary should "
    "be vivid, strategically informed, and suitable for a live broadcast."
)

DRIVERS = {
    "VER": "Verstappen", "HAM": "Hamilton", "LEC": "Leclerc", "NOR": "Norris",
    "PIA": "Piastri", "SAI": "Sainz", "RUS": "Russell", "PER": "Perez",
    "ALO": "Alonso", "STR": "Stroll", "GAS": "Gasly", "OCO": "Ocon",
    "TSU": "Tsunoda", "RIC": "Ricciardo", "HUL": "Hulkenberg", "MAG": "Magnussen",
    "BOT": "Bottas", "ZHO": "Zhou", "ALB": "Albon", "LAW": "Lawson",
}

CIRCUITS = [
    ("Bahrain", 57), ("Jeddah", 50), ("Melbourne", 58), ("Suzuka", 53),
    ("Shanghai", 56), ("Miami", 57), ("Imola", 63), ("Monaco", 78),
    ("Montreal", 70), ("Barcelona", 66), ("Spielberg", 71), ("Silverstone", 52),
    ("Budapest", 70), ("Spa", 44), ("Zandvoort", 72), ("Monza", 53),
    ("Baku", 51), ("Singapore", 62), ("Austin", 56), ("Mexico City", 71),
    ("Interlagos", 71), ("Las Vegas", 50), ("Lusail", 57), ("Abu Dhabi", 58),
]

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

CORNERS = {
    "Bahrain": ["Turn 1", "Turn 4", "Turn 10"],
    "Jeddah": ["Turn 1", "the main straight", "Turn 22"],
    "Melbourne": ["Turn 1", "Turn 3", "Turn 11"],
    "Suzuka": ["the chicane", "130R", "Spoon"],
    "Monza": ["Variante del Rettifilo", "Parabolica", "Lesmo"],
    "Spa": ["Eau Rouge", "Pouhon", "Bus Stop"],
    "Silverstone": ["Copse", "Maggotts", "Stowe"],
    "Monaco": ["Sainte Devote", "the Loews hairpin", "the Swimming Pool"],
    "Baku": ["Turn 1", "the castle section", "Turn 15"],
    "Singapore": ["Turn 1", "Turn 5", "the Anderson Bridge"],
}

DRIVER_CODES = list(DRIVERS.keys())


def pick_pair(positions=(1, 2)):
    d1, d2 = random.sample(DRIVER_CODES, 2)
    return d1, d2, positions[0], positions[1]


def pick_circuit():
    name, laps = random.choice(CIRCUITS)
    return name, laps


def pick_lap(total, phase="mid"):
    if phase == "early":
        return random.randint(1, total // 4)
    elif phase == "late":
        return random.randint(3 * total // 4, total - 1)
    return random.randint(total // 4, 3 * total // 4)


def gen_overtake_drs(n=15):
    """Generate overtake/DRS battle examples."""
    examples = []
    templates = [
        lambda d1, d2, gap, gp, corner: (
            f"{DRIVERS[d1]} is charging towards {DRIVERS[d2]}, the gap down to {gap} seconds "
            f"and DRS is active. Through {corner}, {DRIVERS[d1]} gets a great run and positions "
            f"the car for the overtake. {DRIVERS[d2]} defends the inside but {DRIVERS[d1]} has "
            f"the momentum on fresher rubber."
        ),
        lambda d1, d2, gap, gp, corner: (
            f"Here comes {DRIVERS[d1]}! {gap} seconds behind {DRIVERS[d2]} and closing fast. "
            f"The DRS flap opens and {DRIVERS[d1]} has a significant straight-line speed advantage. "
            f"{DRIVERS[d2]}'s tires are visibly struggling, you can see the car moving around under "
            f"braking. This move is coming, it's just a matter of when."
        ),
        lambda d1, d2, gap, gp, corner: (
            f"Wheel to wheel! {DRIVERS[d1]} draws alongside {DRIVERS[d2]} on the run to {corner}. "
            f"{DRIVERS[d2]} covers the inside, {DRIVERS[d1]} switches to the outside — brave move! "
            f"They run through the corner side by side, neither giving an inch. "
            f"Outstanding racing from both drivers."
        ),
        lambda d1, d2, gap, gp, corner: (
            f"{DRIVERS[d1]} has been within DRS range for three consecutive laps now, "
            f"just {gap} seconds behind {DRIVERS[d2]}. The problem is the dirty air through "
            f"the technical section — {DRIVERS[d1]} loses everything gained on the straight. "
            f"This is the frustration of modern F1 — DRS giveth and dirty air taketh away."
        ),
        lambda d1, d2, gap, gp, corner: (
            f"The gap is {gap} seconds and {DRIVERS[d1]} is all over the gearbox of {DRIVERS[d2]}. "
            f"Fresh rubber making all the difference here at {gp}. {DRIVERS[d2]} is weaving on "
            f"the straight, a clear sign the rear tires have nothing left. One more attempt "
            f"into {corner} and this could be done."
        ),
    ]

    for _ in range(n):
        d1, d2, p1, p2 = pick_pair(random.choice([(2, 1), (3, 2), (4, 3), (5, 4), (6, 5)]))
        gp, total = pick_circuit()
        lap = pick_lap(total, "mid" if random.random() > 0.4 else "late")
        gap = round(random.uniform(0.2, 1.0), 1)
        closing = round(random.uniform(0.2, 0.6), 1)
        t1, t2 = random.choice(COMPOUNDS), random.choice(COMPOUNDS)
        a1, a2 = random.randint(3, 12), random.randint(15, 30)
        corners = CORNERS.get(gp, ["Turn 1", "Turn 3", "Turn 5"])
        corner = random.choice(corners)
        template = random.choice(templates)

        inp = (
            f"LAP {lap}/{total} | {gp} 2024 Race\n"
            f"OBSERVED: {d1} P{p1}, gap to {d2} P{p2}: {gap}s, closing {closing}s/lap. "
            f"{d1} on {t1} (age {a1}). {d2} on {t2} (age {a2}). DRS available.\n"
            f"INFERRED: {d1} has tire advantage and closing pace. Overtake likely within {random.randint(1,3)} laps."
        )

        examples.append({
            "instruction": INSTRUCTION,
            "input": inp,
            "output": template(d1, d2, gap, gp, corner),
        })

    return examples


def gen_pit_strategy(n=12):
    """Generate pit strategy examples."""
    examples = []
    templates = [
        lambda d, pos, tire, age, gp, new_tire: (
            f"{DRIVERS[d]} dives into the pit lane from P{pos}. Those {tire.lower()} tires "
            f"at {age} laps have served their purpose, and the team are switching to {new_tire.lower()}s "
            f"for the run to the flag. The key question is where {DRIVERS[d]} comes out — "
            f"the pit stop execution needs to be perfect to maintain track position."
        ),
        lambda d, pos, tire, age, gp, new_tire: (
            f"Box box for {DRIVERS[d]}! {tire}s at {age} laps, well past the optimal "
            f"window. Going to {new_tire}s now and the team will be hoping the fresh rubber "
            f"more than compensates for the time lost in the pits. At {gp}, a good out-lap "
            f"is crucial to making this strategy work."
        ),
        lambda d, pos, tire, age, gp, new_tire: (
            f"{DRIVERS[d]} extends the stint to {age} laps on {tire.lower()}s before finally "
            f"pitting. That's a marathon stint by any measure. Fresh {new_tire.lower()}s now "
            f"and the pace difference should be dramatic — we could see lap times drop by "
            f"two seconds or more on this first out-lap."
        ),
    ]

    for _ in range(n):
        d = random.choice(DRIVER_CODES)
        pos = random.randint(1, 10)
        gp, total = pick_circuit()
        lap = pick_lap(total)
        tire = random.choice(COMPOUNDS)
        age = random.randint(15, 35)
        new_tire = random.choice([c for c in COMPOUNDS if c != tire])
        template = random.choice(templates)

        inp = (
            f"LAP {lap}/{total} | {gp} 2024 Race\n"
            f"OBSERVED: {d} P{pos}, pitting now. Was on {tire} (age {age}). "
            f"Switching to {new_tire}.\n"
            f"INFERRED: {'Overcut' if age > 25 else 'Standard'} strategy. "
            f"{'Tire degradation severe.' if age > 25 else 'Optimal pit window.'}"
        )

        examples.append({
            "instruction": INSTRUCTION,
            "input": inp,
            "output": template(d, pos, tire, age, gp, new_tire),
        })

    return examples


def gen_safety_car(n=8):
    examples = []
    templates = [
        lambda cause, gp, leader, gap: (
            f"Safety car at {gp}! {cause} and the field bunches up. "
            f"{DRIVERS[leader]}'s lead of {gap} seconds is wiped out in an instant. "
            f"This is a complete reset for the race — those behind on fresher tires "
            f"will be rubbing their hands. The pit wall strategists are already calculating "
            f"whether to pit under the safety car for effectively free positions."
        ),
        lambda cause, gp, leader, gap: (
            f"Yellow flags waving and the safety car is deployed. {cause}. "
            f"At {gp} this changes everything — {DRIVERS[leader]} had built a "
            f"commanding {gap}-second lead but that advantage evaporates behind the safety car. "
            f"Expect a flurry of pit stops as teams scramble to take advantage of the neutralized pace."
        ),
    ]

    causes = [
        "Debris on track after a clash in the midfield",
        "Car stopped on the racing line at the final corner",
        "Barrier damage after an incident needs repair",
        "Multiple cars off at the same corner, gravel on track",
        "Technical failure leaves a car stranded in a dangerous position",
    ]

    for _ in range(n):
        gp, total = pick_circuit()
        lap = pick_lap(total)
        leader = random.choice(DRIVER_CODES)
        gap = round(random.uniform(2.0, 12.0), 1)
        cause = random.choice(causes)
        template = random.choice(templates)

        inp = (
            f"LAP {lap}/{total} | {gp} 2024 Race\n"
            f"OBSERVED: SAFETY CAR DEPLOYED. {cause}. {leader} P1, "
            f"gap to P2 was {gap}s, now neutralized.\n"
            f"INFERRED: Pit window opens. Free tire change opportunity for many."
        )

        examples.append({
            "instruction": INSTRUCTION,
            "input": inp,
            "output": template(cause, gp, leader, gap),
        })

    return examples


def gen_tire_management(n=10):
    examples = []
    templates = [
        lambda d, tire, age, deg, gp: (
            f"{DRIVERS[d]}'s {tire.lower()} tires at {age} laps are showing significant "
            f"degradation — lap times have dropped off by {deg} seconds over the last three tours. "
            f"You can see the car sliding through the medium-speed corners, the rear just doesn't "
            f"have the grip it needs. If the pace continues to fall at this rate, "
            f"an unplanned stop might be the only option."
        ),
        lambda d, tire, age, deg, gp: (
            f"The tire management from {DRIVERS[d]} has been exceptional at {gp}. "
            f"{age} laps on those {tire.lower()}s and still finding consistent pace. "
            f"The key is how {DRIVERS[d]} is handling the high-speed corners — barely any "
            f"sliding, textbook apex speeds. This is the kind of driving that wins championships."
        ),
        lambda d, tire, age, deg, gp: (
            f"Watch {DRIVERS[d]} through the corners — the {tire.lower()} compound at "
            f"{age} laps is on the edge of the cliff. Degradation at {deg} seconds per lap "
            f"and accelerating. The team needs to make a call soon because every lap "
            f"lost now might not be recoverable, even on fresh rubber."
        ),
    ]

    for _ in range(n):
        d = random.choice(DRIVER_CODES)
        gp, total = pick_circuit()
        lap = pick_lap(total, "late")
        tire = random.choice(COMPOUNDS)
        age = random.randint(18, 35)
        deg = round(random.uniform(0.3, 1.2), 1)
        template = random.choice(templates)

        inp = (
            f"LAP {lap}/{total} | {gp} 2024 Race\n"
            f"OBSERVED: {d} on {tire} (age {age}). Lap time loss: {deg}s over last 3 laps. "
            f"Rear grip visibly reduced.\n"
            f"INFERRED: {'Tire cliff approaching.' if deg > 0.7 else 'Gradual degradation, manageable.'}"
        )

        examples.append({
            "instruction": INSTRUCTION,
            "input": inp,
            "output": template(d, tire, age, deg, gp),
        })

    return examples


def gen_position_battle(n=12):
    examples = []
    templates = [
        lambda d1, d2, p1, p2, gap, gp: (
            f"The battle for P{p2} continues between {DRIVERS[d1]} and {DRIVERS[d2]}. "
            f"Just {gap} seconds separating them and neither driver willing to yield. "
            f"{DRIVERS[d1]} had a look on the previous lap but {DRIVERS[d2]} covered it well. "
            f"This is wheel-to-wheel racing at its finest here at {gp}."
        ),
        lambda d1, d2, p1, p2, gap, gp: (
            f"{DRIVERS[d1]} is applying enormous pressure on {DRIVERS[d2]} for P{p2}. "
            f"The gap is {gap} seconds and {DRIVERS[d1]} has been faster in every sector. "
            f"It's a matter of time now — {DRIVERS[d2]} is defending brilliantly but "
            f"the pace differential is too great to hold indefinitely."
        ),
        lambda d1, d2, p1, p2, gap, gp: (
            f"Side by side through the braking zone! {DRIVERS[d1]} goes for the inside, "
            f"{DRIVERS[d2]} holds the outside — both cars millimeters from the white line. "
            f"Incredible racing! {DRIVERS[d2]} just holds on to P{p2} but {DRIVERS[d1]} "
            f"at {gap} seconds behind isn't going away anytime soon."
        ),
    ]

    for _ in range(n):
        p2 = random.randint(1, 15)
        d1, d2, _, _ = pick_pair((p2 + 1, p2))
        gp, total = pick_circuit()
        lap = pick_lap(total)
        gap = round(random.uniform(0.2, 1.2), 1)
        template = random.choice(templates)

        inp = (
            f"LAP {lap}/{total} | {gp} 2024 Race\n"
            f"OBSERVED: {d1} P{p2+1}, {d2} P{p2}. Gap: {gap}s. "
            f"{'DRS available.' if gap < 1.0 else 'Just outside DRS range.'}\n"
            f"INFERRED: {'Overtake imminent.' if gap < 0.5 else 'Pressure building.'}"
        )

        examples.append({
            "instruction": INSTRUCTION,
            "input": inp,
            "output": template(d1, d2, p2 + 1, p2, gap, gp),
        })

    return examples


def gen_team_radio(n=8):
    examples = []
    messages = [
        ("{d}, push now, we need fastest lap.", "Fastest lap bonus point in play."),
        ("{d}, temperatures are critical, lift and coast please.", "Brake and engine temps elevated."),
        ("{d}, we're looking at Plan B, box next lap.", "Strategy change under consideration."),
        ("{d}, car behind has DRS, defend into Turn 1.", "DRS threat approaching."),
        ("{d}, great job, tyres are looking good, keep this pace.", "Tire condition stable."),
        ("{d}, rain in 10 minutes, we may switch to inters.", "Weather change incoming."),
        ("{d}, your teammate is faster, please let him through.", "Team orders being discussed."),
        ("{d}, penalty under investigation, stewards looking at Turn 4.", "Possible penalty."),
    ]

    templates = [
        lambda d, msg, context, gp: (
            f"Important radio message to {DRIVERS[d]} — the team relaying that {context.lower()} "
            f"This could be a pivotal moment in the race at {gp}. {DRIVERS[d]} acknowledges "
            f"and adjusts accordingly. The communication between driver and pit wall is absolutely "
            f"crucial in these situations."
        ),
        lambda d, msg, context, gp: (
            f"We just heard the engineer on the radio to {DRIVERS[d]}: '{msg}' — "
            f"{context} The response from {DRIVERS[d]} was immediate, you could see the "
            f"driving style change on the very next corner. That's the level of trust and "
            f"communication that separates the top teams."
        ),
    ]

    for _ in range(n):
        d = random.choice(DRIVER_CODES)
        gp, total = pick_circuit()
        lap = pick_lap(total)
        msg_template, context = random.choice(messages)
        msg = msg_template.format(d=d)
        template = random.choice(templates)

        inp = (
            f"LAP {lap}/{total} | {gp} 2024 Race\n"
            f"OBSERVED: Team radio to {d}: '{msg}'\n"
            f"INFERRED: {context}"
        )

        examples.append({
            "instruction": INSTRUCTION,
            "input": inp,
            "output": template(d, msg, context, gp),
        })

    return examples


def gen_race_start(n=8):
    examples = []
    templates = [
        lambda d1, d2, d3, gp, change: (
            f"Lights out and away we go at {gp}! {DRIVERS[d1]} gets a clean launch from pole "
            f"but {change}! The first corner is always chaos and today is no exception. "
            f"As they sort themselves out through the opening complex, the order begins "
            f"to settle but the strategists are already recalculating based on these new positions."
        ),
        lambda d1, d2, d3, gp, change: (
            f"And the lights go out in {gp}! {change}. Contact at the first corner between "
            f"the midfield runners as everyone jockeys for position. The top three get through "
            f"cleanly but further back it's mayhem — there will be front wing damage "
            f"and that means early pit stops for several cars."
        ),
    ]

    for _ in range(n):
        d1, d2, d3 = random.sample(DRIVER_CODES, 3)
        gp, total = pick_circuit()
        changes = [
            f"{DRIVERS[d2]} gets a rocket start and challenges for the lead into Turn 1",
            f"{DRIVERS[d3]} jumps three places off the line with an incredible reaction time",
            f"{DRIVERS[d2]} bogs down on the start and drops from P2 to P5",
            f"wheel-to-wheel action through the first corner as {DRIVERS[d2]} and {DRIVERS[d3]} go side by side",
        ]
        change = random.choice(changes)
        template = random.choice(templates)

        inp = (
            f"LAP 1/{total} | {gp} 2024 Race\n"
            f"OBSERVED: LIGHTS OUT. {d1} P1 on pole. {change}.\n"
            f"INFERRED: Opening lap chaos. Positions shuffled."
        )

        examples.append({
            "instruction": INSTRUCTION,
            "input": inp,
            "output": template(d1, d2, d3, gp, change),
        })

    return examples


def main():
    all_examples = []

    generators = [
        (gen_overtake_drs, 15),
        (gen_pit_strategy, 12),
        (gen_safety_car, 8),
        (gen_tire_management, 10),
        (gen_position_battle, 12),
        (gen_team_radio, 8),
        (gen_race_start, 8),
    ]

    for gen_fn, n in generators:
        all_examples.extend(gen_fn(n))

    output_path = Path("data/datasets/sft_train.jsonl")

    # Append to existing
    existing = 0
    if output_path.exists():
        with open(output_path) as f:
            existing = sum(1 for _ in f)

    with open(output_path, "a") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    total = existing + len(all_examples)
    print(f"Appended {len(all_examples)} synthetic examples to {output_path}")
    print(f"Total dataset size: {total} examples")

    # Regenerate splits
    all_data = []
    with open(output_path) as f:
        for line in f:
            all_data.append(json.loads(line))

    random.seed(42)
    val_indices = set(random.sample(range(len(all_data)), max(1, len(all_data) // 5)))
    train = [ex for i, ex in enumerate(all_data) if i not in val_indices]
    val = [ex for i, ex in enumerate(all_data) if i in val_indices]

    with open("data/datasets/sft_train_split.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open("data/datasets/sft_val_split.jsonl", "w") as f:
        for ex in val:
            f.write(json.dumps(ex) + "\n")

    print(f"Train split: {len(train)} | Val split: {len(val)}")


if __name__ == "__main__":
    main()
