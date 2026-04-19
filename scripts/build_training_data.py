"""Build SFT training dataset from realistic F1 race scenarios.

Generates structured prompt -> professional commentary pairs.
Run: python scripts/build_training_data.py
Output: data/datasets/sft_train.jsonl
"""

import json
import random
from pathlib import Path

INSTRUCTION = (
    "You are an expert Formula 1 commentator. Generate professional, insightful "
    "race commentary based on the following race situation. Your commentary should "
    "be vivid, strategically informed, and suitable for a live broadcast."
)

# Real F1 driver pairings and teams (2024 season basis)
DRIVERS = {
    "VER": ("Max Verstappen", "Red Bull"),
    "PER": ("Sergio Perez", "Red Bull"),
    "HAM": ("Lewis Hamilton", "Ferrari"),
    "LEC": ("Charles Leclerc", "Ferrari"),
    "NOR": ("Lando Norris", "McLaren"),
    "PIA": ("Oscar Piastri", "McLaren"),
    "SAI": ("Carlos Sainz", "Williams"),
    "RUS": ("George Russell", "Mercedes"),
    "ALO": ("Fernando Alonso", "Aston Martin"),
    "STR": ("Lance Stroll", "Aston Martin"),
    "GAS": ("Pierre Gasly", "Alpine"),
    "OCO": ("Esteban Ocon", "Alpine"),
    "TSU": ("Yuki Tsunoda", "VCARB"),
    "RIC": ("Daniel Ricciardo", "VCARB"),
    "HUL": ("Nico Hulkenberg", "Haas"),
    "MAG": ("Kevin Magnussen", "Haas"),
    "BOT": ("Valtteri Bottas", "Sauber"),
    "ZHO": ("Zhou Guanyu", "Sauber"),
    "ALB": ("Alex Albon", "Williams"),
    "LAW": ("Liam Lawson", "VCARB"),
}

CIRCUITS = [
    ("Bahrain", "Sakhir", 57),
    ("Saudi Arabia", "Jeddah", 50),
    ("Australia", "Melbourne", 58),
    ("Japan", "Suzuka", 53),
    ("China", "Shanghai", 56),
    ("Miami", "Miami", 57),
    ("Emilia Romagna", "Imola", 63),
    ("Monaco", "Monte Carlo", 78),
    ("Canada", "Montreal", 70),
    ("Spain", "Barcelona", 66),
    ("Austria", "Spielberg", 71),
    ("Great Britain", "Silverstone", 52),
    ("Hungary", "Budapest", 70),
    ("Belgium", "Spa", 44),
    ("Netherlands", "Zandvoort", 72),
    ("Italy", "Monza", 53),
    ("Azerbaijan", "Baku", 51),
    ("Singapore", "Marina Bay", 62),
    ("USA", "Austin", 56),
    ("Mexico", "Mexico City", 71),
    ("Brazil", "Interlagos", 71),
    ("Las Vegas", "Las Vegas", 50),
    ("Qatar", "Lusail", 57),
    ("Abu Dhabi", "Yas Marina", 58),
]

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


def make_input(scenario: dict) -> str:
    lines = [f"LAP {scenario['lap']}/{scenario['total_laps']} | {scenario['gp']} {scenario['year']} Race"]
    lines.append(f"OBSERVED: {scenario['observed']}")
    if scenario.get("inferred"):
        lines.append(f"INFERRED: {scenario['inferred']}")
    if scenario.get("context"):
        lines.append(f"CONTEXT: {scenario['context']}")
    return "\n".join(lines)


# ── SCENARIO TEMPLATES ──────────────────────────────────────────────


def drs_battle_scenarios():
    """DRS-enabled overtake battles."""
    examples = []

    examples.append({
        "scenario": {
            "gp": "Monza", "year": 2024, "lap": 42, "total_laps": 53,
            "observed": "HAM P4, gap to NOR P3: 0.8s, closing 0.3s/lap. HAM on MEDIUM (age 15). NOR on HARD (age 22). DRS available.",
            "inferred": "HAM will be within DRS in ~2 laps. NOR tire degradation accelerating.",
            "context": "Hamilton overtook Sainz 5 laps ago. Norris has not pitted since lap 20.",
        },
        "output": "Hamilton is absolutely reeling in Norris now, the gap down to eight tenths and shrinking every lap. Those mediums are clearly working for Lewis, and you have to wonder about Lando's hard tires — twenty-two laps old and the pace is falling away. DRS is live, and if Hamilton can close that final few tenths in the next lap or two, we could see a move into the Variante della Roggia.",
    })

    examples.append({
        "scenario": {
            "gp": "Jeddah", "year": 2024, "lap": 38, "total_laps": 50,
            "observed": "LEC P2, gap to VER P1: 0.6s. DRS detected. LEC on MEDIUM (age 10). VER on HARD (age 18). Speed trap: LEC 338 km/h, VER 335 km/h.",
            "inferred": "Leclerc has straight-line advantage with DRS. Overtake attempt likely on the main straight.",
            "context": "Leclerc has been within DRS range for 3 consecutive laps. Two previous attempts blocked by Verstappen's defending.",
        },
        "output": "Leclerc is right there again, six tenths and DRS activated! He's had the measure of Verstappen in a straight line all afternoon, three thirty-eight through the speed trap versus three thirty-five for Max. This is the third lap in succession he's been in the window — Verstappen has defended superbly twice already, but the Ferrari's fresh mediums are giving Charles more and more of an edge with every passing lap.",
    })

    examples.append({
        "scenario": {
            "gp": "Spa", "year": 2024, "lap": 30, "total_laps": 44,
            "observed": "NOR P3, gap to PIA P2: 0.4s. DRS active on Kemmel Straight. Both McLarens on MEDIUM (NOR age 12, PIA age 12).",
            "inferred": "Identical strategy, pure driver pace battle. NOR closing through sector 2.",
            "context": "McLaren team radio silent on team orders. Both drivers told to race freely.",
        },
        "output": "Now this is fascinating — Norris is just four tenths behind his own teammate through Eau Rouge and onto the Kemmel Straight with DRS wide open. Same tires, same age, same strategy — this is nothing but raw driver-against-driver pace. Lando's been finding time through the middle sector and that's translating into a real opportunity here. McLaren have said they're free to race, and Norris clearly intends to take full advantage.",
    })

    examples.append({
        "scenario": {
            "gp": "Baku", "year": 2024, "lap": 44, "total_laps": 51,
            "observed": "PER P5, gap to ALO P4: 0.3s. DRS zone approaching. PER on SOFT (age 6). ALO on MEDIUM (age 14).",
            "inferred": "Perez has significant tire advantage. Overtake highly probable on main straight.",
            "context": "Alonso defending aggressively, used two lock-ups in last 3 laps. Perez promoted to softs at last stop.",
        },
        "output": "Perez is all over the back of Alonso now, three tenths and those fresh soft tires are giving Sergio everything he needs. Fernando has been defending like a lion but the lock-ups are telling — those mediums at fourteen laps old simply can't match the grip Perez has. Coming onto the main straight with DRS, this looks inevitable. The question is whether Alonso can make it difficult enough to cost Perez time to the cars ahead.",
    })

    examples.append({
        "scenario": {
            "gp": "Austin", "year": 2024, "lap": 48, "total_laps": 56,
            "observed": "RUS P6, gap to GAS P5: 0.9s, closing 0.5s/lap. RUS on SOFT (age 4). GAS on HARD (age 28).",
            "inferred": "Russell will catch Gasly within 2 laps. Gasly's tires critically degraded.",
            "context": "Russell's late pit stop gamble for softs. Gasly has not pitted since lap 28.",
        },
        "output": "Russell's late-stop gamble is paying dividends — nine tenths behind Gasly and eating up half a second a lap on those brand-new softs. Pierre's been out on those hards since lap twenty-eight, that's nearly thirty laps and the degradation must be severe. George should have this done within a couple of laps, and if the pace holds, there could be more positions up for grabs in the closing stages.",
    })

    return examples


def pit_strategy_scenarios():
    """Pit stop strategy decisions."""
    examples = []

    examples.append({
        "scenario": {
            "gp": "Silverstone", "year": 2024, "lap": 28, "total_laps": 52,
            "observed": "VER P1, pitting now. Was on MEDIUM (age 24). NOR P2, gap was 4.2s. HAM P3 at 6.1s.",
            "inferred": "VER switching to HARD for final stint. Undercut attempt by NOR possible if he pits next lap.",
            "context": "VER's lap times degraded 0.8s over last 5 laps. Optimal window for MEDIUM was 20-22 laps.",
        },
        "output": "Verstappen is in! Box box for Max, and those mediums have gone well beyond their optimal window — we saw the times dropping off significantly over the last five laps. He'll go onto the hard compound for what should be a twenty-four-lap run to the flag. Now the big question: does Norris respond immediately? At four point two seconds, Lando is right in the undercut window. If McLaren leave him out even one more lap, they risk losing that track position advantage.",
    })

    examples.append({
        "scenario": {
            "gp": "Hungary", "year": 2024, "lap": 45, "total_laps": 70,
            "observed": "LEC P3 on MEDIUM (age 20). PIT ENTRY. Switching to SOFT for final 25 laps. VER P1 (+8.2s), NOR P2 (+3.1s) ahead.",
            "inferred": "Aggressive strategy — softs typically last 18-20 laps at Budapest. LEC gambling on pace over longevity.",
            "context": "Ferrari strategist called 'opposite to Norris' on team radio. Track temp 48C.",
        },
        "output": "Leclerc dives into the pits and Ferrari are going aggressive — soft tires for the final twenty-five laps! That is a bold call at Budapest in these temperatures. The soft compound typically gives you eighteen to twenty laps here, so Charles is going to need to build a big enough advantage before the degradation cliff hits. Ferrari have clearly decided they can't beat Norris on the same strategy, so they're rolling the dice. High risk, but if the pace is there early, it could be a masterstroke.",
    })

    examples.append({
        "scenario": {
            "gp": "Barcelona", "year": 2024, "lap": 15, "total_laps": 66,
            "observed": "HAM P7, pitting early. SOFT (age 15) to MEDIUM. Normal first stop window is lap 18-22.",
            "inferred": "Early stop suggests tire graining issues. Mercedes looking for track position via undercut.",
            "context": "Hamilton reported 'terrible rear end' on radio 3 laps ago. Both Aston Martins ahead within 2s.",
        },
        "output": "Hamilton is coming in early, lap fifteen when we expected the first stops around lap eighteen to twenty. Lewis was on the radio complaining about the rear end three laps ago, and the graining on those softs must have been worse than Mercedes anticipated. They're putting him onto mediums now — this is partly damage limitation, partly an aggressive undercut play on the Aston Martins ahead. If this medium stint can go long, it completely reshapes Hamilton's race.",
    })

    examples.append({
        "scenario": {
            "gp": "Singapore", "year": 2024, "lap": 35, "total_laps": 62,
            "observed": "NOR P1 (+3.5s to LEC P2). NOR on HARD (age 20). LEC on HARD (age 15). PER P3 pitting now, switching MEDIUM to HARD.",
            "inferred": "Perez committing to two-stop. NOR and LEC likely one-stop if tires hold.",
            "context": "Safety car on lap 20 bunched the field. Track evolution significant tonight.",
        },
        "output": "Perez is in for his second stop, that confirms a two-stop strategy for the Red Bull. Meanwhile Norris and Leclerc up ahead look committed to going to the end on these hards — Lando's set are twenty laps old but the track evolution under the lights tonight has been remarkable, the grip just keeps coming. Three and a half seconds the gap to Leclerc, comfortable for now, but if Charles's newer rubber starts to tell in the final fifteen laps, that lead could evaporate quickly.",
    })

    examples.append({
        "scenario": {
            "gp": "Mexico City", "year": 2024, "lap": 50, "total_laps": 71,
            "observed": "ALO P8, BOX BOX called. HARD (age 32) to SOFT. Track position will drop to P12.",
            "inferred": "Desperate move for points. Need to pass 4 cars in 21 laps on softs.",
            "context": "Alonso has dropped 2.5s/lap over last 5 laps. Tire cliff reached.",
        },
        "output": "Alonso pits from eighth and that's the tire cliff we've been expecting — two and a half seconds a lap lost over the last five, those hards completely gone after thirty-two laps at altitude. He'll rejoin around twelfth on fresh softs with twenty-one laps to go. Fernando needs to pass four cars to get back into the points, and at his age, on fresh rubber, you wouldn't bet against him. But it's a tall order even for a two-time champion.",
    })

    return examples


def safety_car_scenarios():
    """Safety car and incident scenarios."""
    examples = []

    examples.append({
        "scenario": {
            "gp": "Melbourne", "year": 2024, "lap": 22, "total_laps": 58,
            "observed": "SAFETY CAR DEPLOYED. STR stopped at Turn 9, car off track. All drivers bunching up. Gap between P1 VER and P2 LEC was 5.8s, now neutralized.",
            "inferred": "Pit window opens for leaders. Free tire change opportunity.",
            "context": "VER on MEDIUM (age 18), LEC on MEDIUM (age 15). Most of field on first stint.",
        },
        "output": "Safety car! Stroll has stopped at turn nine, the Aston Martin off the track and that brings out the safety car. This completely reshapes the race — Verstappen's five-point-eight-second lead is gone in an instant. Now the critical question: does Max pit? He's on eighteen-lap-old mediums, this is essentially a free pit stop. If Red Bull don't take this opportunity, Ferrari almost certainly will for Leclerc, and suddenly you've got a battle on your hands at the restart on significantly fresher rubber.",
    })

    examples.append({
        "scenario": {
            "gp": "Suzuka", "year": 2024, "lap": 1, "total_laps": 53,
            "observed": "RED FLAG. Multi-car incident at Turn 1. RIC and MAG contact, debris across track. HUL collected in secondary incident.",
            "inferred": "Extended stoppage likely. Track cleanup required. Grid restart probable.",
            "context": "Wet conditions, light rain falling. All cars started on intermediates.",
        },
        "output": "Red flag at Suzuka! A nightmare start — Ricciardo and Magnussen have come together at the first corner and there is debris everywhere across the track. Hulkenberg has been collected as well in the secondary incident. In these wet conditions that was always a risk into turn one, the spray making visibility almost impossible for the midfield runners. We're going to have an extended stoppage here while they clear the track. All three drivers appear to be out of their cars, which is the most important thing.",
    })

    examples.append({
        "scenario": {
            "gp": "Baku", "year": 2024, "lap": 40, "total_laps": 51,
            "observed": "VSC DEPLOYED. Debris reported between Turn 15-16. VER P1, LEC P2 (+1.2s), NOR P3 (+4.8s).",
            "inferred": "VSC delta time in effect. Pit stop under VSC saves ~15s vs normal stop. NOR in pit window.",
            "context": "NOR needs to pit within 5 laps. This VSC is a gift for McLaren strategy.",
        },
        "output": "Virtual Safety Car and McLaren must be absolutely delighted — Norris needed to pit within the next five laps and a VSC stop costs roughly fifteen seconds less than pitting under green flag conditions. This is a gift from the strategy gods. The question is whether McLaren react quickly enough — every second counts under Virtual Safety Car because it could end at any moment. Norris needs to get into that pit lane immediately.",
    })

    examples.append({
        "scenario": {
            "gp": "Interlagos", "year": 2024, "lap": 55, "total_laps": 71,
            "observed": "SAFETY CAR. Rain intensifying. Multiple cars aquaplaning. ZHO spun at Juncao, recovered. Track water levels rising.",
            "inferred": "Possible red flag if conditions worsen. Tire change to WET likely on restart.",
            "context": "Currently all on INTER. Last 3 laps saw 5+ seconds slower than dry pace. Spray immense on main straight.",
        },
        "output": "Safety car again at Interlagos and the conditions are deteriorating rapidly. Zhou has just spun at Juncao, managed to gather it up, but the aquaplaning is becoming a serious concern. Five seconds slower than dry pace over the last three laps and the spray on the main straight is unbelievable — you simply cannot see the car in front. Race control will be watching this very carefully. If this rain keeps up, a red flag suspension feels inevitable, and when we do get going again, full wets might be the only sensible option.",
    })

    return examples


def tire_degradation_scenarios():
    """Tire degradation and management scenarios."""
    examples = []

    examples.append({
        "scenario": {
            "gp": "Bahrain", "year": 2024, "lap": 40, "total_laps": 57,
            "observed": "PER P4 on SOFT (age 22). Lap times: L38 1:35.2, L39 1:35.8, L40 1:36.5. Gap to ALO P5 was 3.2s, now 1.8s.",
            "inferred": "Severe rear degradation. Losing ~0.6s per lap. ALO will catch within 3 laps.",
            "context": "Perez extended first stint to overcut. Strategy now backfiring.",
        },
        "output": "Perez is in real trouble here. Those soft tires are twenty-two laps old and the degradation has hit hard — he's lost six tenths a lap over the last three tours and Alonso has cut the gap from three seconds to under two. The overcut strategy that looked so clever ten laps ago is unraveling fast. Sergio needs to manage these tires for another seventeen laps and at this rate of degradation, that simply isn't realistic. Red Bull may need to consider a second stop, but that drops him well out of the points.",
    })

    examples.append({
        "scenario": {
            "gp": "Zandvoort", "year": 2024, "lap": 52, "total_laps": 72,
            "observed": "VER P1 on MEDIUM (age 28). Lap times stable at 1:14.5 for last 5 laps. NOR P2 (+2.1s) on MEDIUM (age 23).",
            "inferred": "Verstappen managing tires expertly. Consistent pace despite high age. Home advantage in tire management.",
            "context": "VER averaging 0.2s faster than NOR despite older tires. Wind direction changed 10 laps ago.",
        },
        "output": "Twenty-eight laps on those mediums and Verstappen is still putting in one-fourteen-fives — that is extraordinary tire management from Max on his home circuit. Norris is five laps fresher and still can't match that pace. The Dutchman knows every bump, every camber change around Zandvoort, and it shows in how he's nursing these tires. Two-point-one seconds the gap, and if anything it's growing. This is a masterclass in race management.",
    })

    examples.append({
        "scenario": {
            "gp": "Lusail", "year": 2024, "lap": 35, "total_laps": 57,
            "observed": "Multiple drivers reporting blistering. HAM: 'left front is gone'. RUS: 'massive vibration'. Track temp 52C.",
            "inferred": "Extreme track temperatures causing premature tire failure. Potential safety concern.",
            "context": "Pirelli warning about 30-lap maximum on HARD at this temperature. Several drivers past that.",
        },
        "output": "The tire situation in Qatar is becoming critical — Hamilton on the radio saying his left front is gone, Russell reporting massive vibration. Track temperatures at fifty-two degrees are absolutely punishing these compounds. Pirelli flagged a thirty-lap maximum on the hards in these conditions and we've got several drivers well beyond that. This is moving from a performance issue to a potential safety concern — the blistering can lead to sudden delamination, and at these speeds around Lusail, that's the last thing anyone wants.",
    })

    return examples


def lead_battle_scenarios():
    """Close battles for the lead."""
    examples = []

    examples.append({
        "scenario": {
            "gp": "Silverstone", "year": 2024, "lap": 48, "total_laps": 52,
            "observed": "HAM P1, NOR P2 (+0.5s), VER P3 (+1.2s). HAM on SOFT (age 8). NOR on SOFT (age 6). 4 laps remaining.",
            "inferred": "NOR has fresher tires and was faster in sector 2. Final 4 laps will decide the race.",
            "context": "Hamilton leading at home. Last won here in 2021. Crowd on their feet at every corner.",
        },
        "output": "Four laps to go at Silverstone and Hamilton leads Norris by just half a second! Lewis has the home crowd willing him on at every single corner, but Lando's softs are two laps fresher and he was quicker through Maggotts and Becketts on that last tour. Verstappen lurking just over a second behind, ready to pounce if either of them makes a mistake. This is what Formula One is all about — Hamilton hunting a home victory, Norris pushing for the biggest result of his career. Buckle up for these final laps.",
    })

    examples.append({
        "scenario": {
            "gp": "Monza", "year": 2024, "lap": 50, "total_laps": 53,
            "observed": "LEC P1, SAI P2 (+0.3s). Both Ferrari on MEDIUM. LEC tire age 14, SAI tire age 12. Italian crowd ecstatic.",
            "inferred": "Ferrari 1-2 at home. SAI marginally faster but team unlikely to interfere.",
            "context": "Last Ferrari 1-2 at Monza was 2019. Tifosi atmosphere unprecedented.",
        },
        "output": "Three laps to go and it's Ferrari one-two at Monza! Leclerc leads Sainz by three tenths, and listen to that crowd — the tifosi are creating an absolute wall of sound. Carlos is fractionally quicker on those slightly fresher mediums, but there is no way Ferrari are going to ask these two to swap. This is about the team, this is about the red cars at the cathedral of speed. If they can hold formation for three more laps, it will be the first Ferrari one-two at Monza since twenty-nineteen.",
    })

    examples.append({
        "scenario": {
            "gp": "Las Vegas", "year": 2024, "lap": 45, "total_laps": 50,
            "observed": "VER P1, NOR P2 (+3.8s). VER needs P1 or NOR below P6 to clinch championship. NOR setting fastest laps.",
            "inferred": "NOR can't catch VER but closing. Championship will be decided here.",
            "context": "VER leads championship by 8 points. This is the penultimate race.",
        },
        "output": "Five laps from the championship. Verstappen leads by three-point-eight seconds, Norris setting fastest laps but running out of time and tarmac. Max needs to win here or have Norris finish below sixth to wrap up his fourth world title on the Las Vegas Strip. Lando is giving everything but that gap is just too large to close in five laps. Barring a mechanical failure or a late safety car, we are about to witness Max Verstappen become a four-time World Champion.",
    })

    return examples


def midfield_battle_scenarios():
    examples = []

    examples.append({
        "scenario": {
            "gp": "Montreal", "year": 2024, "lap": 55, "total_laps": 70,
            "observed": "ALO P6, HUL P7 (+0.4s), TSU P8 (+0.9s), GAS P9 (+1.3s). Four-car train. All on MEDIUM.",
            "inferred": "DRS train neutralizing pace differences. Overtaking extremely difficult.",
            "context": "This battle has been ongoing for 15 laps. ALO defending well but pace dropping.",
        },
        "output": "The midfield battle continues to rage — Alonso leading this four-car DRS train with Hulkenberg, Tsunoda, and Gasly all stacked up within one-point-three seconds. It's been fifteen laps now and nobody can make a move stick. The DRS effect is actually working against the chasers here — each car gives a tow to the one behind, so the relative advantage is neutralized. Alonso's race craft is keeping them all at bay, but his pace is dropping and eventually that dam will break.",
    })

    examples.append({
        "scenario": {
            "gp": "Imola", "year": 2024, "lap": 40, "total_laps": 63,
            "observed": "ALB P10 battling MAG P9. Gap 0.2s. ALB on MEDIUM (age 8), MAG on HARD (age 18). ALB faster by 0.4s/lap.",
            "inferred": "ALB should pass within 2 laps. Final point at stake.",
            "context": "Both teams fighting for P8 in constructors. This result worth millions in prize money.",
        },
        "output": "Albon is hounding Magnussen for that final point — two tenths apart and Alex is four tenths a lap quicker on those newer mediums. This isn't just about one point for the driver, this is constructors' championship money on the line for both Williams and Haas. Every position at the end of the year is worth millions in prize money. Magnussen's hards are eighteen laps old and fading, Albon has the pace advantage — this should happen in the next lap or two.",
    })

    return examples


def race_start_scenarios():
    examples = []

    examples.append({
        "scenario": {
            "gp": "Bahrain", "year": 2024, "lap": 1, "total_laps": 57,
            "observed": "LIGHTS OUT. VER P1 good start. LEC jumps NOR into Turn 1 from P3 to P2. PER slow off line, drops from P4 to P7.",
            "inferred": "LEC's launch was exceptional. PER likely clutch issue.",
            "context": "Grid: VER, NOR, LEC, PER, HAM. Formation lap showed no issues.",
        },
        "output": "Lights out in Bahrain and it's Verstappen with a clean getaway from pole! But look at Leclerc — what a start from the Ferrari, he's gone around the outside of Norris into turn one and takes second place! Meanwhile disaster for Perez, a terrible launch from fourth and he's dropped to seventh. The clutch didn't bite for Sergio and that's cost him three positions before the first braking zone. Verstappen leads, Leclerc second, Norris third — and that Ferrari start changes the entire complexion of this race.",
    })

    examples.append({
        "scenario": {
            "gp": "Spa", "year": 2024, "lap": 1, "total_laps": 44,
            "observed": "LIGHTS OUT. HAM P14 (grid penalty) rockets through field. By La Source: P14 to P9. Through Eau Rouge: P9 to P7.",
            "inferred": "Exceptional race start. Clean air through Eau Rouge enabled passes.",
            "context": "Hamilton took new power unit, 10-place grid penalty. Started P14 on MEDIUM while top 10 on SOFT.",
        },
        "output": "Hamilton has absolutely nailed this start from fourteenth! Seven positions gained before the end of the first lap — through La Source he's already ninth, and by the time he crests Eau Rouge he's up to seventh! Lewis started on mediums while the top ten are on softs, so he's on a completely different strategy, and that tire advantage is going to pay dividends later in the race. Stunning opening lap from the seven-time champion. The grid penalty may end up being almost irrelevant.",
    })

    return examples


def final_lap_scenarios():
    examples = []

    examples.append({
        "scenario": {
            "gp": "Austria", "year": 2024, "lap": 71, "total_laps": 71,
            "observed": "FINAL LAP. NOR P1, VER P2 (+0.8s). NOR defending since lap 65. Both on HARD. VER closing 0.1s per sector.",
            "inferred": "VER may catch NOR by final corner. Defending through turns 9-10 critical.",
            "context": "NOR's first career win at stake. VER attempted move at Turn 3 on lap 69, ran wide.",
        },
        "output": "Last lap! Norris leads Verstappen by eight tenths and Max is closing, a tenth per sector — this is going to be incredibly tight! Lando's first ever Formula One victory is within touching distance but Verstappen tried a move at turn three two laps ago and he'll try again. Through the fast right-handers, Norris holds the line — through nine, through ten — and he's done it! Norris crosses the line, LANDO NORRIS WINS HIS FIRST GRAND PRIX! What a drive, what a defence against the reigning world champion!",
    })

    examples.append({
        "scenario": {
            "gp": "Abu Dhabi", "year": 2024, "lap": 58, "total_laps": 58,
            "observed": "FINAL LAP. VER P1 (+12.3s). LEC P2, NOR P3 battling for P2. Season finale.",
            "inferred": "VER cruising to victory. LEC vs NOR battle for P2 in championship determines final standings.",
            "context": "LEC and NOR tied on points for P2 in championship. Head-to-head for runner-up position.",
        },
        "output": "Final lap of the season in Abu Dhabi. Verstappen cruises to the victory twelve seconds clear, but the real drama is behind — Leclerc and Norris side by side through the Yas Marina hotel section, fighting for second place in the race AND second in the championship! They're dead level on points, whoever finishes ahead takes the runner-up spot. Leclerc holds the inside into turn eleven, Norris tries the switchback — but Charles holds firm! Leclerc finishes second, and that should be enough for P2 in the championship!",
    })

    return examples


def weather_change_scenarios():
    examples = []

    examples.append({
        "scenario": {
            "gp": "Silverstone", "year": 2024, "lap": 30, "total_laps": 52,
            "observed": "Rain drops detected sector 3. VER P1, HAM P2 (+2.1s). All on MEDIUM (dry). Radar shows rain band approaching.",
            "inferred": "Crossover to INTER within 5-10 laps. Timing of switch critical.",
            "context": "British weather classic. Sun in sector 1, dark clouds over sector 3.",
        },
        "output": "Here comes the British weather! Rain spotted in sector three and the radar shows a band of precipitation heading straight for the circuit. Everyone is on dry-weather mediums and the timing of the switch to intermediates could decide this grand prix. Do you pit now and lose time on a damp but drying track, or do you stay out and risk being on slicks when the heavens truly open? Verstappen leads Hamilton by two-point-one seconds — whoever makes the right call on the crossover could swing this race entirely.",
    })

    examples.append({
        "scenario": {
            "gp": "Interlagos", "year": 2024, "lap": 20, "total_laps": 71,
            "observed": "Heavy rain started lap 18. Multiple offs. BOT aquaplaned into barrier, out. NOR P1 on INTER. VER P5 struggling.",
            "inferred": "Red flag consideration if conditions worsen. VER uncharacteristically poor in wet.",
            "context": "Track flooding on back straight. Standing water at Juncao.",
        },
        "output": "It is chaos in Sao Paulo! The rain came down like a curtain two laps ago and the track is flooding. Bottas has gone off at speed, aquaplaning straight into the barrier — he's okay but the car is destroyed. Norris is threading the needle in the lead on inters but even he is having moments. And look at Verstappen, down in fifth! Max is struggling for grip and confidence in these conditions, something we rarely see. Standing water at Juncao is a real concern — race control must be considering a red flag.",
    })

    return examples


def team_radio_strategy_scenarios():
    examples = []

    examples.append({
        "scenario": {
            "gp": "Monaco", "year": 2024, "lap": 60, "total_laps": 78,
            "observed": "LEC P1. Team radio: 'Charles, we are thinking about Plan C, box this lap.' LEC responds: 'No no no, stay out, the tires are fine.' STAYED OUT.",
            "inferred": "Driver overriding team strategy. Tire condition marginal but driver feels confident.",
            "context": "LEC leading at home. Emotional significance immense. Tires at 30 laps, within safe window.",
        },
        "output": "Drama on the Ferrari pit wall! They called Leclerc in for Plan C but Charles has overruled the team — 'No no no, stay out, the tires are fine!' You can hear the emotion in his voice. This is Monaco, this is home, and Leclerc is not giving up track position at any cost. The tires are thirty laps old but still within the safe operating window, and at Monaco track position is everything. The team have to trust their driver here — Charles knows what these tires feel like better than any data screen can tell them.",
    })

    examples.append({
        "scenario": {
            "gp": "Suzuka", "year": 2024, "lap": 35, "total_laps": 53,
            "observed": "VER P1. Team radio from engineer: 'Max, NOR has pitted, he's on soft, gap is 22 seconds. You need to push now, targets 1:31.5.' VER: 'Understood.'",
            "inferred": "Red Bull managing the undercut. VER needs to build gap before his own stop.",
            "context": "VER was managing pace +0.5s off potential. Now needs to switch to qualifying mode.",
        },
        "output": "The radio message tells the whole story — Norris has pitted for softs and Verstappen's engineer wants one-thirty-one-fives from Max. He's been managing the pace, cruising half a second off his potential, and now he needs to flip the switch into qualifying mode. Twenty-two seconds is the gap and Red Bull need to preserve enough of that to stay ahead after Max's own stop. This is where Verstappen is at his most dangerous — when the team asks him to push, the response is always devastating.",
    })

    return examples


def championship_decisive_scenarios():
    examples = []

    examples.append({
        "scenario": {
            "gp": "Qatar", "year": 2024, "lap": 50, "total_laps": 57,
            "observed": "VER P5. NOR P1 winning. PIA P2. If result stands: VER champion by 2 points. VER needs P9 or better.",
            "inferred": "VER managing position. No need to risk anything. Championship in hand if finishes here.",
            "context": "VER leads NOR by 60 points with 2 races left (max 60 available). P5 more than enough.",
        },
        "output": "Seven laps to go and Verstappen sits in fifth place — and that is more than enough. Norris can win every remaining race and it won't matter if Max finishes here. The Dutchman is driving with the calm assurance of a man who knows the mathematics are overwhelmingly in his favor. No heroics needed, no risks required, just bring the car home. Seven more laps and Max Verstappen will be a four-time World Champion, joining the most exclusive club in motorsport alongside Prost and Vettel.",
    })

    return examples


def recovery_drive_scenarios():
    examples = []

    examples.append({
        "scenario": {
            "gp": "Jeddah", "year": 2024, "lap": 40, "total_laps": 50,
            "observed": "HAM started P18 after qualifying crash. Now P6 (+0.3s to GAS P5). 14 overtakes completed.",
            "inferred": "Exceptional recovery drive. Car pace clearly top 5 material despite grid position.",
            "context": "Hamilton's qualifying crash damaged rear wing, rebuilt overnight. Fresh engine components.",
        },
        "output": "Fourteen overtakes and counting! Hamilton started eighteenth after that qualifying crash and he's carved his way through to sixth with ten laps to go. The Mercedes mechanics rebuilt this car overnight after the rear wing damage, and Lewis has repaid their efforts in the most Hamilton way possible — a breathtaking recovery drive through the Jeddah field. He's three tenths behind Gasly now for fifth, and on this form, there's absolutely no reason to think he'll stop here.",
    })

    return examples


def build_dataset() -> list[dict]:
    """Build all training examples."""
    all_examples = []

    generators = [
        drs_battle_scenarios,
        pit_strategy_scenarios,
        safety_car_scenarios,
        tire_degradation_scenarios,
        lead_battle_scenarios,
        midfield_battle_scenarios,
        race_start_scenarios,
        final_lap_scenarios,
        weather_change_scenarios,
        team_radio_strategy_scenarios,
        championship_decisive_scenarios,
        recovery_drive_scenarios,
    ]

    for gen_fn in generators:
        for example in gen_fn():
            all_examples.append({
                "instruction": INSTRUCTION,
                "input": make_input(example["scenario"]),
                "output": example["output"],
            })

    return all_examples


def main():
    examples = build_dataset()
    output_path = Path("data/datasets/sft_train.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(examples)} training examples to {output_path}")

    # Also create a small validation split
    random.seed(42)
    val_indices = set(random.sample(range(len(examples)), max(1, len(examples) // 5)))
    train = [ex for i, ex in enumerate(examples) if i not in val_indices]
    val = [ex for i, ex in enumerate(examples) if i in val_indices]

    train_path = Path("data/datasets/sft_train_split.jsonl")
    val_path = Path("data/datasets/sft_val_split.jsonl")

    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open(val_path, "w") as f:
        for ex in val:
            f.write(json.dumps(ex) + "\n")

    print(f"Train split: {len(train)} examples -> {train_path}")
    print(f"Val split: {len(val)} examples -> {val_path}")


if __name__ == "__main__":
    main()
