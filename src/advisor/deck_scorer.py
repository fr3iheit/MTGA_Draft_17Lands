"""
src/advisor/deck_scorer.py
Evaluates pool strength, calculates holistic power scores, and identifies top lanes.
"""

from src import constants
from src.card_utils import get_functional_cmc
from src.advisor.mana_base import ManaSourceAnalyzer
from src.utils import normalize_color_string
from src.sealed_logic import HeuristicEvaluator

# Named projection concepts (distinct from the advisor engine's pick-index
# TOTAL_PICKS=45 and card-total TOTAL_CARDS_DRAFTED=45 on purpose): this scorer
# projects a partial pool toward a full draft's spell target. Pool sizes 43-45
# are behaviorally identical under this projection, so 42 vs 45 is a documented
# consistency choice, not a bug to unify.
SPELL_TARGET_PROJECTION_BASE = 42
TARGET_SPELL_COUNT = 23

# Bayesian shrinkage toward the set mean (global) / global + pair-offset
# (archetype). PROVISIONAL: k belongs to cycle 5's calibration surface
# (per-set fit against later, disjoint outcomes) and is expressed in game
# counts (ever_drawn_game_count), not picks.
SHRINKAGE_K = 2000


def shrunk_wr(raw, prior, n, k=SHRINKAGE_K):
    """n/(n+k) shrinkage of raw toward prior. n <= 0 returns raw unchanged:
    with no reliability information the caller's no-data policy applies."""
    if n <= 0:
        return raw
    return prior + (raw - prior) * (n / (n + k))


def effective_pair_key(card, colors):
    """Which pair prices a card in a 3+ color context: the card's own
    best-WUBRG pair among the deck's colors, not a positional truncation.
    Falls back to the first two colors when the card carries no pair data."""
    if len(colors) <= 2:
        return normalize_color_string("".join(colors))
    stats = card.get("deck_colors", {})
    best, best_wr = None, -1.0
    from itertools import combinations

    for a, b in combinations(sorted(colors), 2):
        key = normalize_color_string(a + b)
        wr = float(stats.get(key, {}).get("gihwr") or 0.0)
        if wr > best_wr:
            best, best_wr = key, wr
    if best is None or best_wr <= 0.0:
        return normalize_color_string("".join(sorted(colors)[:2]))
    return best


def card_wr_posteriors(card, colors, metrics=None):
    """Posterior win-rate estimates for `card` under `colors`, WR scale only —
    callers apply their own transforms. Returns a dict with raw values, the
    game-count n, and the shrunk estimates. Shrinkage only applies for n > 0;
    n comes from `samples` (ever_drawn_game_count; absent on shipped fixtures,
    where every card therefore keeps today's raw-value behavior)."""
    stats = card.get("deck_colors", {})
    global_wr = float(stats.get("All Decks", {}).get("gihwr") or 0.0)
    arch_key = effective_pair_key(card, colors)
    arch_stats = stats.get(arch_key, {}) if arch_key != "All Decks" else {}
    arch_wr = float(arch_stats.get("gihwr") or 0.0) if arch_stats else global_wr
    n = int(arch_stats.get("samples", 0) or 0) if arch_stats else 0

    set_global_mean = global_wr
    offset = 0.0
    if metrics:
        try:
            g_mean, _ = metrics.get_metrics("All Decks", "gihwr")
            if g_mean:
                set_global_mean = float(g_mean)
            if arch_stats and arch_key != "All Decks":
                pair_mean, _ = metrics.get_metrics(arch_key, "gihwr")
                if pair_mean:
                    offset = float(pair_mean) - float(g_mean)
        except Exception:
            offset = 0.0

    return {
        "global_wr": global_wr,
        "arch_wr": arch_wr,
        "n": n,
        "arch_prior": global_wr + offset,
        "shrunk_global": shrunk_wr(global_wr, set_global_mean, n),
        "shrunk_arch": shrunk_wr(arch_wr, global_wr + offset, n),
        "pair_offset": offset,
    }

TIER_TO_GIHWR = {
    "A+": 68.0,
    "A ": 66.0,
    "A-": 64.0,
    "B+": 62.0,
    "B ": 60.0,
    "B-": 58.0,
    "C+": 56.0,
    "C ": 54.0,
    "C-": 52.0,
    "D+": 50.0,
    "D ": 48.0,
    "D-": 46.0,
    "F ": 40.0,
    "SB": 40.0,
    "NA": 0.0,
}


def get_card_rating(card, colors, metrics=None, tier_data=None):
    global_mean = 54.0
    set_has_data = False
    if metrics:
        mean_val, _ = metrics.get_metrics("All Decks", "gihwr")
        if mean_val > 0:
            global_mean = mean_val
            set_has_data = True

    stats = card.get("deck_colors", {})
    global_wr = float(stats.get("All Decks", {}).get("gihwr") or 0.0)

    # Dataset deck_colors keys are WUBRG-ordered ("UB", not "BU"); in a 3+
    # color context the card is priced under its own best pair.
    arch_key = effective_pair_key(card, colors)
    arch_wr = float(stats.get(arch_key, {}).get("gihwr") or 0.0)

    post = card_wr_posteriors(card, colors, metrics)
    arch_v = post["shrunk_arch"] if post["n"] > 0 else arch_wr
    global_v = post["shrunk_global"] if post["n"] > 0 else global_wr
    if arch_wr > 30.0 and global_wr > 30.0:
        return (arch_v * 0.7) + (global_v * 0.3)
    elif global_wr > 30.0:
        return global_v
    elif arch_wr > 30.0:
        return arch_v

    if tier_data:
        name, tier_scores = card.get("name", ""), []
        for tier_obj in tier_data.values():
            if name in tier_obj.ratings:
                grade = tier_obj.ratings[name].rating
                score = TIER_TO_GIHWR.get(grade, 0.0)
                if score > 0.0:
                    tier_scores.append(score)
        if tier_scores:
            return sum(tier_scores) / len(tier_scores)

    # In an established format (the set already has win-rate data), a card with
    # no data of its own is one nobody drafts or plays — effectively unplayable
    # (e.g. Worlds Within Worlds). Don't invent a generous score from a metadata
    # heuristic; that fallback is only for Day 1 before any data exists.
    if set_has_data:
        return 0.0

    return HeuristicEvaluator.evaluate(card)


def identify_top_pairs(pool, metrics, tier_data=None):
    global_mean, global_std = (
        metrics.get_metrics("All Decks", "gihwr") if metrics else (54.0, 4.0)
    )
    if global_mean == 0.0:
        global_mean = 54.0
    if global_std == 0.0:
        global_std = 4.0

    playable_baseline = global_mean - (global_std * 0.5)
    scores = {c: 0.0 for c in constants.CARD_COLORS}

    for card in pool:
        colors = card.get(constants.DATA_FIELD_COLORS, [])
        raw_wr = float(
            card.get("deck_colors", {}).get("All Decks", {}).get("gihwr") or 0.0
        )
        wr = get_card_rating(card, ["All Decks"], metrics, tier_data)
        # Raw-WR gate, shrunk weight: shrinkage pulls small-n cards toward the
        # mean, so gating on the shrunk value would let low-sample noise cross
        # the playable baseline and steer pair selection.
        if raw_wr > playable_baseline:
            points = (wr - playable_baseline) / global_std
            for c in colors:
                scores[c] += points

    sorted_c = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if sorted_c[0][1] <= 0.0:
        return []

    top_4_colors = [c[0] for c in sorted_c[:4]]
    from itertools import combinations

    return [list(pair) for pair in combinations(top_4_colors, 2)]


def calculate_holistic_score(deck, colors, pool_size, metrics, tier_data=None):
    if not deck:
        return 0.0, ""

    global_mean, global_std = metrics.get_metrics("All Decks", "gihwr")
    if global_mean == 0.0:
        global_mean = 54.0
    if global_std == 0.0:
        global_std = 4.0

    spells = [c for c in deck if constants.CARD_TYPE_LAND not in c.get("types", [])]
    spell_count = sum(c.get("count", 1) for c in spells)
    if spell_count == 0:
        return 0.0, ""

    # Dataset deck_colors keys are WUBRG-ordered ("UB", not "BU").
    arch_key = normalize_color_string(
        "".join(colors) if len(colors) <= 2 else "".join(colors[:2])
    )
    # Unrated spells (no data of their own in an established set) are charged
    # at replacement level instead of vanishing from the average: every spell
    # the deck actually plays belongs in the denominator.
    replacement_level = global_mean - (global_std * 1.5)
    ratings = []
    for c in spells:
        for _ in range(c.get("count", 1)):
            rating = get_card_rating(c, [arch_key], metrics)
            ratings.append(rating if rating > 0.0 else replacement_level)

    avg_gihwr = (
        sum(ratings) / len(ratings) if ratings else global_mean - global_std
    )
    z_score = (avg_gihwr - global_mean) / global_std
    power_level = 75.0 + (z_score * 12.0)
    breakdown_notes = []

    cmcs = [get_functional_cmc(c) for c in spells for _ in range(c.get("count", 1))]
    avg_cmc = sum(cmcs) / spell_count

    land_count = sum(c.get("count", 1) for c in deck if "Land" in c.get("types", []))
    ramp_count = min(
        3,
        sum(
            c.get("count", 1)
            for c in deck
            if (
                "fixing_ramp" in c.get("tags", [])
                or "treasure" in str(c.get("oracle_text", c.get("text", ""))).lower()
                or "add {" in str(c.get("oracle_text", c.get("text", ""))).lower()
                or "adds {" in str(c.get("oracle_text", c.get("text", ""))).lower()
            )
            and "Land" not in c.get("types", [])
        ),
    )

    mana_deficit = (avg_cmc * 5.5) - (land_count + ramp_count)
    if mana_deficit > 1.5:
        power_level -= mana_deficit * 3.0
        breakdown_notes.append(f"High Curve / Needs Lands (-{mana_deficit * 3.0:.1f})")
    elif mana_deficit < -1.0 and avg_cmc < 2.8:
        power_level += 5.0
        breakdown_notes.append("Excellent Aggro Curve (+5.0)")

    if len(colors) <= 2:
        power_level += 2.5
        breakdown_notes.append("Rock-Solid Mana (+2.5)")

    supertypes = {
        "Creature",
        "Instant",
        "Sorcery",
        "Enchantment",
        "Artifact",
        "Planeswalker",
        "Land",
        "Legendary",
        "Basic",
        "Snow",
        "World",
        "Tribal",
        "Kindred",
    }
    subtypes, changeling_count = {}, 0
    for c in spells:
        text, count = (
            str(c.get("oracle_text", c.get("text", ""))).lower(),
            c.get("count", 1),
        )
        if "changeling" in text:
            changeling_count += count
        for t in c.get("types", []):
            if t not in supertypes:
                subtypes[t] = subtypes.get(t, 0) + count

    if subtypes:
        top_tribe, tribe_count = max(subtypes.items(), key=lambda x: x[1])
        total_tribe_density = tribe_count + changeling_count
        payoff_count = sum(
            c.get("count", 1)
            for c in spells
            if "chosen type" in str(c.get("oracle_text", c.get("text", ""))).lower()
            or top_tribe.lower() in str(c.get("oracle_text", c.get("text", ""))).lower()
        )

        if total_tribe_density >= 6 and payoff_count >= 2:
            bonus = (total_tribe_density * 0.5) + (payoff_count * 1.5)
            power_level += bonus
            breakdown_notes.append(f"{top_tribe} Synergy (+{bonus:.1f})")

    if len(colors) >= 3:
        domain_payoffs = sum(
            c.get("count", 1)
            for c in spells
            if "colors among" in str(c.get("oracle_text", c.get("text", ""))).lower()
            or "basic land types"
            in str(c.get("oracle_text", c.get("text", ""))).lower()
        )
        fixing_count = ManaSourceAnalyzer(deck).total_fixing_cards

        if domain_payoffs >= 2 and fixing_count >= 4:
            power_level += 6.0
            breakdown_notes.append("Supported Domain/Soup (+6.0)")
        elif fixing_count < len(colors) - 1:
            penalty = (len(colors) - 1 - fixing_count) * 6.0
            power_level -= penalty
            breakdown_notes.append(f"Greedy Mana Strain (-{penalty:.1f})")

    evasion_count = sum(
        c.get("count", 1)
        for c in spells
        if "evasion" in c.get("tags", [])
        or any(
            kw in str(c.get("oracle_text", c.get("text", ""))).lower()
            for kw in [
                "flying",
                "trample",
                "menace",
                "can't be blocked",
                "unblockable",
                "deals damage to any target",
            ]
        )
    )
    if evasion_count < 3 and avg_cmc > 2.5:
        power_level -= 5.0
        breakdown_notes.append("Lacks Evasion/Reach (-5.0)")

    expected_spells = int(
        TARGET_SPELL_COUNT
        * (min(SPELL_TARGET_PROJECTION_BASE, pool_size) / SPELL_TARGET_PROJECTION_BASE)
    )
    if spell_count < expected_spells - 1:
        penalty = ((expected_spells - 1) - spell_count) * 10.0
        power_level -= penalty
        breakdown_notes.append(f"Incomplete Deck (-{penalty:.1f})")

    return max(0.0, power_level), ", ".join(breakdown_notes)


def estimate_record(power_level, is_bo3=False):
    if is_bo3:
        if power_level < 65:
            return "0-2 / 1-2"
        if power_level < 78:
            return "2-1"
        return "3-0"
    else:
        if power_level < 60:
            return "0-3 / 1-3"
        if power_level < 70:
            return "2-3 / 3-3"
        if power_level < 80:
            return "4-3 / 5-3"
        if power_level < 88:
            return "6-3"
        return "7-x"
