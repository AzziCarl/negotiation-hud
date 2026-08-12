
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Set, Tuple


# -----------------------------
# 0) Phrase library (text -> features)
# -----------------------------

# Tip: students will write many variants. Add synonyms here, not in the classifier.
KEYWORD_MAP: Dict[str, str] = {
    # --- Posture / torso ---
    "crossed arms": "crossed_arms",
    "arms crossed": "crossed_arms",
    "folded arms": "crossed_arms",
    "arms folded": "crossed_arms",
    "closed posture": "crossed_arms",

    "leaning back": "leaning_back",
    "leans back": "leaning_back",
    "reclined": "leaning_back",
    "leans away": "leaning_back",

    "leaning forward": "leaning_forward",
    "leans forward": "leaning_forward",
    "sits forward": "leaning_forward",
    "leans in": "leaning_forward",

    "relaxed posture": "relaxed_posture",
    "open posture": "relaxed_posture",
    "relaxed": "relaxed_posture",

    "upright posture": "upright_posture",
    "sits upright": "upright_posture",
    "straight posture": "upright_posture",
    "rigid posture": "rigid_posture",
    "tense posture": "rigid_posture",
    "stiff posture": "rigid_posture",

    # --- Head / face / expression ---
    "smiling": "smiling",
    "smiles": "smiling",
    "big smile": "smiling",
    "grinning": "smiling",

    "frowning": "frowning",
    "frowns": "frowning",
    "scowling": "frowning",

    "tight lips": "tight_lips",
    "pressed lips": "tight_lips",
    "pursed lips": "tight_lips",
    "jaw clenched": "jaw_clenched",
    "clenched jaw": "jaw_clenched",

    "raised eyebrows": "raised_eyebrows",
    "eyebrows raised": "raised_eyebrows",
    "eyebrow raise": "raised_eyebrows",

    "sighs": "sighing",
    "sighing": "sighing",

    # --- Eye contact / gaze ---
    "good eye contact": "good_eye_contact",
    "strong eye contact": "good_eye_contact",
    "steady eye contact": "good_eye_contact",
    "maintains eye contact": "good_eye_contact",

    "avoids eye contact": "weak_eye_contact",
    "weak eye contact": "weak_eye_contact",
    "looks down": "weak_eye_contact",
    "looking down": "weak_eye_contact",
    "won't look": "weak_eye_contact",

    "looks away": "looking_away",
    "looking away": "looking_away",
    "glances away": "looking_away",
    "keeps looking away": "looking_away",

    "stares": "staring",
    "staring": "staring",
    "eye roll": "eye_roll",
    "rolling eyes": "eye_roll",

    # --- Agreement / listening signals ---
    "nodding": "nodding",
    "nods": "nodding",
    "nod": "nodding",
    "taking notes": "taking_notes",
    "takes notes": "taking_notes",
    "writes notes": "taking_notes",

    "saying yes": "verbal_agreement",
    "agrees": "verbal_agreement",
    "agreement": "verbal_agreement",

    # --- Disagreement / resistance ---
    "shaking head": "shaking_head",
    "shakes head": "shaking_head",
    "saying no": "verbal_disagreement",
    "disagrees": "verbal_disagreement",

    # --- Hands / movement (nervousness, impatience) ---
    "fidgeting": "fidgeting",
    "fidgets": "fidgeting",
    "playing with hands": "fidgeting",
    "playing with pen": "fidgeting",
    "clicking pen": "fidgeting",

    "tapping": "tapping",
    "taps fingers": "tapping",
    "finger tapping": "tapping",
    "foot tapping": "tapping",
    "bouncing leg": "tapping",
    "shaking leg": "tapping",

    "rubbing hands": "self_soothing",
    "touching face": "self_soothing",
    "touches face": "self_soothing",
    "rubs neck": "self_soothing",
    "rubbing neck": "self_soothing",

    # --- Time / exit cues ---
    "checking watch": "checking_watch",
    "looks at watch": "checking_watch",
    "checking phone": "checking_phone",
    "looks at phone": "checking_phone",
    "looking at phone": "checking_phone",
    "looking around": "looking_around",
    "glancing around": "looking_around",
    "toward the door": "exit_oriented",
    "towards the door": "exit_oriented",
    "leans toward the door": "exit_oriented",

    # --- Dominance / control ---
    "raised voice": "raised_voice",
    "speaks loudly": "raised_voice",
    "talks loudly": "raised_voice",
    "yelling": "raised_voice",

    "interrupting": "interrupting",
    "interrupts": "interrupting",
    "talks over": "interrupting",

    "pointing finger": "pointing",
    "points at": "pointing",
    "finger pointing": "pointing",

    # --- Speech pace / tone ---
    "speaks fast": "speaks_fast",
    "talks fast": "speaks_fast",
    "rapid speech": "speaks_fast",
    "speaking quickly": "speaks_fast",

    "speaks slowly": "speaks_slow",
    "talks slowly": "speaks_slow",
    "speaking slowly": "speaks_slow",

    "calm voice": "calm_voice",
    "steady voice": "calm_voice",
    "soft voice": "calm_voice",
    "voice shaking": "voice_shaky",
    "shaky voice": "voice_shaky",
}


def parse_description_to_features(description: str) -> Set[str]:
    """Extract features from a free-text description using the phrase library."""
    text = description.lower()
    features: Set[str] = set()

    for phrase, feature in KEYWORD_MAP.items():
        if phrase in text:
            features.add(feature)

    return features


# -----------------------------
# 1) Rule-based classifier (scores + rationale)
# -----------------------------

State = str


def _empty_scores() -> Dict[State, int]:
    return {
        "Dominant / Controlling": 0,
        "Impatient / Wants to end": 0,
        "Defensive / Not convinced": 0,
        "Engaged / Interested": 0,
        "Nervous / Under pressure": 0,
        "Confident / Comfortable": 0,
    }


def _add_vote(
    scores: Dict[State, int],
    rationale: Dict[State, List[str]],
    state: State,
    feature: str,
    weight: int = 1,
) -> None:
    scores[state] += weight
    rationale[state].append(feature)


def score_states(features: Set[str]) -> Tuple[Dict[State, int], Dict[State, List[str]]]:
    """Return (scores, rationale) for each state.

    rationale[state] is a list of feature names that contributed votes to that state.
    """
    feats = {f.lower() for f in features}
    scores = _empty_scores()
    rationale: Dict[State, List[str]] = {k: [] for k in scores.keys()}

    for f in feats:
        # Dominant / controlling
        if f in {"raised_voice", "interrupting", "pointing", "staring"}:
            _add_vote(scores, rationale, "Dominant / Controlling", f)

        # Impatient / wants to end
        if f in {"checking_watch", "checking_phone", "looking_around", "exit_oriented", "sighing"}:
            _add_vote(scores, rationale, "Impatient / Wants to end", f)

        # Defensive / not convinced
        if f in {
            "crossed_arms",
            "leaning_back",
            "tight_lips",
            "jaw_clenched",
            "frowning",
            "shaking_head",
            "eye_roll",
            "rigid_posture",
        }:
            _add_vote(scores, rationale, "Defensive / Not convinced", f)

        # Engaged / interested
        if f in {
            "leaning_forward",
            "good_eye_contact",
            "nodding",
            "taking_notes",
            "raised_eyebrows",
            "verbal_agreement",
        }:
            _add_vote(scores, rationale, "Engaged / Interested", f)

        # Nervous / under pressure
        if f in {
            "fidgeting",
            "tapping",
            "self_soothing",
            "weak_eye_contact",
            "looking_away",
            "speaks_fast",
            "voice_shaky",
        }:
            _add_vote(scores, rationale, "Nervous / Under pressure", f)

        # Confident / comfortable
        if f in {
            "relaxed_posture",
            "speaks_slow",
            "calm_voice",
            "smiling",
            "upright_posture",
        }:
            _add_vote(scores, rationale, "Confident / Comfortable", f)

    return scores, rationale


def classify_state(features: Set[str]) -> str:
    """Backwards-compatible: returns a single best state label."""
    pred = predict_state(features)
    return pred.primary_state


@dataclass(frozen=True)
class Prediction:
    primary_state: str
    secondary_state: str | None
    confidence: float
    scores: Dict[State, int]
    rationale: Dict[State, List[str]]
    extracted_features: Set[str]


def predict_state(features: Set[str], *, tie_margin: int = 1) -> Prediction:
    """Predict primary/secondary state + confidence + rationale.

    tie_margin:
      If the 2nd best score is within `tie_margin` of the best score,
      return a secondary label to reflect mixed signals.
    """
    scores, rationale = score_states(features)

    best_score = max(scores.values())
    if best_score == 0:
        return Prediction(
            primary_state="Neutral / Hard to read",
            secondary_state=None,
            confidence=0.0,
            scores=scores,
            rationale=rationale,
            extracted_features=set(features),
        )

    # Sort states by score desc
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_state, best = ranked[0]
    second_state, second = ranked[1]

    # Confidence: how strong the winner is relative to all votes
    total_votes = sum(scores.values())
    confidence = (best / total_votes) if total_votes > 0 else 0.0

    secondary: str | None = None
    if second > 0 and (best - second) <= tie_margin:
        secondary = second_state

    return Prediction(
        primary_state=best_state,
        secondary_state=secondary,
        confidence=round(confidence, 3),
        scores=scores,
        rationale=rationale,
        extracted_features=set(features),
    )


# -----------------------------
# 2) Map state -> perception, emotion, communication, strategy
# -----------------------------

def interpret_human_factors(state: str) -> Dict[str, str]:
    s = state.lower()

    if "defensive" in s:
        return {
            "perception": "They likely see your proposal as risky or unfair.",
            "emotion": "Anxious or resistant.",
            "communication": "Closed posture and low information sharing.",
            "strategy": "Slow down, ask open questions, use objective criteria.",
        }

    if "engaged" in s:
        return {
            "perception": "They see potential value in your proposal.",
            "emotion": "Curious and positive.",
            "communication": "Open and cooperative.",
            "strategy": "Build on agreement and move toward a concrete proposal.",
        }

    if "nervous" in s:
        return {
            "perception": "They are unsure about the consequences.",
            "emotion": "Anxious or insecure.",
            "communication": "Hesitant and unstable.",
            "strategy": "Reduce pressure, clarify information, reassure them.",
        }

    if "impatient" in s:
        return {
            "perception": "They see the discussion as too long or unproductive.",
            "emotion": "Frustrated or bored.",
            "communication": "Rushed and minimal.",
            "strategy": "Be concise and present 1–2 clear options.",
        }

    if "dominant" in s:
        return {
            "perception": "They believe they have the stronger position.",
            "emotion": "Confident or irritated.",
            "communication": "Aggressive or controlling.",
            "strategy": "Stay calm, redirect to principles and objective criteria.",
        }

    if "confident" in s:
        return {
            "perception": "They feel in control of the situation.",
            "emotion": "Calm and self-assured.",
            "communication": "Clear and steady.",
            "strategy": "Use rational, principle-based discussion to find agreement.",
        }

    return {
        "perception": "Insufficient signals to understand their view.",
        "emotion": "Emotion unclear.",
        "communication": "Low-signal or ambiguous.",
        "strategy": "Ask clarifying questions and explore their interests.",
    }


# -----------------------------
# 3) Agent wrapper (pluggable sensor)
# -----------------------------

FeatureExtractor = Callable[[str], Set[str]]


class NegotiationAgent:
    """Decision-support agent.

    Swap `feature_extractor` later if you want camera-based sensing.
    For now, default is text keyword extraction.
    """

    def __init__(self, feature_extractor: FeatureExtractor = parse_description_to_features):
        self.feature_extractor = feature_extractor

    def analyze(self, description: str) -> Prediction:
        features = self.feature_extractor(description)
        return predict_state(features)


# -----------------------------
# 4) CLI demo
# -----------------------------

def _format_rationale(rationale: Dict[State, List[str]], state: State) -> str:
    feats = rationale.get(state, [])
    if not feats:
        return "(no strong signals)"
    # show unique features, stable order
    seen = set()
    ordered = []
    for f in feats:
        if f not in seen:
            ordered.append(f)
            seen.add(f)
    return ", ".join(ordered)


def main() -> None:
    print("=== Negotiation Human-Factors AI Agent (Explainable) ===")
    print("Describe the other person's body language (free text):")
    print()

    description = input("> ")
    agent = NegotiationAgent()

    pred = agent.analyze(description)

    extracted = ", ".join(sorted(pred.extracted_features)) if pred.extracted_features else "None"
    print(f"\nExtracted features: {extracted}")

    print("\n--- Analysis ---")
    if pred.secondary_state:
        print(f"State (primary):   {pred.primary_state}")
        print(f"State (secondary): {pred.secondary_state}  (mixed signals)")
    else:
        print(f"State:             {pred.primary_state}")

    print(f"Confidence:        {pred.confidence}")

    # Score breakdown (only show non-zero to keep it readable)
    nonzero = {k: v for k, v in pred.scores.items() if v > 0}
    if nonzero:
        print("\nScore breakdown:")
        for k, v in sorted(nonzero.items(), key=lambda kv: kv[1], reverse=True):
            print(f"- {k}: {v}  | signals: {_format_rationale(pred.rationale, k)}")

    # Human factors (use primary; if mixed, primary is what to respond to first)
    factors = interpret_human_factors(pred.primary_state)
    print("\nPerception:   ", factors["perception"])
    print("Emotion:      ", factors["emotion"])
    print("Communication:", factors["communication"])
    print("Strategy:     ", factors["strategy"])

    # If nothing was detected, give a helpful prompt (prevents awkward demo failures)
    if pred.primary_state == "Neutral / Hard to read":
        print(
            "\nTip: try mentioning cues like eye contact, posture (leaning), hands (fidgeting), "
            "time signals (checking watch/phone), and voice tone (fast/slow/raised)."
        )


if __name__ == "__main__":
    main()
