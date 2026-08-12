import streamlit as st
from negotiation_agent import (
    classify_state,
    interpret_human_factors,
    parse_description_to_features,
)

st.set_page_config(page_title="Negotiation HUD", layout="centered")

st.title("🕶️ Negotiation HUD")
st.caption("Class inputs body-language cues → AI agent classifies state → outputs strategy.")

mode = st.radio("Input mode", ["Checklist (recommended)", "Free text"], horizontal=True)

features = set()

# -------- Checklist Mode --------
if mode == "Checklist (recommended)":
    st.subheader("Select observed cues")

    # These are FEATURE NAMES your classifier already understands.
    # Add/remove items freely as long as they match your classify_state rules.
    checklist = {
        "Arms crossed": "crossed_arms",
        "Leaning back": "leaning_back",
        "Leaning forward": "leaning_forward",
        "Good eye contact": "good_eye_contact",
        "Weak / avoids eye contact": "weak_eye_contact",
        "Nodding": "nodding",
        "Fidgeting": "fidgeting",
        "Checking watch": "checking_watch",
        "Looking around": "looking_around",
        "Raised voice": "raised_voice",
        "Interrupting": "interrupting",
        "Relaxed posture": "relaxed_posture",
        "Speaks slowly": "speaks_slow",
        "Speaks fast": "speaks_fast",
    }

    col1, col2 = st.columns(2)
    items = list(checklist.items())
    half = (len(items) + 1) // 2

    for i, (label, feat) in enumerate(items):
        col = col1 if i < half else col2
        with col:
            if st.checkbox(label):
                features.add(feat)

# -------- Free Text Mode --------
else:
    st.subheader("Describe body language in your own words")
    desc = st.text_area(
        "Example: 'He is leaning back with crossed arms and avoids eye contact.'",
        height=120,
    )
    features = parse_description_to_features(desc) if desc.strip() else set()

# -------- Analyze --------
st.divider()

if st.button("ANALYZE"):
    if not features:
        st.warning("No cues detected. Select at least 1 checkbox or add more detail in text.")
    else:
        state = classify_state(features)
        factors = interpret_human_factors(state)

        st.success(f"**State:** {state}")
        st.write("**Extracted features:**", ", ".join(sorted(features)))

        st.subheader("Interpretation")
        st.write(f"**Perception:** {factors['perception']}")
        st.write(f"**Emotion:** {factors['emotion']}")
        st.write(f"**Communication:** {factors['communication']}")
        st.write(f"**Strategy:** {factors['strategy']}")
