"""
YouTube Data API v3 helper for the Yoga video library.

Dual-mode:
  • If YOUTUBE_API_KEY is set in .streamlit/secrets.toml → live API search
  • Otherwise → curated fallback pool with randomisation
"""

import random, requests, streamlit as st

# ── Category → search queries mapping ────────────────────────────────────────
CATEGORY_QUERIES = {
    "🌟 All": [
        "yoga for diabetes control",
        "complete yoga routine diabetes",
        "best yoga asanas blood sugar",
        "yoga for diabetic patients beginner",
    ],
    "🔥 Pancreatic Health": [
        "yoga for pancreas stimulation diabetes",
        "yoga for insulin production",
        "yoga poses pancreas health diabetes",
    ],
    "⚡ Metabolic Activation": [
        "metabolism boosting yoga diabetes",
        "surya namaskar diabetes",
        "yoga for weight loss diabetes",
    ],
    "💨 Pranayama": [
        "pranayama for diabetes control",
        "kapalbhati anulom vilom diabetes",
        "breathing exercises lower blood sugar",
    ],
    "🌙 Relaxation": [
        "yoga nidra for diabetes",
        "stress relief yoga blood sugar",
        "shavasana meditation cortisol reduction",
    ],
}

# ── Curated fallback pool (used when no API key is available) ─────────────────
_FALLBACK = {
    "🌟 All": [
        {"id": "j7rKKpwdXNE", "title": "Yoga for Diabetes — Full Routine", "channel": "Yoga With Adriene"},
        {"id": "4pKly2JojMw", "title": "20-Min Yoga for Blood Sugar Control", "channel": "SarahBethYoga"},
        {"id": "7cqzSNgNo1M", "title": "Yoga For Anxiety and Stress", "channel": "Satvic Yoga"},
        {"id": "zaAzy8DyLEw", "title": "5 Yoga Asanas For Controlling Diabetes | Yoga For Diabetes | How To Control Diabetes |", "channel": "Ventuno Yoga"},
        {"id": "ePylP2XmNRs", "title": "Diabetes Exercises At Home Workout: To Help Control Diabetes (Level 1)", "channel": "Caroline Jordan"},
        {"id": "T9vQlk2QVU4", "title": "Yoga for Diabetes | Holistic Yoga Practice with Asanas & Pranayama for Lowering Blood Sugar Levels", "channel": "Bharti Yoga"},
        {"id": "f1LvJUt9fIg", "title": "Yoga for Stress Relief: Calm Mind & Body | Saurabh Bothra Yoga", "channel": "Saurabh Bothra"},
        {"id": "hRwzELaCHDA", "title": "Yoga for Diabetes: 5 Simple Poses That Bring Blood Sugar Levels Down | Stomach & Pancreas | Asanas", "channel": "The Yoga Institute"},
        {"id": "TSnxM9DbppY", "title": "Yoga for Diabetes: The Simple Poses That Bring Blood Sugar Levels Down in 30 Minutes by Indea Yoga", "channel": "Bharatha Yoga "},
    ],
    "🔥 Pancreatic Health": [
        {"id": "GoDFqtP8zLY", "title": "Dhanurasana | Dhanurasana Variations", "channel": "VentunoYoga"},
        {"id": "ezyMaQEaVaI", "title": "Ardha Matsyendrasana — Spinal Twist", "channel": "Ventuno Yoga"},
        {"id": "PlMt-27mCP4", "title": "Paschimottanasana — Seated Forward Bend", "channel": "YogaYin"},
        {"id": "82p0aGNJSF4", "title": "Vajrasana | Thunderbolt | Diamond Pose", "channel": "NatyaSutra Yoga"},
        {"id": "9AqYH97de50", "title": "Bhujangasana — Cobra Pose Benefits", "channel": "Ventuno Yoga"},
        {"id": "6wd0fcWGOA4", "title": "Mandukasana — Frog Pose for Pancreas", "channel": "Swami Ramdev"},
    ],
    "⚡ Metabolic Activation": [
        {"id": "oBu-pQG6sTY", "title": "Surya Namaskar — 12 Steps Explained", "channel": "Yoga With Adriene"},
        {"id": "TzahhHZvOks", "title": "Viparita Karani (Legs Up the Wall Pose) ", "channel": "Siddhi Yoga International"},
        {"id": "UEEsdXn8oG8", "title": "Fat Burning Yoga Workout 20 Min", "channel": "YOGATX"},
        {"id": "9kOCY0KNByw", "title": "Power Yoga for Weight Loss", "channel": "Yoga With Tim"},
        {"id": "bPaUa4g8xHA", "title": "Boost Metabolism | Breath Work To Boost Metabolism", "channel": "Ventuno Yoga"},
        {"id": "j8bEWn2E9uo", "title": "Short Wake Up Flow - 15 Minute Morning Yoga", "channel": "Yoga With Adriene"},
    ],
    "💨 Pranayama": [
        {"id": "2li5PyxPS18", "title": "Kapalabhati Pranayama | Skull Shining Breathing Technique|", "channel": "Yoga & You"},
        {"id": "Nhw92icsQ1A", "title": "Steps to Perform Anuloma Viloma correctly I Dr. Hansaji", "channel": "The Yoga Institute"},
        {"id": "400XLjXrG0w", "title": "How to Practice Bhastrika Pranayama: Step-by-Step Guide | Bellows Breath | Yoga With Archana Alur", "channel": "Yoga With Archana Alur"},
        {"id": "blbv5UTBCGg", "title": "Pranayama For Beginners | 10 mins to release stress", "channel": "Satvic Yoga"},
        {"id": "8VwufJrUhic", "title": "5 Breathing Exercises for Diabetes", "channel": "Ventuno Yoga"},
        {"id": "LiUnFJ8P4gM", "title": "4-7-8 Calm Breathing Exercise | 10 Minutes of Deep Relaxation | Anxiety Relief | Pranayama Exercise", "channel": "Hands-On Meditation"},
    ],
    "🌙 Relaxation": [
        {"id": "TcO40hEcVl4", "title": "Guided Deep Relaxation Meditation (Savasana/Corpse Pose) 5 Minutes", "channel": "Yoga With Tim"},
        {"id": "M0u9GST_j3s", "title": "Yoga Nidra — Yogic Sleep (NSDR)", "channel": "Ally Boothroyd"},
        {"id": "Nw2oBIrQGLo", "title": "15 minute CALMING YOGA for Stress Relief and Anxiety", "channel": "SarahBethYoga"},
        {"id": "bLpChrgS0AY", "title": "Guided Meditation for Anxiety & Stress", "channel": "Goodful"},
        {"id": "1ZYbU82GVz4", "title": "Restorative Yoga for Deep Relaxation", "channel": "Yoga With Bird"},
        {"id": "aEqlQvczMJQ", "title": "Body Scan Meditation for Sleep", "channel": "The Honest Guys"},
    ],
}


# ── YouTube Data API v3 search ────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _yt_search(query: str, max_results: int = 6) -> list[dict]:
    """
    Search YouTube via Data API v3. Returns list of
    {"id": str, "title": str, "channel": str, "thumb": str}.
    """
    api_key = st.secrets.get("YOUTUBE_API_KEY", "")
    if not api_key:
        return []

    # Append exclusion terms to keep news and TV out of the results
    safe_query = f"{query} -news -tv -breaking -latest"

    try:
        resp = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "q": safe_query,
                "type": "video",
                "maxResults": max_results,
                "videoCategoryId": "26",   # How-to & Style
                "relevanceLanguage": "en",
                "key": api_key,
            },
            timeout=8,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return [
            {
                "id": it["id"]["videoId"],
                "title": it["snippet"]["title"],
                "channel": it["snippet"]["channelTitle"],
                "thumb": it["snippet"]["thumbnails"]["high"]["url"],
            }
            for it in items
        ]
    except Exception:
        return []


# ── Public API ────────────────────────────────────────────────────────────────
def fetch_videos(category: str, search_query: str = "", max_results: int = 6, force_fallback: bool = False) -> list[dict]:
    """
    Fetch videos for a category (or free-text search). Returns list of dicts with
    keys: id, title, channel, thumb.
    """
    if force_fallback:
        # Bypass API and use curated pool directly
        pool = list(_FALLBACK.get(category, _FALLBACK["🌟 All"]))
        random.shuffle(pool)
        return pool[:max_results]

    # 1) If user typed a search query, override category
    if search_query.strip():
        results = _yt_search(search_query.strip(), max_results)
        if results:
            return results
        # Fallback: return random mix from all pools
        pool = []
        for v in _FALLBACK.values():
            pool.extend(v)
        random.shuffle(pool)
        return pool[:max_results]

    # 2) Category-based
    queries = CATEGORY_QUERIES.get(category, CATEGORY_QUERIES["🌟 All"])
    query = random.choice(queries)

    results = _yt_search(query, max_results)
    if results:
        return results

    # 3) Fallback to curated pool
    pool = list(_FALLBACK.get(category, _FALLBACK["🌟 All"]))
    random.shuffle(pool)
    return pool[:max_results]


def get_personalized_category(session_state: dict) -> str | None:
    """
    Based on user health data in session state, suggest which category to
    auto-select. Returns None if no health data found.
    """
    glucose = session_state.get("Glucose", 0)
    bmi     = session_state.get("BMI", 0)

    if glucose > 140:
        return "🔥 Pancreatic Health"
    if bmi > 30:
        return "⚡ Metabolic Activation"
    # Could add stress indicators here in the future
    return None
