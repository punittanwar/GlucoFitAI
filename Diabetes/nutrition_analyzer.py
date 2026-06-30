import requests
import re

# ── Built-in food database — values are PER 100g ────────────────────────────
# Format: {food_name: [cal, carbs_g, protein_g, fat_g]}
FOOD_DB_PER_100G = {
    # Indian breads (per 100g)
    'chapati':     [240, 41.0, 7.1, 6.2],
    'chapatti':    [240, 41.0, 7.1, 6.2],
    'roti':        [240, 41.0, 7.1, 6.2],
    'naan':        [291, 50.0, 9.6, 5.7],
    'paratha':     [289, 40.0, 6.1, 12.2],
    'puri':        [375, 45.0, 6.3, 18.8],
    'idli':        [100, 20.3, 4.9, 0.3],
    'dosa':        [195, 27.9, 5.2, 7.0],
    'uttapam':     [180, 26.0, 5.0, 6.0],
    # Rice & grains (per 100g cooked)
    'rice':        [130, 28.2, 2.7, 0.3],
    'biryani':     [145, 19.0, 7.5, 4.0],
    'khichdi':     [100, 16.0, 3.8, 2.5],
    'upma':        [118, 19.3, 3.3, 3.3],
    'poha':        [180, 36.0, 3.5, 3.5],
    'oats':        [68,  12.0, 2.4, 1.4],
    'cornflakes':  [357, 84.0, 7.5, 0.7],
    'pasta':       [157, 30.7, 5.0, 1.1],
    'noodles':     [138, 25.0, 4.5, 2.0],
    # Lentils & legumes (per 100g cooked)
    'dal':         [58,  10.0, 3.8, 0.2],
    'toor dal':    [58,  10.0, 3.8, 0.2],
    'moong dal':   [53,   9.0, 3.5, 0.2],
    'chana':       [164, 27.0, 8.9, 2.6],
    'rajma':       [127, 22.8, 8.7, 0.5],
    'sambar':      [25,   3.8, 1.3, 0.8],
    'chole':       [164, 27.0, 8.9, 2.6],
    # Dairy (per 100g)
    'milk':        [61,   4.8, 3.2, 3.3],
    'dahi':        [61,   4.7, 3.4, 3.3],
    'curd':        [61,   4.7, 3.4, 3.3],
    'yogurt':      [61,   4.7, 3.4, 3.3],
    'raita':       [55,   4.5, 2.8, 3.0],
    'paneer':      [265,  3.4, 18.3, 20.8],
    'cheese':      [402,  1.3, 25.0, 33.1],
    'ghee':        [900,  0.0, 0.0, 99.9],
    'butter':      [717,  0.1, 0.9, 81.1],
    'cream':       [195,  3.7, 2.8, 19.1],
    'lassi':       [67,  11.0, 2.0, 1.5],
    # Beverages (per 100ml ≈ 100g)
    'chai':        [30,   3.7, 1.2, 1.3],
    'tea':         [30,   3.7, 1.2, 1.3],
    'coffee':      [20,   3.3, 0.2, 0.3],
    'juice':       [45,  11.0, 0.4, 0.1],
    'buttermilk':  [40,   4.8, 3.3, 0.9],
    # Vegetables (per 100g cooked)
    'sabzi':       [45,   6.0, 1.5, 2.0],
    'aloo':        [77,  17.0, 2.0, 0.1],
    'potato':      [77,  17.0, 2.0, 0.1],
    'bhaji':       [90,   8.0, 2.0, 5.5],
    'palak':       [23,   3.6, 2.9, 0.4],
    'spinach':     [23,   3.6, 2.9, 0.4],
    'gobi':        [25,   5.3, 1.9, 0.3],
    'cauliflower': [25,   5.3, 1.9, 0.3],
    'brinjal':     [25,   5.9, 1.0, 0.2],
    'bhindi':      [33,   7.4, 1.9, 0.2],
    'okra':        [33,   7.4, 1.9, 0.2],
    'carrot':      [41,   9.6, 0.9, 0.2],
    'beans':       [31,   7.0, 1.8, 0.1],
    'peas':        [81,  14.0, 5.4, 0.4],
    'onion':       [40,   9.3, 1.1, 0.1],
    'tomato':      [18,   3.9, 0.9, 0.2],
    'cucumber':    [16,   3.6, 0.7, 0.1],
    'mushroom':    [22,   3.3, 3.1, 0.3],
    'capsicum':    [20,   4.6, 0.9, 0.2],
    'cabbage':     [25,   5.8, 1.3, 0.1],
    'salad':       [15,   2.5, 0.9, 0.2],
    # Non-veg (per 100g cooked)
    'egg':         [155,  1.1, 13.0, 11.0],
    'chicken':     [239,  0.0, 27.0, 14.0],
    'mutton':      [294,  0.0, 25.0, 21.0],
    'fish':        [206,  0.0, 22.0, 12.0],
    'prawn':       [99,   0.2, 24.0, 0.3],
    'shrimp':      [99,   0.2, 24.0, 0.3],
    'keema':       [250,  0.0, 20.0, 18.0],
    # Fruits (per 100g)
    'banana':      [89,  23.0, 1.1, 0.3],
    'apple':       [52,  14.0, 0.3, 0.2],
    'orange':      [47,  12.0, 0.9, 0.1],
    'mango':       [60,  15.0, 0.8, 0.4],
    'papaya':      [43,  11.0, 0.5, 0.3],
    'grapes':      [69,  18.0, 0.7, 0.2],
    'watermelon':  [30,   7.6, 0.6, 0.2],
    'pomegranate': [83,  19.0, 1.7, 1.2],
    'guava':       [68,  14.3, 2.6, 1.0],
    'pineapple':   [50,  13.0, 0.5, 0.1],
    # Snacks & sweets (per 100g)
    'samosa':      [262, 32.0, 4.5, 13.0],
    'pakora':      [225, 25.0, 5.0, 12.0],
    'vada':        [290, 25.0, 5.5, 19.0],
    'biscuit':     [450, 65.0, 6.0, 18.0],
    'pizza':       [266, 33.0, 11.0, 10.0],
    'burger':      [295, 24.0, 17.0, 14.0],
    'sandwich':    [210, 28.0, 9.0, 7.0],
    'cake':        [350, 50.0, 5.0, 15.0],
    'chocolate':   [546, 60.0, 5.0, 31.0],
    'ladoo':       [400, 50.0, 7.0, 19.0],
    'jalebi':      [380, 60.0, 2.0, 15.0],
    'halwa':       [300, 40.0, 3.0, 15.0],
    'gulab jamun': [325, 42.0, 5.0, 16.0],
    'kheer':       [160, 22.0, 4.0, 6.5],
    # Basics (per 100g)
    'bread':       [265, 49.0, 9.0, 3.2],
    'sugar':       [387, 100., 0.0, 0.0],
    'honey':       [304, 82.0, 0.3, 0.0],
    'oil':         [884,  0.0, 0.0, 100.],
    'nuts':        [607, 20.0, 20.0, 50.0],
    'almonds':     [579, 22.0, 21.0, 50.0],
    'peanuts':     [567, 16.0, 26.0, 49.0],
    'soup':        [28,   3.8, 1.3, 0.8],
    'water':       [0,    0.0, 0.0, 0.0],
    'salt':        [0,    0.0, 0.0, 0.0],
}

# ── Standard serving sizes (in grams) ───────────────────────────────────────
# Default serving weight per item when user says "1 chapati" (no grams given)
DEFAULT_SERVING_G = {
    'chapati': 45, 'chapatti': 45, 'roti': 45, 'naan': 90, 'paratha': 90,
    'puri': 40, 'idli': 39, 'dosa': 86, 'uttapam': 100,
    'rice': 186, 'biryani': 200, 'khichdi': 200, 'upma': 150, 'poha': 100,
    'oats': 150, 'pasta': 140, 'noodles': 100,
    'dal': 200, 'toor dal': 200, 'moong dal': 200, 'sambar': 200,
    'chana': 100, 'rajma': 100, 'chole': 100,
    'milk': 200, 'chai': 150, 'tea': 150, 'coffee': 150, 'juice': 200,
    'lassi': 200, 'buttermilk': 200,
    'dahi': 100, 'curd': 100, 'yogurt': 100, 'raita': 100,
    'paneer': 75, 'cheese': 30, 'ghee': 10, 'butter': 10, 'cream': 30,
    'egg': 50, 'chicken': 120, 'mutton': 100, 'fish': 100,
    'prawn': 100, 'shrimp': 100, 'keema': 100,
    'banana': 118, 'apple': 182, 'orange': 131, 'mango': 150,
    'samosa': 80, 'pakora': 50, 'vada': 60,
    'bread': 30, 'biscuit': 15,
    'pizza': 107, 'burger': 150, 'sandwich': 120,
    'soup': 250, 'salad': 150,
}

# ── Volume conversions to grams (approximate) ──────────────────────────────
VOLUME_TO_G = {
    'cup': 200, 'cups': 200,
    'bowl': 250, 'bowls': 250,
    'katori': 150, 'katoris': 150,
    'glass': 250, 'glasses': 250,
    'plate': 300, 'plates': 300,
    'slice': 30, 'slices': 30,
    'piece': 50, 'pieces': 50,
    'tbsp': 15, 'tablespoon': 15, 'tablespoons': 15,
    'tsp': 5, 'teaspoon': 5, 'teaspoons': 5,
    'handful': 30,
    'litre': 1000, 'liter': 1000, 'l': 1000,
    'ml': 1,
}

QUANTITY_WORDS = {
    'half': 0.5, 'quarter': 0.25, 'one': 1, 'two': 2, 'three': 3,
    'four': 4, 'five': 5, 'six': 6, 'a': 1, 'an': 1,
}


def _parse_item(item_str: str):
    """
    Parse a food item string and return (food_name, weight_in_grams).
    Handles formats like:
      '50g dahi'         → ('dahi', 50)
      '200g dal'         → ('dal', 200)
      'half cup rice'    → ('rice', 100)
      '2 chapati'        → ('chapati', 90)   # 2 × 45g default serving
      '1 bowl dal'       → ('dal', 250)
      '3 idli'           → ('idli', 117)     # 3 × 39g
      'paneer'           → ('paneer', 75)    # 1 × default serving
    """
    raw = item_str.strip().lower().replace(',', '')

    # ── Pattern 1: explicit grams  "50g dahi", "200 g dal", "50 gm rice"
    m = re.match(r'^(\d+(?:\.\d+)?)\s*(?:g|gm|gms|gram|grams)\s+(.+)', raw)
    if m:
        grams = float(m.group(1))
        food = m.group(2).strip()
        return food, grams

    # ── Pattern 2: explicit ml  "200ml milk"
    m = re.match(r'^(\d+(?:\.\d+)?)\s*ml\s+(.+)', raw)
    if m:
        grams = float(m.group(1))  # 1ml ≈ 1g for most foods
        food = m.group(2).strip()
        return food, grams

    # ── Pattern 3: word/number quantity + optional volume + food
    #    "half cup rice", "2 bowl dal", "3 chapati", "1 glass milk"
    qty = 1.0
    rest = raw

    # Extract leading number
    m = re.match(r'^(\d+(?:\.\d+)?)\s+(.*)', rest)
    if m:
        qty = float(m.group(1))
        rest = m.group(2).strip()
    else:
        # Try word quantity
        for word, val in sorted(QUANTITY_WORDS.items(), key=lambda x: -len(x[0])):
            if rest.startswith(word + ' '):
                qty = val
                rest = rest[len(word):].strip()
                break

    # Check for volume unit
    parts = rest.split(None, 1)
    if len(parts) >= 2 and parts[0] in VOLUME_TO_G:
        volume_unit = parts[0]
        food = parts[1].strip()
        grams = qty * VOLUME_TO_G[volume_unit]
        return food, grams

    # No volume unit → use count × default serving
    food = rest.strip()
    matched_key = _match_food(food)
    if matched_key:
        default_g = DEFAULT_SERVING_G.get(matched_key, 100)
        return food, qty * default_g

    return food, qty * 100  # Unknown food, assume 100g per serving


def _match_food(food_str: str):
    """Find the best matching key in FOOD_DB_PER_100G."""
    food_str = food_str.lower().strip()
    # Exact match
    if food_str in FOOD_DB_PER_100G:
        return food_str
    # Partial match — longest key that appears in the string
    best_key, best_len = None, 0
    for key in FOOD_DB_PER_100G:
        if key in food_str and len(key) > best_len:
            best_key, best_len = key, len(key)
    return best_key


def _calc_from_db(food_key: str, grams: float):
    """Calculate nutrition for given grams of a food item."""
    per100 = FOOD_DB_PER_100G[food_key]
    factor = grams / 100.0
    return [per100[0] * factor,  # calories
            per100[1] * factor,  # carbs
            per100[2] * factor,  # protein
            per100[3] * factor]  # fat


def fetch_nutrition(food_input: str, app_id: str = None, app_key: str = None) -> dict:
    """Fetch nutrition: tries Edamam API first, falls back to built-in DB."""
    url = "https://api.edamam.com/api/nutrition-data"
    cleaned = food_input.replace('\n', ' and ').replace('\r', '')
    items = [i.strip() for i in cleaned.split(' and ') if i.strip()]
    total = {'calories': 0.0, 'total_carbs': 0.0, 'total_protein': 0.0,
             'total_fat': 0.0, 'items': [], 'errors': []}

    for item in items:
        fetched = False

        # 1️⃣ Try Edamam API first
        if app_id and app_key:
            try:
                r = requests.get(url, params={
                    'app_id': app_id, 'app_key': app_key, 'ingr': item
                }, timeout=8)
                if r.status_code == 200:
                    d = r.json()
                    cal  = float(d.get('calories', 0) or 0)
                    carb = float((d.get('totalNutrients', {}).get('CHOCDF', {}).get('quantity', 0) or 0))
                    prot = float((d.get('totalNutrients', {}).get('PROCNT', {}).get('quantity', 0) or 0))
                    fat  = float((d.get('totalNutrients', {}).get('FAT',   {}).get('quantity', 0) or 0))
                    wt   = float((d.get('totalWeight', 0) or 0))
                    if cal > 0:
                        total['calories'] += cal
                        total['total_carbs'] += carb
                        total['total_protein'] += prot
                        total['total_fat'] += fat
                        total['items'].append([item, round(wt, 1), round(cal, 1),
                                               round(carb, 1), round(prot, 1), round(fat, 1)])
                        fetched = True
            except Exception:
                pass

        # 2️⃣ Fallback to built-in DB with quantity parsing
        if not fetched:
            food_str, grams = _parse_item(item)
            food_key = _match_food(food_str)
            if food_key:
                cal, carb, prot, fat = _calc_from_db(food_key, grams)
                total['calories'] += cal
                total['total_carbs'] += carb
                total['total_protein'] += prot
                total['total_fat'] += fat
                label = f"{item} ({grams:.0f}g)"
                total['items'].append([label, round(grams, 1), round(cal, 1),
                                       round(carb, 1), round(prot, 1), round(fat, 1)])
                fetched = True

        if not fetched:
            total['errors'].append(f"'{item}' not recognized")

    return total if (total['calories'] > 0 or total['errors']) else None


def classify_diet(total_carbs: float, total_protein: float, total_fat: float) -> dict:
    total_macro = total_carbs + total_protein + total_fat
    if total_macro == 0:
        return {'diet_type': 'Balanced Diet', 'carb_ratio': 0.0, 'protein_ratio': 0.0, 'fat_ratio': 0.0}
    carb_r = total_carbs   / total_macro
    prot_r = total_protein / total_macro
    fat_r  = total_fat     / total_macro
    if carb_r > 0.55:   diet_type = 'Carbohydrate Rich Diet'
    elif prot_r > 0.35: diet_type = 'Protein Rich Diet'
    elif fat_r  > 0.40: diet_type = 'Fat Rich Diet'
    else:               diet_type = 'Balanced Diet'
    return {'diet_type': diet_type, 'carb_ratio': carb_r, 'protein_ratio': prot_r, 'fat_ratio': fat_r}


def diet_type_to_model_value(diet_type: str) -> int:
    return {'Carbohydrate Rich Diet': 0, 'Protein Rich Diet': 1, 'Fat Rich Diet': 2, 'Balanced Diet': 3}.get(diet_type, 3)


def diet_type_to_lifestyle_label(diet_type: str) -> str:
    return {'Carbohydrate Rich Diet': 'High Carbs', 'Protein Rich Diet': 'Balanced', 'Fat Rich Diet': 'High Fat', 'Balanced Diet': 'Balanced'}.get(diet_type, 'Balanced')


def manual_label_to_classifier_label(label: str) -> str:
    return {'High Carbs': 'Carbohydrate Rich Diet', 'High Fat': 'Fat Rich Diet', 'Low Protein': 'Protein Rich Diet', 'Balanced': 'Balanced Diet'}.get(label, 'Balanced Diet')
