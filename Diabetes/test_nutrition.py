from nutrition_analyzer import fetch_nutrition, classify_diet

r = fetch_nutrition('50g dahi and 200g dal and half cup rice and 2 chapati and 1 glass milk and 100g chicken')
print("=" * 80)
print(f"{'Food Item':<35} {'Wt(g)':>6} {'Cal':>7} {'Carbs':>6} {'Prot':>6} {'Fat':>6}")
print("-" * 80)
for i in r['items']:
    print(f"{i[0]:<35} {i[1]:>6.0f} {i[2]:>7.1f} {i[3]:>6.1f} {i[4]:>6.1f} {i[5]:>6.1f}")
print("-" * 80)
print(f"{'TOTAL':<35} {'':>6} {r['calories']:>7.0f} {r['total_carbs']:>6.1f} {r['total_protein']:>6.1f} {r['total_fat']:>6.1f}")
c = classify_diet(r['total_carbs'], r['total_protein'], r['total_fat'])
print(f"\nDiet Type: {c['diet_type']}")
print(f"Carbs: {c['carb_ratio']*100:.0f}% | Protein: {c['protein_ratio']*100:.0f}% | Fat: {c['fat_ratio']*100:.0f}%")
if r.get('errors'):
    print(f"\nUnrecognized: {r['errors']}")
