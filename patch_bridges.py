path = r"C:\Users\Likhith\Documents\projects\ravana\experiments\experiment_cross_domain.py"
with open(path, 'r') as f:
    content = f.read()

# Fix the bridge injection - make them truly cross-domain
old_bridges = '''    # "anger" -> is -> "intense_bridge" -> causes -> "expansion"
    add_abstract_bridge(model, "intense_bridge", "anger", "expansion", "causal", weight=0.8)
    # "kindness" -> is -> "warm_bridge" -> causes -> "trust"
    add_abstract_bridge(model, "warm_bridge", "kindness", "trust", "causal", weight=0.8)'''

new_bridges = '''    # Cross-domain bridges: B subject -> bridge -> A target (causal)
    # "anger" (B) -> intense_bridge -> "expansion" (A)
    add_abstract_bridge(model, "intense_bridge", "anger", "expansion", "causal", weight=0.8)
    # "kindness" (B) -> warm_bridge -> "warmth" (A)
    add_abstract_bridge(model, "warm_bridge", "kindness", "warmth", "causal", weight=0.8)
    # "sadness" (B) -> cold_bridge -> "freezing" (A) - but freezing not in vocab, use "ice"
    add_abstract_bridge(model, "cold_bridge", "sadness", "ice", "causal", weight=0.8)
    # "generosity" (B) -> give_bridge -> "growth" (A)
    add_abstract_bridge(model, "give_bridge", "generosity", "growth", "causal", weight=0.8)

    # Also A -> B bridges for bidirectional
    # "heat" (A) -> fire_bridge -> "conflict" (B)
    add_abstract_bridge(model, "fire_bridge", "heat", "conflict", "causal", weight=0.8)
    # "light" (A) -> insight_bridge -> "understanding" (B)
    add_abstract_bridge(model, "insight_bridge", "light", "understanding", "causal", weight=0.8)'''

content = content.replace(old_bridges, new_bridges)

with open(path, 'w') as f:
    f.write(content)

print("Fixed bridge connections for true cross-domain")