#!/usr/bin/env python3
"""Apply remaining curiosity drive edits (save, load, CLI) to scripts/ravana_chat.py."""
with open('scripts/ravana_chat.py', 'r', encoding='utf-8') as f:
    code = f.read()

changes = 0

# 5. Add curiosity state to save() - find 'metacognitive_review_turn' in dict context
pos = code.find("'metacognitive_review_turn': self._metacognitive_review_turn,")
if pos >= 0:
    end = code.find("\n", pos)
    after = code[end:end+50]
    # Look for '# Background learning' comment after
    bg_pos = code.find("# Background web learning", pos)
    if bg_pos < 0:
        bg_pos = code.find("# Background learning", pos)
    if bg_pos >= 0:
        insert = """            # Curiosity Drive state
            'curiosity_drive_enabled': self._curiosity_drive_enabled,
            'concept_visit_count': self._concept_visit_count,
            'concept_learning_progress': self._concept_learning_progress,
            'curiosity_topics_queue': self._curiosity_topics_queue,
            'last_auto_learn_turn': self._last_auto_learn_turn,
            'curiosity_urgency': self._curiosity_urgency,
            """
        code = code[:bg_pos] + insert + code[bg_pos:]
        print("5. Added curiosity state to save()")
        changes += 1
    else:
        print("5. FAILED - Could not find 'Background learning' comment")
else:
    print("5. FAILED - Could not find save() dict entry for metacognitive_review_turn")

# 6. Restore curiosity state in _load()
pos = code.find("self._metacognitive_review_turn = state.get('metacognitive_review_turn', 0)")
if pos >= 0:
    eol = code.find("\n", pos)
    eol = code.find("\n", eol + 1)  # next line
    insert = """
        # Restore curiosity drive state
        self._curiosity_drive_enabled = state.get('curiosity_drive_enabled', True)
        self._concept_visit_count = state.get('concept_visit_count', {})
        self._concept_learning_progress = state.get('concept_learning_progress', {})
        self._curiosity_topics_queue = state.get('curiosity_topics_queue', [])
        self._last_auto_learn_turn = state.get('last_auto_learn_turn', 0)
        self._curiosity_urgency = state.get('curiosity_urgency', 0.0)
"""
    code = code[:eol] + insert + code[eol:]
    print("6. Restored curiosity state in _load()")
    changes += 1
else:
    print("6. FAILED - Could not find _load() anchor for metacognitive_review_turn")

# 7. Add --no-curiosity CLI flag (file uses double quotes)
pos = code.find('"--no-beliefs", action="store_true", help="Disable belief store"')
if pos >= 0:
    eol = code.find("\n", pos)
    insert = '\n    parser.add_argument("--no-curiosity", action="store_true", help="Disable autonomous curiosity-driven learning")'
    code = code[:eol] + insert + code[eol:]
    print("7. Added --no-curiosity CLI flag")
    changes += 1
else:
    print("7. FAILED - Could not find CLI flag anchor")

# 8. Apply the --no-curiosity flag in main()
pos = code.find("engine.use_beliefs = False")
if pos >= 0:
    eol = code.find("\n", pos)
    eol = code.find("\n", eol + 1)  # 'if args.trace:' line
    insert = """    if args.no_curiosity:
        engine._curiosity_drive_enabled = False
        print('  [Curiosity] Autonomous learning disabled')
"""
    code = code[:eol] + insert + code[eol:]
    print("8. Applied --no-curiosity flag in main()")
    changes += 1
else:
    print("8. FAILED - Could not find main() flag application anchor")

with open('scripts/ravana_chat.py', 'w', encoding='utf-8') as f:
    f.write(code)

print(f"\nApplied {changes}/4 remaining edits!")
