#!/usr/bin/env python3
"""Fix missing fallback_idea variable in _format_contrastive."""

with open('scripts/ravana_chat.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the exact line after the deterministic return and before templates
old = '            return f"{starter.capitalize()}, {before_ant} {ant} but they shape how we understand."\n\n        # Stochastic: variety of contrastive structures\n        templates = ['

new = '            return f"{starter.capitalize()}, {before_ant} {ant} but they shape how we understand."\n\n        fallback_idea = "ideas"\n        # Stochastic: variety of contrastive structures\n        templates = ['

count = content.count(old)
print(f'Found {count} occurrence(s)')
if count == 1:
    content = content.replace(old, new, 1)
    with open('scripts/ravana_chat.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Added fallback_idea = "ideas"')
else:
    print('Could not find exact match. Searching by parts...')
    # Try finding the deterministic return line
    idx = content.find('but they shape how we understand')
    if idx >= 0:
        chunk = content[idx:idx+130]
        print(f'Found at {idx}:')
        print(repr(chunk))
