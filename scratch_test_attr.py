import sys
sys.path.insert(0, 'ravana/src')
from ravana.chat.attribute_gate import asks_name_only
print('what is my dog name:', asks_name_only('what is my dog name'))
print('what is my dog called:', asks_name_only('what is my dog called'))
print('what is my dog name and what does it do:', asks_name_only('what is my dog name and what does it do'))
