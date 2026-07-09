import sys, os
sys.path.insert(0, 'ravana-v2')
sys.path.insert(0, '.')
os.environ['RAVANA_SILENT'] = '1'

from scripts.ravana_chat import CognitiveChatEngine

print('Loading engine...')
engine = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
engine._network_available = False  # Skip web search to avoid network/DNS hangs in offline sandbox environments
print('Engine loaded!')

print('Decoder training count:', engine._decoder_training_count)
print('Decoder web training count:', engine._decoder_web_training_count)

# Test a simple query - decoder not ready yet (505 < 500)
print('\n--- Testing simple query (decoder not ready) ---')
response = engine.process_turn('what is trust')
print('Response:', response)
print('Strategy:', engine._last_strategy)

# Test complex query (should trigger reasoning loop and web search)
print('\n--- Testing complex query ---')
response = engine.process_turn('how does a neural network work')
print('Response:', response[:500] if len(response) > 500 else response)
print('Strategy:', engine._last_strategy)

print('\nDecoder training count after:', engine._decoder_training_count)
print('Decoder web training count after:', engine._decoder_web_training_count)