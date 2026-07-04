import sys, os, time
sys.path.insert(0, '.')
sys.path.insert(0, 'ravana/src')
sys.path.insert(0, 'ravana_ml/src')
os.environ['RAVANA_SILENT'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

print('Step 1: importing...', flush=True)
t0 = time.time()
try:
    from scripts.ravana_chat import CognitiveChatEngine
    print(f'Step 2: imported in {time.time()-t0:.1f}s', flush=True)
except Exception as e:
    print(f'Import error: {e}', flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('Step 3: creating engine...', flush=True)
t0 = time.time()
try:
    engine = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
    print(f'Step 4: engine created in {time.time()-t0:.1f}s', flush=True)
except Exception as e:
    print(f'Engine error: {e}', flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('Done!', flush=True)
