import ast, sys
for f in ['ravana/src/ravana/chat/engine.py', 'ravana/src/ravana/chat/engine_self_query.py', 'ravana/src/ravana/chat/user_model.py']:
    try:
        ast.parse(open(f, encoding='utf-8').read())
        print(f'{f} OK')
    except SyntaxError as e:
        print(f'{f} SYNTAX ERROR: {e}')
        sys.exit(1)
