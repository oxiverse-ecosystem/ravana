import ast
ast.parse(open('ravana/src/ravana/chat/engine.py').read())
print('engine.py: SYNTAX OK')
ast.parse(open('ravana/src/ravana/chat/engine_self_query.py').read())
print('engine_self_query.py: SYNTAX OK')
