"""Audit test coverage: compare source modules against test imports."""
import os, glob, re, sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_source_modules(root_dir):
    """Get dict of {module_filepath: module_basename} for source .py files."""
    modules = {}
    for f in sorted(glob.glob(os.path.join(root_dir, '**/*.py'), recursive=True)):
        bn = os.path.basename(f)
        if bn == '__init__.py':
            continue
        rel = os.path.relpath(f, PROJECT)
        modules[rel] = bn.replace('.py', '')
    return modules

def get_test_imports():
    """Get dict of {test_file: [list_of_imported_package_modules]}."""
    test_dir = os.path.join(PROJECT, 'tests')
    results = {}
    for tf in sorted(glob.glob(os.path.join(test_dir, '**/*.py'), recursive=True)):
        bn = os.path.basename(tf)
        if bn in ('__init__.py', 'conftest.py'):
            continue
        rel = os.path.relpath(tf, PROJECT)
        try:
            with open(tf) as f:
                content = f.read()
        except:
            continue
        # Find all imports from ravana packages
        imports = set()
        for m in re.findall(r'from\s+(ravana[^\s]+)', content):
            imports.add(m.split('.')[0])
        for m in re.findall(r'import\s+(ravana[^\s]+)', content):
            parts = m.split('.')[0].split(' ')
            for p in parts:
                if p.startswith('ravana'):
                    imports.add(p)
        results[rel] = sorted(imports)
    return results

# Collect sources
print("=" * 70)
print("COVERAGE AUDIT: Source modules vs test files")
print("=" * 70)

pkgs = {
    'ravana_ml': 'ravana_ml/src/ravana_ml',
    'ravana': 'ravana/src/ravana',
    'ravana_grace (core)': 'ravana-v2/src/ravana_grace/core',
    'ravana_grace (agent)': 'ravana-v2/src/ravana_grace/agent',
    'ravana_grace (dialogue)': 'ravana-v2/src/ravana_grace/dialogue',
    'ravana_grace (probes)': 'ravana-v2/src/ravana_grace/probes',
    'ravana_grace (training)': 'ravana-v2/src/ravana_grace/training',
    'ravana_grace (interface_agent)': 'ravana-v2/src/ravana_grace/interface_agent',
}

all_sources = {}
for name, rel_dir in pkgs.items():
    full_dir = os.path.join(PROJECT, rel_dir)
    if os.path.isdir(full_dir):
        modules = get_source_modules(full_dir)
        all_sources[name] = modules

# Get test imports
test_imports = get_test_imports()

# Also check which tests import from which packages
tests_by_package = {}
for tf, imps in test_imports.items():
    for imp in imps:
        tests_by_package.setdefault(imp, []).append(tf)

# Analyze coverage
print("\n1. Top-level package coverage (which packages have tests importing them)")
print("-" * 50)
for pkg in sorted(tests_by_package.keys()):
    print(f"  {pkg}: {len(tests_by_package[pkg])} test files")

print("\n2. Module-level coverage analysis")
print("-" * 50)

# For each package, list source modules and whether they have a dedicated test
for pkg_name in sorted(all_sources.keys()):
    modules = all_sources[pkg_name]
    if not modules:
        continue
    print(f"\n  [{pkg_name}] ({len(modules)} source modules)")
    
    # Get test files that import this package
    for pkg_key, test_files in tests_by_package.items():
        if pkg_key == pkg_name.split()[0]:
            print(f"    Tests importing this package: {len(test_files)} files")
            break

    # Simple heuristic: check if any test file basename matches module name
    test_basenames = set()
    for tf in test_imports:
        bn = os.path.basename(tf)
        if bn not in ('__init__.py', 'conftest.py'):
            name = bn.replace('.py', '')
            test_basenames.add(name)
    
    uncovered = []
    for rel, mod_name in modules.items():
        # Check common patterns
        has_test = False
        for tb in test_basenames:
            # test_rlm_v1 covers rlm, test_free_energy covers tensor, etc.
            if mod_name in tb.replace('test_', '').replace('_', ''):
                has_test = True
                break
        if not has_test:
            uncovered.append(rel)
    
    if uncovered:
        print(f"    POTENTIALLY UNCOVERED ({len(uncovered)} modules):")
        for u in uncovered[:15]:
            print(f"      - {u}")
    else:
        print(f"    All modules have matching test patterns")

print("\n3. CI Integration tests")
print("-" * 50)
ci_tests = [f for f in sorted(test_imports.keys()) if f.startswith('tests/ci')]
int_tests = [f for f in sorted(test_imports.keys()) if f.startswith('tests/integration')]
print(f"  CI tests: {len(ci_tests)} files - {[os.path.basename(f) for f in ci_tests]}")
print(f"  Integration tests: {len(int_tests)} files - {[os.path.basename(f) for f in int_tests]}")

print("\n4. Summary")
print("-" * 50)
test_files = [f for f in test_imports if not f.startswith('tests/ci') and not f.startswith('tests/integration')]
print(f"  Total unit test files: {len(test_files)}")
print(f"  Total integration test files: {len(int_tests)}")
print(f"  Total CI test files: {len(ci_tests)}")
print(f"  Packages with test imports: {sorted(tests_by_package.keys())}")

# Check if ravana_chat_src and ravana_ml are tested
if 'ravana_ml' in tests_by_package:
    print(f"  ravana_ml: YES (imported by tests)")
if 'ravana' in tests_by_package:
    print(f"  ravana: YES (imported by tests)")
if 'ravana_grace' in tests_by_package:
    print(f"  ravana_grace: YES (imported by tests)")

print("\n  Key remaining gaps to investigate:")
print("  - ravana_chat_src package (separate chat engine)")
print("  - ravana-v2 interface agent scripts")
print("  - ravana-v2 research code")
