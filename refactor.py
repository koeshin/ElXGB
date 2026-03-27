import os
import shutil
import glob

dirs = ['core', 'crypto', 'tests', 'benchmark']
for d in dirs:
    os.makedirs(d, exist_ok=True)

moves = {
    'active_party.py': 'core/active_party.py',
    'passive_party.py': 'core/passive_party.py',
    'data_aligner.py': 'core/data_aligner.py',
    'heservice.py': 'crypto/heservice.py',
    'dp_injector.py': 'crypto/dp_injector.py',
    'test_dp.py': 'tests/test_dp.py',
    'test_dp_compare.py': 'tests/test_dp_compare.py',
    'test_active_party.py': 'tests/test_active_party.py',
    'test_passive_party.py': 'tests/test_passive_party.py',
    'test_histogram.py': 'tests/test_histogram.py',
    'test_gain.py': 'tests/test_gain.py',
    'compare_xgboost.py': 'benchmark/compare_xgboost.py'
}

for src, dst in moves.items():
    if os.path.exists(src):
        shutil.move(src, dst)

def fix_imports(filepath):
    if not os.path.exists(filepath): return
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = content.replace('from heservice import', 'from crypto.heservice import')
    content = content.replace('from dp_injector import', 'from crypto.dp_injector import')
    content = content.replace('from data_aligner import', 'from core.data_aligner import')
    content = content.replace('from active_party import', 'from core.active_party import')
    content = content.replace('from passive_party import', 'from core.passive_party import')
    
    # If it's a test file or benchmark, add sys path so it can import core and crypto locally
    if ('test_' in os.path.basename(filepath)) or ('compare' in os.path.basename(filepath)):
        if 'sys.path.insert' not in content:
            insert_code = "import sys\nimport os\nsys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))\n"
            content = insert_code + content
            
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

for d in dirs:
    for f in glob.glob(d + '/*.py'):
        fix_imports(f)
        
print("Refactoring completed successfully. Project is now structured into core/, crypto/, tests/, and benchmark/ directories.")
