
import sys
import os
sys.path.insert(0, r'c:\Users\DELL\Documents\Mathematical_models\StanLogic\src')
os.chdir(r'c:\Users\DELL\Documents\Mathematical_models\StanLogic\tests\KMapSolver')
from stanlogic import BoolMinHcal
import random
import time

# Worker function inline
def run_worker():
    core_id = 3
    chunk_start = 192
    chunk_end = 256
    csv_path = r'c:\Users\DELL\Documents\Mathematical_models\StanLogic\tests\KMapSolver\outputs\kmapsolver4d_24bit_core3.csv'
    seed = 42
    num_vars = 24
    
    try:
        os.system('title Core {} - Processing chunks {}-{}'.format(core_id, chunk_start, chunk_end-1))
        
        print('\n' + '='*70)
        print('CORE {} - PROCESSING STARTED'.format(core_id))
        print('='*70)
        print('Chunk range: {:,} to {:,} ({:,} chunks)'.format(chunk_start, chunk_end-1, chunk_end - chunk_start))
        print('Output CSV: {}'.format(os.path.basename(csv_path)))
        print('='*70 + '\n')
        
        solver = BoolMinHcal(
            num_vars=num_vars,
            csv_path=csv_path,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            seed=seed,
            verbose=True
        )
        
        stats = solver.minimize(progress_interval=10)
        
        stats_file = csv_path.replace('.csv', '_stats.txt')
        with open(stats_file, 'w') as f:
            f.write('processed_chunks:{}\n'.format(stats['processed_chunks']))
            f.write('total_terms:{}\n'.format(stats['total_terms']))
            f.write('minimized_chunks:{}\n'.format(stats['minimized_chunks']))
            f.write('total_time:{}\n'.format(stats['total_time']))
            f.write('total_elapsed:{}\n'.format(stats['total_elapsed']))
        
        print('\n' + '='*70)
        print('CORE {} - COMPLETED'.format(core_id))
        print('='*70)
        print('Processed {:,} chunks in {:.1f}s'.format(stats['processed_chunks'], stats['total_elapsed']))
        print('Generated {:,} terms'.format(stats['total_terms']))
        print('='*70 + '\n')
        print('Press any key to close this window...')
        input()
        
    except Exception as e:
        print('\n' + '='*70)
        print('CORE {} - ERROR'.format(core_id))
        print('='*70)
        print('ERROR: {}'.format(e))
        import traceback
        traceback.print_exc()
        
        stats_file = csv_path.replace('.csv', '_stats.txt')
        with open(stats_file, 'w') as f:
            f.write('error:{}\n'.format(e))
        
        print('\nPress any key to close this window...')
        input()

run_worker()
