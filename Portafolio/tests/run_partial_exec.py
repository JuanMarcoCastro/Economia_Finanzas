import nbformat
import runpy
import sys
import os
from pathlib import Path

nb_path = Path(__file__).parents[0] / 'action.ipynb'
if not nb_path.exists():
    # try parent folder
    nb_path = Path(__file__).parents[1] / 'action.ipynb'

print('Notebook path:', nb_path)
nb = nbformat.read(str(nb_path), as_version=4)

# Patterns to avoid executing (network/data fetching or session open)
skip_patterns = [
    'ld.open_session', 'lseg.data', 'fetch_refinitiv', 'fetch_ba', 'get_history',
    'Definition(', 'definition.get_data', 'if __name__', "main()", 'run_ba_backtest(', 'run_project_backtest('
]

exec_ns = {}
print('Executing safe cells from notebook to load function definitions...')
for cell in nb.cells:
    if cell.cell_type != 'code':
        continue
    src = cell.source
    if any(pat in src for pat in skip_patterns):
        # skip cells that likely contact external APIs or run examples
        print('Skipping cell containing:', [pat for pat in skip_patterns if pat in src])
        continue
    try:
        exec(src, exec_ns)
    except Exception as e:
        print('Error executing a cell (ignored):', e)

# Check that functions exist
required = ['simulate_strategies', 'make_recommendation', 'make_horizon_recommendation', 'generate_trade_log_from_positions']
for name in required:
    if name not in exec_ns:
        print(f'Function {name} not loaded; aborting test.')
        sys.exit(1)

import pandas as pd
import numpy as np

# Create synthetic data
dates = pd.date_range(start='2025-03-01', periods=250, freq='B')
np.random.seed(42)
rets = np.random.normal(0.0005, 0.02, size=len(dates))
price = 100 * (1 + pd.Series(rets)).cumprod()
df = pd.DataFrame(index=dates)
df['close'] = price.values
# Build open/high/low
df['open'] = df['close'].shift(1).fillna(df['close']).values
df['high'] = df[['open','close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.002, size=len(dates))))
df['low'] = df[['open','close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.002, size=len(dates))))
df['volume'] = (np.random.randint(100, 1000, size=len(dates))).astype(float)

print('Synthetic data prepared with', len(df), 'rows')

# Run simulate_strategies
results = exec_ns['simulate_strategies'](df, entry_date=str(dates[0].date()))
print('\n--- simulate_strategies results keys ---')
print(sorted(results.keys()))

# Run make_recommendation
rec, score = exec_ns['make_recommendation'](results)
print('\nmake_recommendation ->', rec)
print('score =', score)

# Run horizon recommendation
hrec = exec_ns['make_horizon_recommendation'](results, project_end='2025-12-31', min_target_pct=0.02)
print('\nmake_horizon_recommendation ->', hrec)

# Generate trade logs for available pos columns
df2 = results['df']
for pos_col in ['pos_sma','pos_macd','pos_rsi','pos_bb']:
    if pos_col in df2.columns:
        log = exec_ns['generate_trade_log_from_positions'](df2, pos_col=pos_col, commission=0.0, slippage=0.0)
        print(f'-- {pos_col}: {len(log)} trades, total pnl {log["pnl"].sum() if not log.empty else 0}')

print('\nPartial execution test completed successfully.')
