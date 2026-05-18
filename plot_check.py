import os
import sys
import importlib.util

BASE = os.getcwd()
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'mid term'))

file_path = os.path.join(BASE, 'mid term', 'Plot_pressure_drop.py')
spec = importlib.util.spec_from_file_location('plot_pressure_drop', file_path)
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)

import matplotlib
matplotlib.use('Agg')
fig, result, data = pp.run_and_plot(L=1.0, N=100, save_path='pressure_drop_test.png')
print('L', result['L'], 'N', result['N'])
print('node0_cold', result['node_data'][0]['T_cold'], 'nodeN_cold', result['node_data'][-1]['T_cold'])
print('x0', result['node_data'][0]['x_pos'], 'xN', result['node_data'][-1]['x_pos'])
print('first5_hot', data[1][:5])
print('first5_cold', data[2][:5])
print('saved', os.path.exists('pressure_drop_test.png'))
