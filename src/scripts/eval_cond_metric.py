import os
import sys
import subprocess

if len(sys.argv) == 3:
    gpu_id = sys.argv[1]
    target = sys.argv[2]
else:
    gpu_id = '0'
    target = sys.argv[1]

env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = gpu_id


CMD = ['python', 'src/main.py', '-e']

for ckpt_name in os.listdir('checkpoints'):
    if not ckpt_name.startswith(target + '-'):
        continue
    ckpt_dir = os.path.join('checkpoints', ckpt_name)
    config_name = ckpt_name[:ckpt_name.find('-train-')]
    # config_path = f'src/exp_configs/CIFAR10/{config_name}.json'
    config_path = f'src/exp_configs/TINY_ILSVRC2012/{config_name}.json'
    cmd = CMD + ['-c', config_path, '--checkpoint_folder', ckpt_dir, '--eval_type', 'valid']
    print(' '.join(cmd))
    # subprocess.run(cmd, env=env)
