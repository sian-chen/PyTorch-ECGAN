import os
import subprocess

CMD = ['python', 'main.py', '-e']
for ckpt_name in os.listdir('checkpoints'):
    if not('ACGAN' in ckpt_name or 'ecgan' in ckpt_name):
        continue
    ckpt_dir = os.path.join('checkpoints', ckpt_name)
    config_name = ckpt_name[:ckpt_name.find('-train-')]
    config_path = f'exp_configs/CIFAR10/{config_name}.json'
    cmd = CMD + ['-c', config_path, '--checkpoint_folder', ckpt_dir]
    print(cmd)
    subprocess.run(cmd)
