import os
import itertools

# TEMPLATE = 'exp_configs/CIFAR10/ecgan0.json'
# OUTPUT_DIR = 'exp_configs/CIFAR10/'
TEMPLATE = 'exp_configs/TINY_ILSVRC2012/ecgan0.json'
# OUTPUT_DIR = 'exp_configs/TINY_ILSVRC2012/'
OUTPUT_DIR = 'exp_configs/tune_tiny/'

TUNED_PARAMS = {
    'alpha': [0.99, 0.9],
    'cls_disc_lambda': [0.1],
    'd_lr': [0.0004],
    'g_lr': [0.0002]
}

# TUNED_PARAMS = {
    # 'uncond_lambda': [1.],
    # 'cls_disc_lambda': [0.1],
    # 'cls_disc_lambda': [0.1, 0.01],
    # 'contrastive_type-contrastive_lambda': [ ('ContraGAN', 0.), ('ContraGAN', 1.)],
    # 'd_lr': [0.0002, 0.0004],
    # 'g_lr': [0.00005, 0.0001, 0.0002]
# }

# TUNED_PARAMS = {
    # 'cond_lambda': [1.],
    # 'contrastive_type-contrastive_lambda': [
        # ('ContraGAN', 0.), ('ContraGAN', 1.)],
    # 'd_lr-g_lr': [
        # (0.0008, 0.0002),
        # (0.0004, 0.0002),
        # (0.0002, 0.0002),
        # (0.0001, 0.0002),
        # (0.0008, 0.0001),
        # (0.0004, 0.0001),
        # (0.0002, 0.0001),
        # (0.0001, 0.0001),
        # (0.0008, 0.00005),
        # (0.0004, 0.00005),
        # (0.0002, 0.00005),
        # (0.0001, 0.00005),
    # ],
# }


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        config = {}
        for key, val in zip(keys ,instance):
            if '-' in key:
                for key2, val2 in zip(key.split('-'), val):
                    config[key2] = val2
            else:
                config[key] = val
        yield config


def add_arg(template, arg_name, val):
    st = template.find(arg_name) + len(arg_name) + 3
    ed = template.find(',', st)
    val_str = f'"{val}"' if isinstance(val, str) else str(val)
    return template[:st] + val_str + template[ed:]
    new_data = data[:st] + '"cls_disc_lambda": "N/A",\n        "cls_gen_lambda": "N/A",\n        ' + data[st:]


def gen_config(template, params):
    if 'alpha' in params:
        params['cond_lambda'] = params['alpha']
        params['uncond_lambda'] = round(1 - params['alpha'], 4)
        params['cls_disc_lambda'] = round(params['cls_disc_lambda'] * (1 - params['alpha']), 4)
    if params.get('uncond_lambda', 0) == 0 and params.get('cond_lambda', 0) == 1:
        prefix = 'ecgan1'
    elif params.get('uncond_lambda', 0) == 1 and params.get('cond_lambda', 0) == 0:
        prefix = 'ecgan2'
    elif params.get('uncond_lambda', 0) > 0 and params.get('cond_lambda', 0) > 0:
        prefix = 'ecgan_' + (str(params.get('cond_lambda', 0)) + '_' + str(params.get('uncond_lambda', 0))).replace('.', 'p')

    config_name = prefix
    if params.get('contrastive_lambda', 0) == 0:
        config_name += '_none'
    elif params.get('contrastive_type', 'ContraGAN') == 'ContraGAN':
        config_name += '_contragan' + str(params.get('contrastive_lambda', 0)).replace('.', 'p')
    elif params.get('contrastive_type', 'ContraGAN') == 'NT_Xent':
        config_name += '_ntxent' + str(params.get('contrastive_lambda', 0)).replace('.', 'p')

    if params.get('cls_disc_lambda', 0):
        config_name += '_ce' + str(params['cls_disc_lambda']).replace('.', 'p')

    if 'd_lr' in params:
        config_name += '_lr{:.0e}_{:.0e}'.format(params['d_lr'], params['g_lr'])

    for arg_name, val in params.items():
        template = add_arg(template, arg_name, val)

    config_path = os.path.join(OUTPUT_DIR, config_name + '.json')
    print(f'{config_path}')
    # print(params)
    with open(config_path, 'w') as fp:
        fp.write(template)


def main():
    with open(TEMPLATE) as fp:
        template = fp.read()

    for params in product_dict(**TUNED_PARAMS):
        gen_config(template, params)


if __name__ == '__main__':
    main()
