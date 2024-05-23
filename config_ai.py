# configurations
config = {
    'seed': 919,
    'net_g': 'lpft/0.1', # select the pre-trained g
    'lr': 2e-4,
    'lr_disc': 2e-4,
    'ndf': 1024,
    'batch_size': 16,
    'M': 10,
    'integration_steps': 10,
    'spatial_dimension': '[B, C, H, W]',
    'total_steps': 200001,
    'warmup_steps': 5000,
    'warmup_steps_disc': 20000,
    'saving_path': './models'
}