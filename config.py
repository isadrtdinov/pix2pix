class Params:
    # the most important parameter
    random_seed = 224422

    # system params
    verbose = True
    device = None  # to be set on runtime
    num_workers = 2

    # dataset params
    datasets_dir = 'datasets'
    dataset = 'churches'
    train_suffix = 'train'
    valid_suffix = 'val'
    flip = True
    normalize = True

    # images params
    image_size = (512, 1024)
    input_left = True
    in_channels = 3
    out_channels = 3

    # experimenter params
    runs_dir = 'runs'
    save_checkpoints = True
    load_checkpoint = None
    checkpoints_subdir = 'checkpoints'
    checkpoints_template = 'pix2pix{}.pt'
    checkpoints_freq = 10
    examples_subdir = 'examples'
    metadata_file = 'metadata.json'
    metrics_file = 'metrics.json'
    examples_ids = [6, 9, 11, 16, 45]

    # generator params
    generator_channels = 64
    generator_layers = 4
    generator_kernel = 5
    generator_dropout = 0.5
    generator_norm = 'instance'

    # discriminator params
    adversarial = True
    discriminator_channels = 64
    discriminator_layers = 3
    discriminator_norm = 'instance'

    # train params
    batch_size = 4
    num_epochs = 50
    lr = 3e-4
    loss = 'L2'
    loss_lambda = 100.0


def set_params():
    return Params()
