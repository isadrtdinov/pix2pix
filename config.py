class Params:
    # the most important parameter
    random_seed = 224422

    # system params
    verbose = True
    device = None  # to be set on runtime
    num_workers = 8

    # dataset params
    datasets_dir = 'datasets'
    dataset = 'facades'
    train_suffix = 'train'
    valid_suffix = 'val'

    # images params
    image_size = (256, 256)
    input_first = True
    in_channels = 3
    out_channels = 3

    # experimenter params
    runs_dir = 'runs'
    checkpoints_subdir = 'checkpoints'
    checkpoints_template = 'pix2pix{}.pt'
    examples_subdir = 'examples'
    metadata_file = 'metadata.json'
    metrics_file = 'metrics.json'
    examples_ids = [10, 20, 30, 40, 50]

    # generator params
    generator_channels = 64
    generator_layers = 4
    generator_kernel = 3
    generator_dropout = 0.5

    # discriminator params
    adversarial = False

    # train params
    batch_size = 16
    num_epochs = 60
    lr = 1e-4
    loss = 'L1'


def set_params():
    return Params()
