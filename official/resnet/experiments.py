import itertools
import numpy as np
LIMIT_TESTS = False	# Toggle based on whether we want to run exhaustive experiments or a subset 


class Dataset(object):

    def __init__(self):

        # # #
        # Dataset general
        self.dataset_path = ""
        self.proportion_training_set = 0.95
        self.shuffle_data = True

        # # #
        # For reusing tfrecords:
        self.reuse_TFrecords = False
        self.reuse_TFrecords_ID = 0
        self.reuse_TFrecords_path = ""

        # # #
        # Set random labels
        self.random_labels = False
        self.scramble_data = False

        # # #
        # Transfer learning
        self.transfer_learning = False
        self.transfer_pretrain = True
        self.transfer_label_offset = 0
        self.transfer_restart_name = "_pretrain"
        self.transfer_append_name = ""

        # Find dataset path:
#         for line in open("datasets/paths", 'r'):
#             if 'Dataset:' in line:
#                 self.dataset_path = line.split(" ")[1].replace('\r', '').replace('\n', '')

    # # #
    # Dataset general
    # Set base tfrecords
    def generate_base_tfrecords(self):
        self.reuse_TFrecords = False

    # Set reuse tfrecords mode
    def reuse_tfrecords(self, experiment):
        self.reuse_TFrecords = True
        self.reuse_TFrecords_ID = experiment.ID
        self.reuse_TFrecords_path = experiment.name

    # # #
    # Transfer learning
    def do_pretrain_transfer_lerning(self):
        self.transfer_learning = True
        self.transfer_append_name = self.transfer_restart_name

    def do_transfer_transfer_lerning(self):
        self.transfer_learning = True
        self.transfer_pretrain = True
        self.transfer_label_offset = 5
        self.transfer_append_name = "_transfer"


class DNN(object):	# TODO change for MNIST

    def __init__(self):
        self.name = 'resnet'
        self.pretrained = False
        self.version = 1
        self.layers = 4
        self.stride = 2
        self.neuron_multiplier = np.ones([self.layers])
        self.num_input_channels = 1

    def set_num_layers(self, num_layers):
        self.layers = num_layers
        self.neuron_multiplier = np.ones([self.layers])


class Hyperparameters(object):	# TODO change for MNIST

    def __init__(self):
        self.batch_size = 2 
        self.learning_rate = 1e-2
        self.num_epochs_per_decay = 1.0
        self.learning_rate_factor_per_decay = 0.95
        self.weight_decay = 0
        self.max_num_epochs = 100
        self.image_size = 28	# Changed for MNIST
        self.drop_train = 1
        self.drop_test = 1
        self.momentum = 0.9
        self.augmentation = False
       
        # Params specific to this study, set to defaults
        self.background_size = 0
        self.num_train_ex = 2**3
        self.full_size = False
        self.inverted_pyramid = False

    def set_background_size(self, background_size):
#         self.image_size += background_size
        self.background_size = background_size
        

class Experiments(object):

    def __init__(self, id, name):
        self.name = "base"
        self.log_dir_base = '/om2/user/sanjanas/resnet-ecc-data/models/'
        self.raw_data_dir = '/om2/user/xboix/data/ImageNet/raw-data'
        self.local_scratch_dir = '/om2/user/sanjanas/resnet-ecc-data/imagenet/'

        # Recordings
        self.max_to_keep_checkpoints = 5
        self.recordings = False
        self.num_batches_recordings = 0

        # Plotting
        self.plot_details = 0
        self.plotting = False

        # Test after training:
        self.test = False

        # Start from scratch even if it existed
        self.restart = False

        # Skip running experiments
        self.skip = False

        # Save extense summary
        self.extense_summary = True

        # Add ID to name:
        self.ID = id
        self.name = 'ID' + str(self.ID) + "_" + name

        # Add additional descriptors to Experiments
        self.dataset = Dataset()
        self.dnn = DNN()
        self.hyper = Hyperparameters()

    def do_recordings(self, max_epochs):
        self.max_to_keep_checkpoints = 0
        self.recordings = True
        self.hyper.max_num_epochs = max_epochs
        self.num_batches_recordings = 10

    def do_plotting(self, plot_details=0):
        self.plot_details = plot_details
        self.plotting = True

# # #
# Create set of experiments
opt = []
plot_freezing = []

# General hyperparameters
learning_rates = [10**i for i in range(-6, 0)]	# 6 values
batch_sizes = [10, 25, 50]

# Experiment-specific hyperparameters
num_train_exs = [50, 100, 200, 400, 800, 1300]

idx = 0
for num_train_ex in num_train_exs:

    data = Experiments(idx, 'numtrainex' + str(num_train_ex)) 
    data.hyper.max_num_epochs = 0
    data.hyper.num_train_ex = num_train_ex
    opt.append(data)

    idx += 1


if __name__ == '__main__':
    # print(calculate_IDs([False, True], [3, 14, 28], [16, 32], [40], learning_rates[:]))
    # print(calculate_IDs(background_sizes[:], [8], [128], [0.1]))
    # print(calculate_randombg_IDs(num_train_exs[:], [40], learning_rates[:]))
    pass
