import uuid


size_dict_train = {"patches" : 1711, 'patches_gm' : 5145, 'patches_tl' : 15427, 'patches_tl_gm': 46310, 'patches_sc' : 5140,
                   'patches_sc_gm' : 15434, 'patches_sc_tl' : 54457, 'patches_sc_tl_gm': 138868, 'patches_rot' : 18142,
                   'patches_rot_gm' : 46333, 'patches_rot_tl' : 138869, 'patches_rot_tl_gm' : 416816, 'patches_rot_sc' : 46265,
                   'patches_rot_sc_gm' : 138938, 'patches_rot_sc_tl' : 416402, 'patches_rot_sc_tl_gm' : 1256216,
                   'patches_mini' : 600, "patches_32" : 2458, "patches_32_gm" : 7382, "patches_32_sc" : 7383, "patches_32_sc_gm" : 22152}

size_dict_val = {"patches" : 303, 'patches_gm':  910, 'patches_tl' : 2724, 'patches_tl_gm' : 8174, 'patches_sc' : 909,
                 'patches_sc_gm' : 2726, 'patches_sc_tl' : 8170, 'patches_sc_tl_gm' :  24507, 'patches_rot' : 2723,
                 'patches_rot_gm' : 8178, 'patches_rot_tl' : 24506, 'patches_rot_tl_gm' : 73556, 'patches_rot_sc' : 8165,
                 'patches_rot_sc_gm' : 24518, 'patches_rot_sc_tl' : 73484, 'patches_rot_sc_tl_gm': 215635,
                 'patches_mini' : 303, "patches_32" : 436, "patches_32_gm" : 1304, "patches_32_sc" : 1303, "patches_32_sc_gm" : 3910}



auto_enc = False


dataset_name = 'auto_encoder_all_patches' if auto_enc else 'patches_gm'

train_data_dir_base ="/home/ltm741/thesis/datasets/arg_data_sets_few_whole/{0}/train/"
train_data_dir = train_data_dir_base.format(dataset_name)


val_data_dir_base = "/home/ltm741/thesis/datasets/arg_data_sets_few_whole/{0}/validation"
validation_data_dir = val_data_dir_base.format(dataset_name)

#eval_data_dir = "/home/ltm741/thesis/datasets/{0}/validation".format(dataset_name)


full_imgs_path = "/home/ltm741/thesis/datasets/BG_Sequences_w_ROI_Annotated/Oktober 15, 2016/"


weights_folder = "weights/"
settings_folder = "settings/"

#img_width = 280
#img_height = 176

img_width = 32 if '32' in dataset_name else 64
img_height = img_width

nb_validation_samples = size_dict_val[dataset_name]

nb_train_samples = 86350 if auto_enc  else size_dict_train[dataset_name]

nb_epoch = 300


class Net_settings(object):


    def __init__(self, img_width,img_height, train_data_dir, validation_data_dir, nb_train_samples,nb_validation_samples, nb_epoch, guid = None, model_name = "", description = "", dataset = "", sample_mean = False, sample_std = False):
        # path to the model weights file.
        self.load_weights_path = ""
        # dimensions of our images.
        self.img_width, self.img_height = img_width, img_height

        self.train_data_dir = train_data_dir
        self.validation_data_dir = validation_data_dir
        self.nb_train_samples = nb_train_samples
        self.nb_validation_samples = nb_validation_samples
        self.nb_epoch = nb_epoch
        if guid == None:
            # create a uuid for this setting
            guid = uuid.uuid4()
        self.guid = guid
        self.save_weights_path = "{0}/{1}.h5".format(weights_folder, self.guid)
        self.model_name = model_name
        self.description = description
        self.dataset = dataset
        self.sample_mean = sample_mean
        self.sample_std = sample_std


    def get_dict(self):
        sd = {}
        sd["save_weights_path"] = self.save_weights_path
        sd["load_weights_path"] = self.load_weights_path
        # dimensions of our images.
        sd["img_width"] = self.img_width
        sd["img_height"] = self.img_height

        sd["train_data_dir"] = self.train_data_dir
        sd["validation_data_dir"] = self.validation_data_dir
        sd["nb_train_samples"] = self.nb_train_samples
        sd["nb_validation_samples"] = self.nb_validation_samples
        sd["nb_epoch"] = self.nb_epoch
        sd["model_name"] = self.model_name
        sd["description"] = self.description
        sd["dataset"] = self.dataset
        sd["sample_mean"] = self.sample_mean
        sd["sample_std"] = self.sample_std

        return sd



def default_settings(dataset = None):

    if dataset == None:
        train_dir = train_data_dir
        val_dir = validation_data_dir
        ds = dataset_name
    else:
        train_dir = train_data_dir_base.format(dataset)
        val_dir = val_data_dir_base.format(dataset)
        ds = dataset

    return Net_settings(img_width,
                        img_height,
                        train_dir,
                        val_dir,
                        nb_train_samples,
                        nb_validation_samples,
                        nb_epoch,
                        dataset = ds,
                        sample_mean = True,
                        sample_std = True)
