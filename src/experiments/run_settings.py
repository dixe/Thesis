import uuid


auto_enc = False


dataset_name = 'auto_encoder_all_patches' if auto_enc else 'patches'
 
train_data_dir = "/home/ltm741/thesis/datasets/arg_data_sets/{0}/train/".format(dataset_name)

validation_data_dir = "/home/ltm741/thesis/datasets/{0}/validation".format(dataset_name)

#eval_data_dir = "/home/ltm741/thesis/datasets/{0}/validation".format(dataset_name)


full_imgs_path = "/home/ltm741/thesis/datasets/BG_Sequences_w_ROI_Annotated/Oktober 15, 2016/"


weights_folder = "weights/"
settings_folder = "settings/"

#img_width = 280
#img_height = 176

img_width = 64
img_height = img_width

nb_validation_samples = 1610

nb_train_samples = 86350 if auto_enc  else 9122  #120  #7850

nb_epoch = 30


class Net_settings(object):


    def __init__(self, img_width,img_height, train_data_dir, validation_data_dir, nb_train_samples,nb_validation_samples, nb_epoch, guid = None, model_name = "", description = ""):
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
        return sd



def default_settings():
    return Net_settings(img_width,
                        img_height,
                        train_data_dir,
                        validation_data_dir,
                        nb_train_samples,
                        nb_validation_samples,
                        nb_epoch)
