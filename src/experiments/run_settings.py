import uuid

dataset_name = 'data-64-1000-400'

eval_data_dir = '/home/nikolaj/Thesis/datasets/{0}/validation'.format(dataset_name)

train_data_dir = "/home/nikolaj/Thesis/datasets/{0}/train/".format(dataset_name)

validation_data_dir = "/home/nikolaj/Thesis/datasets/{0}/validation/".format(dataset_name)

save_weights_path = "fine_tuned_model_18.h5"

load_weights_path = "fine_tuned_model.h5"

save_folder = "weights"

img_width = 64
img_height = img_width

nb_validation_samples = 800

nb_train_samples = 2000

nb_epoch = 1





class Net_settings(object):


    def __init__(self, save_weights_path, load_weights_path,img_width,img_height, train_data_dir, validation_data_dir, nb_train_samples,nb_validation_samples, nb_epoch, guid = None):
        # path to the model weights file.
        self.save_weights_path = save_weights_path
        self.load_weights_path = load_weights_path
        # dimensions of our images.
        self.img_width, self.img_height = img_width, img_height

        self.train_data_dir = train_data_dir
        self.validation_data_dir = validation_data_dir
        self.nb_train_samples = nb_train_samples
        self.nb_validation_samples = nb_validation_samples
        self.nb_epoch = nb_epoch
        self.guid = guid

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

        return sd



def default_settings():
    return Net_settings(save_weights_path,
                        load_weights_path,
                        img_width,
                        img_height,
                        train_data_dir,
                        validation_data_dir,
                        nb_train_samples,
                        nb_validation_samples,
                        nb_epoch)
