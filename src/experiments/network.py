import run_settings as rs


class Net_settings(object):


    def __init__(self, save_weights_path, load_weights_path,img_width,img_height, train_data_dir, validation_data_dir, nb_train_samples,nb_validation_samples, nb_epoch):
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


class Base_network(object):
    """
    Base class for nets handles data paths, writing results and weights
    """


    def __init__(self, net_settings):
        self.settings = net_settings

    def get_model(self):
        raise NotImplementedError( "Should have implemented this" )

    def get_model_test(self):
        raise NotImplementedError( "Should have implemented this" )


    def save_model_weight(self, model):
        model.save_weights(self.settings.save_weights_path)
        # maybe add a result file



def default_settings():
    return Net_settings(rs.save_weights_path,
                        rs.load_weights_path,
                        rs.img_width,
                        rs.img_height,
                        rs.train_data_dir,
                        rs.validation_data_dir,
                        rs.nb_train_samples,
                        rs.nb_validation_samples,
                        rs.nb_epoch)
