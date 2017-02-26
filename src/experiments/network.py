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

    def save_model_weight(self, model):
        model.save_weights(self.settings.save_weights_path)
        # maybe add a result file


    def train_model(self, model):

        # compile the model with a SGD/momentum optimizer
        # and a very slow learning rate.
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy',])

        # prepare data augmentation configuration
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            self.settings.train_data_dir,
            target_size=(self.settings.img_height, self.settings.img_width),
            batch_size=32,
            class_mode='binary')

        validation_generator = test_datagen.flow_from_directory(
            self.settings.validation_data_dir,
            target_size=(self.settings.img_height, self.settings.img_width),
            batch_size=32,
            class_mode='binary')

        # fine-tune the model
        model.fit_generator(
            train_generator,
            samples_per_epoch=self.settings.nb_train_samples,
            nb_epoch=self.settings.nb_epoch,
            validation_data=validation_generator,
            nb_val_samples=self.settings.nb_validation_samples)

        return model

    def fine_tune_and_save(self):
        model = self.get_model()
        model = self.train_model(model)
        self.save_model_weight(model)



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
