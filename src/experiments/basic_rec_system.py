import simple_model as sm
import Weightstore as ws
from keras.preprocessing.image import load_img, img_to_array


def predict_img(net, img, window_size = 64, stride = 2):

    model = net.get_model_test()

    strides_x = len(img[0][0])
    strides_y = len(img[0][0][0])

    for i in range(strides_x):
        for j in range(strides_y):
            patch = img[:,i:i + window_size, j:j+window_size]

            print patch

            print model.predict(patch)










def get_settings_from_sysarg():
    sys.argv = filter(lambda x : x != '',sys.argv )
    guid_substring = sys.argv[-1]
    return ws.get_settings(guid_substring)


def test_simple(net):

    path = SETTHIS

    img = np.array([img_to_array(load_img(path))])

    predict_img(net, img)








if __name__ == "__main__":

    settigns = get_settings_from_sysarg()

    net = sm.get_net(settings)

    test_simple(net)
