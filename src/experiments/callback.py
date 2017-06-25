import keras
import sys
import Weightstore as ws
import run_settings as rs

class EarlyStoppingByLossVal(keras.callbacks.Callback):
    def __init__(self, dataset, settings):
        super(keras.callbacks.Callback, self).__init__()
        self.losses = []
        self.dataset = dataset
        self.settings = settings
        self.val_losses = []
        self.lowest_eval = 1000000 # no eval error should be lager than this
        self.best_weight = None

        self.store_string = "epoch: {0}, {1}: {2}, train_loss : {3}, train_acc : {4}, val_loss : {5}, val_acc : {6}\n"


    def on_epoch_end(self, epoch, logs={}):

        EPOCH_MAX = rs.nb_epoch

        if epoch % 3 == 0:
            self.store_weights()

        cur_eval_loss = logs.get('val_loss')

        # store best eval loss weights
        if cur_eval_loss < self.lowest_eval:
            self.lowest_eval = cur_eval_loss
            self.best_weight = self.model.get_weights()

        self.losses.append(logs.get('val_loss'))
        self.val_losses.append(cur_eval_loss)


        #self.glt_stop(self.lowest_eval, cur_eval_loss, epoch)

        self.pqt_stop(self.lowest_eval, cur_eval_loss, epoch,
                      logs.get('loss'), logs.get('acc'), logs.get('val_loss'), logs.get('val_acc'))


        if epoch > EPOCH_MAX:
            self.model.stop_training = True
            self.model.set_weights(self.best_weight)


    def store_weights(self):

        self.model.save_weights(self.settings.save_weights_path)
        ws.store_settings(self.settings, self.model, None)


    def pqt_stop(self, eopt, evat, epoch, train_loss, train_acc, val_loss, val_acc):

        PQNUM = 3

        k = 5

        loss_inv = self.losses[max(epoch-k+1,0) : epoch+1]

        pkt = 1000 * (sum(loss_inv) / (1.0  * (k * min(loss_inv))) -1)

        glt = 100 * (evat / (1.0 * eopt) - 1)

        pqt = (1.0* glt) / pkt

        store_string = self.store_string.format(epoch, "pqt", pqt, train_loss, train_acc, val_loss, val_acc)


        with open("{0}_pqt_{1}.csv".format(self.settings.model_name, self.dataset), 'a') as pqt_f:
            pqt_f.write(store_string)


        if pqt > PQNUM and epoch > 40:
            print("Epoch %05d: early stopping THR" % epoch)
            self.model.set_weights(self.best_weight)
            self.model.stop_training = True





    def glt_stop(self, eopt, evat, epoch):

        GLNUM = 5

        glt = 100 * (evat / (1.0 * eopt) - 1)

        with open('glt_patches_gm.txt', 'a') as glt_f:
            glt_f.write("epoch: {0}, glt: {1}\n".format(epoch,glt))

        if glt > 10000*GLNUM:
            print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
            self.model.set_weights(self.best_weight)




class Stop_callback(keras.callbacks.Callback):


    def on_epoch_end(self, epoch, logs):
        print sys.stdin
        if not sys.stdin.isatty():
            print "Not empty"
            exit()
