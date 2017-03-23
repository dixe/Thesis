import os
import shutil

if __name__ == "__main__":

    path = 'E://Speciale//auto_encoder_data//'

    new_root = 'E://Speciale//auto_encoder_all_imgs//'


    idd = 0
    for r,ds,fs in os.walk(path):

        for f in fs:
            new_name = "{0}-{1}".format(idd,f)
            shutil.copy(r + '//' + f, new_root + new_name)

        idd+=1
