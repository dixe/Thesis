
import simple_model as sm
import run_settings as rs
import os



BASE_NAME = "patches_32{0}"

def path_sub_name(rot,sc,tl,gm):
    return "{0}{1}{2}{3}".format("_rot" if rot else "", "_sc" if sc else "", "_tl" if tl else "", "_gm" if gm else "")



def run_data_arg_exp(names):    
    rot_tf = [False, True] if 'rot' in names else [False]
    sc_tf = [False, True] if 'sc' in names else [False]
    tl_tf = [False, True] if 'tl' in names else [False]
    gm_tf = [False, True] if 'gm' in names else [False]


    for rot in rot_tf:
        for sc in sc_tf:
            for tl in tl_tf:
                for gm in gm_tf:
                    dataset_name = BASE_NAME.format(path_sub_name(rot,sc,tl,gm))

                    if not os.path.isfile("pqt_{0}.csv".format(dataset_name)):
                        print "training on {0}".format(dataset_name)
                        sm.train_dataset(dataset_name)
                    else:
                        print "Skipping {0}".format(dataset_name)


if __name__ == "__main__":

    names = ['sc','gm']
    print "Running on {0}".format(names)

    run_data_arg_exp(names)