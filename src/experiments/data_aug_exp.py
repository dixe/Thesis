
import simple_model as sm
import run_settings as rs
import os


def path_sub_name(rot,sc,tl,gm):
    return "{0}{1}{2}{3}".format("_rot" if rot else "", "_sc" if sc else "", "_tl" if tl else "", "_gm" if gm else "")


if __name__ == "__main__":
    
    tf = [False, True]

    for rot in tf:
        for sc in tf:
            for tl in tf:
                for gm in tf:
                    dataset_name = "patches{0}".format(path_sub_name(rot,sc,tl,gm))

                    if not os.path.isfile("pqt_{0}.csv".format(dataset_name)):
                        print "training on {0}".format(dataset_name)
                        sm.train_dataset(dataset_name)
                    else:
                        print "Skipping {0}".format(dataset_name)
