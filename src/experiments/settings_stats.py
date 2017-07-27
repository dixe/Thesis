import run_settings as rs
import Weightstore as ws
import sys.argv

def load_settings_from_file(filePath):

    return []


def create_settings_row(settings):


    row_str = "{0} & {1} & {2} & {3} & {4}\\\\ \\hline \n"

    name = settings.model_name.replace("_","\_")
    dataset = settings.dataset.replac("_","\_")

    row_str.format(name, settings.sample_mean, sample_std, dataset, settings.guid)



def create_settings_table(settings):

    header = "\\begin{tabular}{c|c|c|c|c}\n"
    header += "Model Name & Sample mean & Sample std & Dataset & Guid \\\\ \\hline \n"
    body = ""
    for s in settings:
        body += create_settings_row(s)

    footer = "\\end{tabular}"


def get_arg_from_sysargv(arg_name):
    if arg_name not in sys.argv:
        return None

    index = sys.argv.index(arg_name)
    return sys.argv[index + 1]

if __name__ == "__main__":

    path = ""
    if "path" in sys.argv:
        path = get_arg_from_sysargv("path")
    else:
        print ("Requires a path to file with settings")

    outpath = None
    if "outpath" in sys.argv:
        outpath = get_arg_from_sysargv("outpath")

    settings = load_settings_from_file(path)

    table = create_settings_table(settings)


    if outpath is None:
        print (table)
    else:
        with open(outpath,'w') as f:
            f.wirte(table)
