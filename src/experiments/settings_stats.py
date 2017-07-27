import sys
import run_settings as rs
import Weightstore as ws



def load_settings_from_file(filePath):

    lines = []
    with open(filePath,'r') as f:
        for l in f.readlines():
            lines.append(l)

    settings = []
    for l in lines:
        if l and not l.startswith('#'):
            l =  l.strip()
            setting = ws.load_settings(l)
            settings += setting

    return settings


def create_settings_row(settings):


    row_str = "{0} & {1} & {2} & {3} & {4}\\\\ \\hline \n"

    name = settings.model_name.replace("_","\_")
    dataset = settings.dataset.replace("_","\_")

    row_str = row_str.format(name, settings.sample_mean, settings.sample_std, dataset, settings.guid)

    return row_str

def create_settings_table(settings):

    header = "\\begin{tabular}{c|c|c|c|c}\n"
    header += "Model Name & Sample mean & Sample std & Dataset & Guid \\\\ \\hline \n"
    body = ""
    settings = sorted(settings, key=lambda x : x.model_name)

    for s in settings:
        body += create_settings_row(s)

    footer = "\\end{tabular}"

    return header + body + footer
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
