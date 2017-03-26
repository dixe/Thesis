import run_settings as rs
import json
import uuid
import sqlite3
import sys

DBNAME = "weightdb.db"


def get_db_conn():
    conn = sqlite3.connect(DBNAME)
    return conn


def load_settings_file(name, guid_string):
    with open(name, 'r+') as fp:
        set_dict = json.load(fp)



    return (rs.Net_settings( set_dict["img_width"],
                                         set_dict["img_height"],
                                         set_dict["train_data_dir"],
                                         set_dict["validation_data_dir"],
                                         set_dict["nb_train_samples"],
                                         set_dict["nb_validation_samples"],
                                         set_dict["nb_epoch"],
                                         uuid.UUID(guid_string),
                                         set_dict.get("model_name","")))


def load_settings(guid_substring):

    conn = get_db_conn()

    c = conn.cursor()

    t = ("%{0}%".format(guid_substring),)

    c.execute('SELECT * FROM settings WHERE uuid  like ?',t)
    uuid_settings = c.fetchall()

    settings = []
    for s in uuid_settings:
        name = s[1]

        settings.append(load_settings_file(name, s[0]))

    return settings

def store_settings(settings):
    if settings.guid == None:
        settings.guid = uuid.uuid4()

    name = rs.settings_folder + str(settings.guid) + ".nns"
    with open(name, 'w+') as fp:
        json.dump(settings.get_dict(),fp)

    conn = get_db_conn()

    c = conn.cursor()

    vals = (str(settings.guid), name, settings.model_name)

    c.execute("INSERT OR REPLACE INTO settings VALUES(?,?,?)", vals)
    conn.commit()
    return settings


def get_settings(guid_substring):
    settings = load_settings(guid_substring)
    num_settings = len(settings)
    if num_settings != 1:
        if num_settings == 0:
            print "No settings found: {0}".format(guid_substring)
            return None

        print "Multiple settings found"
        for s in settings:
            print s.guid
        return None
    return settings[0]


def get_settings_model_name(model_name):
    conn = get_db_conn()
    c = conn.cursor()

    t = ("%{0}%".format(model_name),)

    c.execute('SELECT * FROM settings WHERE model_name like ?',t)
    uuid_settings = c.fetchall()

    for settings in uuid_settings:
        print "{0}, {1}".format(settings[2], settings[0])

    return uuid_settings


def add_settings(guid_name):

    settings_path = "settings/{0}.nns".format(guid_name)

    setting = load_settings_file(settings_path, guid_name)

    model_name = setting.model_name


    conn = get_db_conn()
    c = conn.cursor()

    t = (guid_name,settings_path,model_name)

    print "inserting"

    c.execute('INSERT INTO settings VALUES (?,?,?)',t)
    conn.commit()




def test():

    settings = load_settings("f8463")

    print len(settings)


if __name__ == "__main__":



    if "ins" in sys.argv:
        guid = sys.argv[-1]


        add_settings(guid)

        exit()

    model_name = sys.argv[-1]

    get_settings_model_name(model_name)
