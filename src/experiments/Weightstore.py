import run_settings as rs
import json
import uuid
import sqlite3
import sys

DBNAME = "weightdb_exps.db"


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
                             set_dict.get("model_name",""),
                             set_dict.get("description",""),
                             set_dict.get("dataset",""),
                             set_dict.get("sample_mean",False),
                             set_dict.get("sample_std",False)
                         ))


def load_settings(guid_substring):

    conn = get_db_conn()

    c = conn.cursor()

    t = ("%{0}%".format(guid_substring),)

    c.execute('SELECT * FROM settings WHERE uuid  like ?',t)
    uuid_settings = c.fetchall()

    settings = []
    for s in uuid_settings:
        name = "settings/{0}.nns".format(s[0])

        settings.append(load_settings_file(name, s[0]))

    return settings

def store_settings(settings, model, history):

    if settings.guid == None:
        settings.guid = uuid.uuid4()


    if not model is None:
        # store model as json, with uuid as name, in model
        model_json = model.to_json()
        with open ("models/{0}.json".format(str(settings.guid)),'w+') as jf:
            jf.write(model_json)

    name = rs.settings_folder + str(settings.guid) + ".nns"
    with open(name, 'w+') as fp:
        json.dump(settings.get_dict(),fp)

    loss = -1
    acc = -1
    val_acc = -1
    val_loss = -1

    if history is not None:
        loss = history['loss'][-1]
        acc = history['acc'][-1]
        val_acc = history['val_acc'][-1]
        val_loss = history['val_loss'][-1]

    conn = get_db_conn()

    c = conn.cursor()



    vals = (str(settings.guid), settings.model_name, settings.description, loss, acc, val_loss, val_acc, settings.dataset)

    c.execute("INSERT OR REPLACE INTO settings VALUES (?,?,?,DateTime('now'),?,?,?,?,?)", vals)
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
        print "{0}, {1}, {2}, {3}, {4}".format(settings[2], settings[8], settings[3], settings[1], settings[0])

    return uuid_settings


def add_settings(guid_name):

    settings_path = "settings/{0}.nns".format(guid_name)

    settings = load_settings_file(settings_path, guid_name)

    model_name = settings.model_name


    conn = get_db_conn()
    c = conn.cursor()

    t = (guid_name,settings_path,model_name, settings.description)

    c.execute("INSERT INTO settings VALUES (?,?,?,?,DateTime('now'))",t)
    conn.commit()


def unique_models():

    settings = get_settings_model_name("")

    print ""
    print ""
    print ""

    names = set(map(lambda x : x[1], settings))

    for n in names:
        print n


def update_description(guid_substring, new_description):
    setting = get_settings(guid_substring)

    if setting == None:
        exit()

    conn = get_db_conn()
    c = conn.cursor()

    SQL = "UPDATE settings SET description = ? where uuid = ?"

    c.execute(SQL,(new_description, str(setting.guid)))
    conn.commit()

    print "Updated {0} description to {1}".format(setting.guid, new_description)


if __name__ == "__main__":

    if 'unique' in sys.argv:
        unique_models()
        exit()


    if "ins" in sys.argv:
        guid = sys.argv[-1]

        add_settings(guid)

        exit()

    if 'des' in sys.argv:
        guid = sys.argv[-1]
        description = sys.argv[sys.argv.index('des') + 1]

        update_description(guid, description)

        exit()

    model_name = sys.argv[-1]

    get_settings_model_name(model_name)
