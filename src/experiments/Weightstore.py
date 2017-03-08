import run_settings as rs
import json
import uuid
import sqlite3

DBNAME = "weightdb.db"


def get_db_conn():
    conn = sqlite3.connect(DBNAME)
    return conn

def load_settings(guid_substring):

    conn = get_db_conn()

    c = conn.cursor()

    t = ('%' +guid_substring + '%',)

    c.execute('SELECT * FROM settings WHERE uuid  like ?',t)
    uuid_settings = c.fetchall()

    settings = []
    for s in uuid_settings:
        name = s[1]
        with open(name, 'r+') as fp:
            set_dict = json.load(fp)


        settings.append(rs.Net_settings( set_dict["save_weights_path"],
                                 set_dict["load_weights_path"],
                                 set_dict["img_width"],
                                 set_dict["img_height"],
                                 set_dict["train_data_dir"],
                                 set_dict["validation_data_dir"],
                                 set_dict["nb_train_samples"],
                                 set_dict["nb_validation_samples"],
                                 set_dict["nb_epoch"],
                                 uuid.UUID(s[0])))

    return settings



def store_settings(settings):
    if settings.guid == None:
        settings.guid = uuid.uuid4()

    name = rs.save_folder + str(settings.guid) + ".nns"
    with open(name, 'w+') as fp:
        json.dump(settings.get_dict(),fp)

    conn = get_db_conn()

    c = conn.cursor()

    vals = (str(settings.guid), name)
    print vals
    c.execute("INSERT INTO settings VALUES(?,?)", vals)
    conn.commit()
    return settings

if __name__ == "__main__":
    #settings = rs.default_settings()
    #settings = store_settings(settings)

    settings = load_settings("cfebe88")

    print len(settings)
