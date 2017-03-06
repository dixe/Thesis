import run_settings as rs
import json
import uuid
import os

def load_weigh(guid_substring):
    for pass



def store_settings(settings):
    if settings.guid == None:
        settings.guid = uuid.uuid4()

    name = str(settings.guid) + ".nns"
    with open(name, 'w+') as fp:
        json.dump(settings.get_dict(),fp)








if __name__ == "__main__":
    settings = rs.default_settings()
    store_settings(settings)
