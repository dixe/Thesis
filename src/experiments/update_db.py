import sqlite3
import Weightstore as ws

def v_0_2():
    conn = ws.get_db_conn()

    c = conn.cursor()

    SQL = "ALTER TABLE settings ADD COLUMN model_name TEXT;"

    c.execute(SQL)

    conn.commit()

    print "added model_name as TEXT column"


if __name__ == "__main__":

    v_0_2()
