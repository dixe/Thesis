import sqlite3
import Weightstore as ws

def v_0_2():
    conn = ws.get_db_conn()

    c = conn.cursor()

    SQL = "ALTER TABLE settings ADD COLUMN model_name TEXT;"

    c.execute(SQL)

    conn.commit()

    print "added model_name as TEXT column"


def v_0_3():
    conn = ws.get_db_conn()

    c = conn.cursor()

    SQL = "ALTER TABLE settings ADD COLUMN description TEXT;"

    c.execute(SQL)

    conn.commit()

    print "added description as TEXT column"


def v_0_4():
    conn = ws.get_db_conn()

    c = conn.cursor()

    SQL = "ALTER TABLE settings ADD COLUMN dt DATETIME;"

    c.execute(SQL)

    conn.commit()

    print "added datetime column"


def v_0_5():
    conn = ws.get_db_conn()

    c = conn.cursor()

    SQL = "ALTER TABLE settings ADD COLUMN results TEXT;"

    c.execute(SQL)

    conn.commit()

    print "added results as TEXT column"




if __name__ == "__main__":

    #v_0_4()
    v_0_5()
