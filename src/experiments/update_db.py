import sqlite3
import Weightstore as ws
import sys

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



def v_0_6():
    conn = ws.get_db_conn()

    c = conn.cursor()


    SQL = """ALTER TABLE settings ADD COLUMN acc REAL;
           ALTER TABLE settings ADD COLUMN loss REAL;
           ALTER TABLE settings ADD COLUMN val_loss REAL;
           ALTER TABLE settings ADD COLUMN val_acc REAL;"""

    c.execute(SQL)

    conn.commit()

    print "added results as TEXT column"


def v_0_7():
    conn = ws.get_db_conn()

    c = conn.cursor()

    SQL = "ALTER TABLE settings ADD COLUMN dataset TEXT;"

    c.execute(SQL)

    conn.commit()

    print "added dataset as TEXT column"




def create_db():

    conn = ws.get_db_conn()

    c = conn.cursor()


    SQL = "CREATE TABLE settings(uuid varchar(64) PRIMARY KEY, model_name TEXT, description TEXT, dt DATETIME, loss REAL, acc REAL, val_loss REAL, val_acc REAL, dataset TEXT);"

    c.execute(SQL)

    conn.commit()

    print "Created db"


if __name__ == "__main__":
    

    #v_0_7()

    create_db()
