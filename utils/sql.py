import sqlite3


class SqliteOperator:
    def __init__(self, path):
        self.conn = sqlite3.connect(path)
        self.cursor = self.conn.cursor()

    def select(self, q):
        self.cursor.execute(q)
        return self.cursor.fetchall()

    def execute(self, q, param=None):
        if param:
            self.cursor.execute(q, param)
        else:
            self.cursor.execute(q)


    def commit(self):
        self.conn.commit()


    def close(self):
        self.cursor.close()
        self.conn.commit()
        self.conn.close()