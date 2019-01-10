import sqlite3
import unittest
import os

from fever.reader.document_database import FEVERDocumentDatabase

class TestFeverDocumentDatabase(unittest.TestCase):

    def setUp(self):
        if os.path.exists("db.db"):
            os.remove("db.db")

        conn = sqlite3.connect("db.db")
        c = conn.cursor()
        c.execute("CREATE TABLE documents (id PRIMARY KEY, text, lines);")

        pairs = [
            ["test1", "Page\nPage", "0\t\Page\tLink\n1\tPage\tLink"],
            ["test2", "Page\nPage", "0\t\Page\tLink\n1\tPage\tLink"],
        ]

        c.executemany("INSERT INTO documents VALUES (?,?,?)", pairs)
        conn.commit()
        conn.close()

    def tearDown(self):
        if os.path.exists("db.db"):
            os.remove("db.db")

    def test_fever_db_raises_error_if_not_exists(self):
        with self.assertRaises(FileNotFoundError):
            FEVERDocumentDatabase("missing.db")

    def test_fever_db_raises_no_error_if_exists(self):
        db = FEVERDocumentDatabase("db.db")

    def test_fever_db_raises_no_error_if_get_existing_page(self):
        db = FEVERDocumentDatabase("db.db")
        self.assertEqual(2,len(db.get_doc_lines("test1")))

    def test_fever_db_raises_error_if_get_missing_page(self):
        db = FEVERDocumentDatabase("db.db")
        with self.assertRaises(Exception):
            db.get_doc_lines("test3_missing")

