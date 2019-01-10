import sqlite3
import unittest
import os

from fever.reader.document_database import FEVERDocumentDatabase
from fever.reader.fever_reader import FEVERDatasetReader


class TestFeverReader(unittest.TestCase):

    def setUp(self):
        if os.path.exists("db2.db"):
            os.remove("db2.db")

        conn = sqlite3.connect("db2.db")
        c = conn.cursor()
        c.execute("CREATE TABLE documents (id PRIMARY KEY, text, lines);")

        pairs = [
            ["test1", "Page 1 Line 1\nPage 1 Line 1", "0\tPage 1 Line 1\tLink\n1\tPage 1 Line 2\tLink"],
            ["test2", "Page 2 Line 1\nPage 2 Line 2", "0\tPage 2 Line 1\tLink\n1\tPage 2 Line 2\tLink"],
            ["test3", "Page 3 Line 1\n\nPage 3 Line 3", "0\tPage 3 Line 1\tLink\n"
                                                        "1\t\n"
                                                        "2\tPage 3 Line 3\tLink"],
        ]

        c.executemany("INSERT INTO documents VALUES (?,?,?)", pairs)
        conn.commit()
        conn.close()

    def tearDown(self):
        if os.path.exists("db2.db"):
            os.remove("db2.db")

    def test_fever_db_raises_error_if_not_exists(self):
        with self.assertRaises(FileNotFoundError):
            FEVERDatasetReader("missing.db")

    def test_fever_db_raises_no_error_if_exists(self):
        reader = FEVERDatasetReader("db2.db")

    def test_fever_db_get_page(self):
        reader = FEVERDatasetReader("db2.db")
        self.assertEqual(2,len(reader.get_doc_lines("test2")))

    def test_fever_db_get_line(self):
        reader = FEVERDatasetReader("db2.db")
        self.assertEqual("Page 2 Line 2", reader.get_doc_line("test2",1))

    def test_fever_db_get_random_line(self):
        reader = FEVERDatasetReader("db2.db")
        self.assertEqual("Page 2 Line 2", reader.get_doc_line("test2",-1))

    def test_get_random_line(self):
        db = FEVERDatasetReader("db2.db")
        lines = db.get_doc_lines("test3")
        ne_lines = db.get_non_empty_lines(lines)

        self.assertTrue(ne_lines[0] == lines[0])
        self.assertTrue(ne_lines[1] == lines[2])
        self.assertTrue(len(ne_lines)==2)
        self.assertTrue(len(lines)==3)

    def test_get_non_empty_lines(self):
        lines = ["a","b","c"]
        db = FEVERDatasetReader("db2.db")
        line = db.get_random_line(lines)

        self.assertTrue(line in lines)




