import unittest
from rakhsh import ingest, register_loader
from rakhsh.ingest import supported_extensions
from tests._util import SAMPLES


class TestIngest(unittest.TestCase):
    def test_text_chunks_have_locators(self):
        chunks = ingest("cinema_book.txt", (SAMPLES / "cinema_book.txt").read_bytes())
        self.assertGreater(len(chunks), 10)
        self.assertTrue(all(c.text and c.locator for c in chunks))

    def test_rows_from_csv_sqlite_json(self):
        for f in ("movies.csv", "library.sqlite", "notes.json"):
            chunks = ingest(f, (SAMPLES / f).read_bytes())
            self.assertGreater(len(chunks), 0, f)

    def test_magic_bytes_beat_extension(self):
        # an SQLite file with a wrong extension must still be detected
        chunks = ingest("library.dat", (SAMPLES / "library.sqlite").read_bytes())
        self.assertGreater(len(chunks), 0)

    def test_register_custom_loader(self):
        from rakhsh import Chunk
        register_loader([".rak"], lambda data, name, max_words: [Chunk(source=name, locator="1", title="x", text=data.decode())])
        self.assertIn("rak", supported_extensions())
        self.assertEqual(ingest("a.rak", "سلام رخش".encode())[0].text, "سلام رخش")


if __name__ == "__main__":
    unittest.main()
