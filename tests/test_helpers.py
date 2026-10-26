import unittest

from src.chunking import create_chunks


class ChunkingTests(unittest.TestCase):
    def test_heading_metadata_is_preserved(self):
        docs = [{"doc_name": "example.md", "text": "# Root\n\nIntro.\n\n## Steps\n\nDo this first.\n\nDo this second."}]
        chunks = create_chunks(docs, max_chars=200, overlap=20)
        self.assertTrue(chunks)
        self.assertTrue(any(chunk["section"] == "Steps" for chunk in chunks))

    def test_long_paragraph_is_split_into_nonempty_chunks(self):
        text = "word " * 300
        chunks = create_chunks([{"doc_name": "long.md", "text": "# Long\n\n" + text}], max_chars=100, overlap=10)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(str(chunk["text"]).strip() for chunk in chunks))

    def test_article_metadata_is_present(self):
        chunks = create_chunks([{"doc_name": "wifi.md", "text": "# Wi-Fi\n\n## Restart\n\nRestart the device."}])
        self.assertEqual(chunks[0]["doc_name"], "wifi.md")
        self.assertIn("text", chunks[0])


if __name__ == "__main__":
    unittest.main()
