import unittest

from src.inspire.pipeline import build_query, chunk_text, load_config, pack_evidence, prepare


class PipelineTests(unittest.TestCase):
    def test_query_is_the_case_text_without_intent(self):
        result = prepare("  虚构患儿 乏力  ", [{"id": "D1", "text": "虚构患儿乏力"}])
        self.assertEqual(result["query"], "虚构患儿 乏力")
        self.assertIsNone(result["intent_recognition"])
        self.assertIsNone(result["diagnoses"])

    def test_empty_case_is_rejected(self):
        with self.assertRaises(ValueError):
            build_query("   ")

    def test_overlap_is_shared_by_neighboring_chunks(self):
        text = " ".join(f"w{i}" for i in range(600))
        chunks = chunk_text(text, size=512, overlap=64)
        self.assertEqual(chunks[0]["token_count"], 512)
        self.assertEqual(chunks[1]["start_token"], 448)
        self.assertEqual(chunks[1]["token_count"], 152)
        self.assertEqual(chunks[0]["text"].split()[-64:], chunks[1]["text"].split()[:64])

    def test_ten_full_chunks_stop_at_the_token_limit(self):
        chunks = [{"id": i, "token_count": 512, "text": "x"} for i in range(10)]
        packed = pack_evidence(chunks, max_chunks=10, max_tokens=4000)
        self.assertEqual(packed["chunk_count"], 7)
        self.assertEqual(packed["token_count"], 3584)

    def test_short_chunks_can_reach_the_count_limit(self):
        chunks = [{"id": i, "token_count": 100, "text": "x"} for i in range(15)]
        packed = pack_evidence(chunks, max_chunks=10, max_tokens=4000)
        self.assertEqual(packed["chunk_count"], 10)
        self.assertEqual(packed["token_count"], 1000)

    def test_packing_does_not_skip_ahead(self):
        chunks = [
            {"id": "a", "token_count": 3000, "text": "a"},
            {"id": "b", "token_count": 2000, "text": "b"},
            {"id": "c", "token_count": 100, "text": "c"},
        ]
        packed = pack_evidence(chunks, max_chunks=10, max_tokens=4000)
        self.assertEqual([item["id"] for item in packed["chunks"]], ["a"])

    def test_public_config_matches_the_reported_limits(self):
        config = load_config()
        self.assertEqual(config["chunk_size_tokens"], 512)
        self.assertEqual(config["chunk_overlap_tokens"], 64)
        self.assertEqual(config["dense_candidates"], 20)
        self.assertEqual(config["selected_chunks"], 10)
        self.assertEqual(config["evidence_token_limit"], 4000)
        self.assertFalse(config["intent_recognition"])


if __name__ == "__main__":
    unittest.main()
