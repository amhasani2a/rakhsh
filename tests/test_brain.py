import unittest
from rakhsh import Registry, NotReady, Brain
from tests._util import trained_registry, fast_cfg

Q_FORGET = "فیلمی درباره فراموشی و قتل"


class TestBrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = trained_registry(); cls.b = cls.reg.brain("lib")

    def test_untrained_brain_refuses(self):
        with self.assertRaises(NotReady):
            Brain("empty", fast_cfg()).search("سلام")

    def test_ask_returns_quote_with_trace(self):
        r = self.b.ask("نویسنده بوف کور")
        self.assertIn("answer", r); self.assertIn("meta", r)
        m = r["meta"]
        self.assertIn(m["level_name"], ("reflex", "spike", "deep"))
        self.assertAlmostEqual(sum(m["stop"]), 1.0, places=3)      # hazard gate = distribution over levels
        self.assertTrue(r["answer"]["source"])

    def test_probe_is_decision_only(self):
        p = self.b.probe("جدایی نادر از سیمین")
        self.assertIn("stop", p)

    def test_teach_new_knowledge_is_answerable(self):
        self.b.teach("رنگ مورد علاقه مدیر", answer="رنگ مورد علاقه مدیر شرکت آبی فیروزه‌ای است.", actor="human")
        # the exact question is a reflex immediately; paraphrases work after the background retrain
        self.assertIn("فیروزه", self.b.ask("رنگ مورد علاقه مدیر")["answer"]["text"])
        self.b.wait()
        self.assertIn("فیروزه", self.b.ask("مدیر چه رنگی دوست دارد")["answer"]["text"])

    def test_trust_builds_reflex_arc(self):
        t = self.b.teach(Q_FORGET, answer="لئونارد شلبی مردی است که پس از یک حمله دچار اختلال حافظه", actor="agent:claude")
        self.b.teach(Q_FORGET, chunk=t["chunk"], actor="agent:claude")      # 0.8 + 0.8 >= threshold
        r = self.b.ask(Q_FORGET)
        self.assertEqual(r["meta"]["path"], "reflex-arc")
        self.assertEqual(r["answer"]["chunk"], t["chunk"])

    def test_low_trust_agent_cannot_write_reflex(self):
        q = "سوال آزمایشی برای عامل کم‌اعتماد"
        r = self.b.ask(q)
        self.b.feedback(r["id"], "up", actor="agent:spam")
        self.assertNotEqual(self.b.ask(q)["meta"]["path"], "reflex-arc")

    def test_feedback_pick_and_sleep(self):
        r = self.b.ask("فیلم ترسناک")
        alt = r["hits"][-1]["chunk"]
        self.b.feedback(r["id"], "pick", chunk=alt, actor="human")
        out = self.b.sleep()
        self.assertIsInstance(out, dict)

    def test_state_shape(self):
        s = self.b.state()
        for k in ("name", "status", "chunks", "neurons", "episodes", "sources"):
            self.assertIn(k, s)


class TestStorage(unittest.TestCase):
    def test_roundtrip_and_snapshot(self):
        reg = trained_registry(); b = reg.brain("lib")
        before = b.search("بوف کور")["ranking"][0][0]
        sid = reg.snapshot("lib", "t")
        self.assertIn(sid, reg.snapshots("lib"))
        b.save()
        b2 = Registry(reg.home, brain_cfg=fast_cfg()).brain("lib", create=False)
        self.assertEqual(b2.state()["chunks"], b.state()["chunks"])
        self.assertEqual(b2.search("بوف کور")["ranking"][0][0], before)

    def test_no_pickle_on_disk(self):
        reg = trained_registry(); reg.brain("lib").save()
        exts = {p.suffix for p in (reg.home / "brains" / "lib").rglob("*") if p.is_file()}
        self.assertTrue(exts <= {".json", ".npz", ".jsonl"}, exts)


if __name__ == "__main__":
    unittest.main()
