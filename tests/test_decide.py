import tempfile, unittest
from rakhsh import Registry

DATA = [("سلام", "no-llm"), ("ساعت چنده", "no-llm"), ("ممنون", "no-llm"),
        ("این ایمیل را خلاصه کن", "small-llm"), ("این متن را ترجمه کن", "small-llm"), ("غلط املایی را درست کن", "small-llm"),
        ("یک استراتژی بازاریابی پنج ساله برای استارتاپ بنویس", "large-llm"),
        ("این قرارداد حقوقی را تحلیل و ریسک‌ها را بگو", "large-llm"), ("معماری یک سیستم توزیع‌شده را طراحی کن", "large-llm")]


class TestDecider(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = Registry(tempfile.mkdtemp())
        cls.d = cls.reg.create_decider("router", ["no-llm", "small-llm", "large-llm"], "ordinal", "route prompts")
        cls.d.teach([{"text": t, "label": l} for t, l in DATA])

    def test_few_shot_routing(self):
        self.assertEqual(self.d.decide("سلام خوبی")["label"], "no-llm")
        self.assertEqual(self.d.decide("این پاراگراف را ترجمه کن")["label"], "small-llm")

    def test_has_confidence_and_evidence(self):
        r = self.d.decide("یک برنامه تجاری کامل طراحی کن")
        self.assertIn("confidence", r); self.assertIn("evidence", r)

    def test_persistence(self):
        self.d.save()
        d2 = Registry(self.reg.home).decider("router")
        self.assertEqual(d2.decide("سلام")["label"], self.d.decide("سلام")["label"])

    def test_bad_labels(self):
        from rakhsh import RakhshError
        with self.assertRaises(RakhshError):
            self.reg.create_decider("bad", ["a"])


if __name__ == "__main__":
    unittest.main()
