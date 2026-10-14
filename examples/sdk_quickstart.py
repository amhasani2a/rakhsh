"""Rakhsh as a Python library — 20 lines from zero to a learning brain.

    PYTHONPATH=. python examples/sdk_quickstart.py
"""
import pathlib, tempfile
from rakhsh import Registry

SAMPLES = pathlib.Path(__file__).resolve().parents[1] / "samples"
reg = Registry(tempfile.mkdtemp(prefix="rakhsh-"))          # use "~/.rakhsh" for a permanent home
brain = reg.brain("cinema")

brain.ingest("cinema_book.txt", (SAMPLES / "cinema_book.txt").read_bytes())
brain.wait()                                                # self-supervised training runs in the background
print(brain.status["message"])

r = brain.ask("فیلمی دربارهٔ آلزایمر پدر و دادگاه")
print("→", r["answer"]["title"], "|", r["answer"]["locator"])
print("  level:", r["meta"]["level_name"], "· spikes:", [s[0] for s in r["meta"]["spikes"][:6]])

brain.feedback(r["id"], "up")                               # 👍  → Hebbian synapses + episodic reflex neuron
r2 = brain.ask("فیلمی دربارهٔ آلزایمر پدر و دادگاه")
print("second time path:", r2["meta"]["path"])              # 'reflex-arc' = answered from memory, no thinking

brain.teach("پیشنهاد امشب چیه؟", answer="امشب «جدایی نادر از سیمین» را ببین.")   # teach brand-new knowledge
print(brain.ask("پیشنهاد امشب چیه؟")["answer"]["text"])
brain.save()
