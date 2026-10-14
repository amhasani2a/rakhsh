"""Teacher → student: a big LLM labels a few prompts once; a tiny Rakhsh Decider learns to route
every future prompt in microseconds (no LLM call needed to decide which LLM to call).

    PYTHONPATH=. python examples/llm_router_teacher_student.py
"""
import tempfile
from rakhsh import Registry, TrustPolicy

reg = Registry(tempfile.mkdtemp(), trust=TrustPolicy(agents={"teacher-llm": 0.9}))
router = reg.create_decider("llm-router", ["no-llm", "small-llm", "large-llm"], mode="ordinal",
                            description="cheapest model that can handle the prompt")

# In production these labels come from the teacher LLM via MCP tool `decider_teach`.
teacher_labels = {
    "no-llm":    ["سلام", "ممنون", "ساعت چنده", "خداحافظ"],
    "small-llm": ["این ایمیل را خلاصه کن", "این متن را ترجمه کن", "غلط املایی را درست کن", "یک عنوان کوتاه بده"],
    "large-llm": ["یک استراتژی بازاریابی پنج ساله بنویس", "این قرارداد را تحلیل و ریسک‌ها را بگو",
                  "معماری یک سیستم توزیع‌شده را طراحی کن", "این مقالهٔ علمی را نقد کن"],
}
router.teach([{"text": t, "label": l} for l, ts in teacher_labels.items() for t in ts], actor="agent:teacher-llm")

for prompt in ["سلام خوبی؟", "این پاراگراف را به انگلیسی ترجمه کن", "یک برنامهٔ تجاری کامل طراحی کن", "هوا"]:
    d = router.decide(prompt)
    print(f"{prompt:40s} → {d['label']:10s} conf={d['confidence']:.2f}  abstained={d['abstained']}  evidence={d.get('evidence', [])[:3]}")
