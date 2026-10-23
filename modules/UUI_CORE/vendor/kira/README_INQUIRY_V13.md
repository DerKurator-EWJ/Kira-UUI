# KIRA Unified Core 1.3 – Inquiry Controller

Adds `inquiry_controller.py` and connects it to `EvaluatedNumericStream.inquiry`.

- `ask_about(subject,predicate,context=...)`: ask about ordinary unknown relations without any prediction error.
- `sync_reviews()`: derive review questions from open evidence contradictions.
- `next_questions(limit)`: persisted priority-ordered queue.
- `answer(inquiry_id, answer_ref)`: record answer provenance; does **not** resolve underlying evidence review.
- `dismiss(inquiry_id)`: close a question without changing evidence.
- Deduplicates unknown/context questions and per-review contradiction questions.

## Example

```python
from evaluation_bridge import EvaluatedNumericStream
stream = EvaluatedNumericStream('kira.sqlite')
stream.inquiry.ask_about('Stuhl', 'Verwendung')
print(stream.inquiry.next_questions())
stream.close()
```

## Limitations

This is a deterministic inquiry **queue and question-template generator**, not autonomous experimental design, language understanding, or independent scientific reasoning. A future controller must supply candidate actions, costs, safety constraints and answer validation. Existing evidence remains immutable and must be reviewed explicitly. SQLite transactions are not yet atomic across all numeric-stream writes.

Run `python -m unittest discover -p 'test_*.py'` from this directory.
