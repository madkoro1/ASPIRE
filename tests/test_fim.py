import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from aspire.training.fim import ast_spans, encode_answer, infill_record, match_random, split_source
from aspire.training.prepare import prepare


class CharacterTokenizer:
    is_fast = True
    eos_token_id = 0

    def __call__(self, text, **kwargs):
        return dict(input_ids=[ord(c) for c in text],
                    offset_mapping=[(i, i + 1) for i in range(len(text))])


class DataTests(unittest.TestCase):
    def test_utf8_negative_comparison_and_reconstruction(self):
        source = 'p("伤").\r\nq(X):-p(X),not blocked(X),X<=5.\r\n'
        spans = ast_spans(source)
        pieces = {source[s['start']:s['end']]: s['kind'] for s in spans}
        self.assertEqual(pieces['not blocked(X)'], 'body_literal')
        self.assertEqual(pieces['X<=5'], 'comparison')
        self.assertNotIn('p("伤").', pieces)  # Facts are context, not targets.
        for span in spans:
            self.assertEqual(''.join(split_source(source, span)), source)
        source = 'q(X):-p("伤",X),not r(X).'
        self.assertIn('not r(X)', [source[s['start']:s['end']] for s in ast_spans(source)])

    def test_whole_program_and_unsafe_input(self):
        source = 'p(X):-q(X).'
        self.assertFalse(any(s['kind'] == 'rule' for s in ast_spans(source)))
        self.assertEqual(ast_spans('p(1).'), [])
        with self.assertRaises(ValueError):
            ast_spans('#include "anything.lp".')

    def test_answer_mask_eos_and_fail_closed(self):
        encoded, stats = encode_answer(CharacterTokenizer(), 'q', 'p(1).')
        self.assertEqual(encoded['labels'][-6:], [ord(c) for c in 'p(1).'] + [0])
        self.assertTrue(all(x == -100 for x in encoded['labels'][:-6]))
        self.assertEqual(stats['supervised_tokens'], 6)
        with self.assertRaisesRegex(ValueError, 'overlength'):
            encode_answer(CharacterTokenizer(), 'q', 'p(1).', limit=3)
        with self.assertRaisesRegex(ValueError, 'empty answer'):
            encode_answer(CharacterTokenizer(), 'q', ' ')

    def test_boundary_crossing_rejected(self):
        class Bad(CharacterTokenizer):
            def __call__(self, text, **kwargs):
                return dict(input_ids=[1], offset_mapping=[(0, len(text))])
        with self.assertRaisesRegex(ValueError, 'crosses'):
            encode_answer(Bad(), 'q', 'p.')

    def test_delimiter_collision_and_whitespace(self):
        source = 'p(1).\nq(X):-p(X).\n'
        span = dict(start=6, end=len(source) - 1)
        row = infill_record('<ASP_FIM_TASK>', source, span)
        self.assertIn('<ASP_FIM__TASK>', row['question'])
        self.assertEqual(row['answer'], source[6:-1])
        self.assertEqual(''.join(split_source(source, span)), source)

    def test_random_match_and_unmatchable(self):
        tok, source = CharacterTokenizer(), 'p(1).\nq(X):-p(X).'
        span, stats, _ = match_random(tok, 'q', source, 6, random.Random(2), attempts=512)
        self.assertIsNotNone(span)
        self.assertEqual(stats['supervised_tokens'], 6)
        self.assertEqual(''.join(split_source(source, span)), source)
        span, _, _ = match_random(tok, 'q', source, 1000, random.Random(2))
        self.assertIsNone(span)

    def test_freezing_overlap_shared_slots_and_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [dict(question='derive q', answer='p(1).\nq(X):-p(X),X<=5.'),
                       dict(question='fact', answer='p(1).')]
            (root / 'train.jsonl').write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
            val = [records[0], dict(question='different', answer='r(1).')]
            (root / 'val.jsonl').write_text('\n'.join(json.dumps(r) for r in val), encoding='utf-8')
            for folder in ('a', 'b'):
                report = prepare(root / 'train.jsonl', root / 'val.jsonl', root / folder,
                                 CharacterTokenizer(), rate=1, attempts=2000)
            self.assertEqual(report['accepted_fim_pairs'], 1)
            self.assertEqual(report['fallback_counts'], {'no_supported_span': 1})
            self.assertEqual(report['overlap']['exact_qa_rows'], 1)
            self.assertEqual(report['validation_exact_qa_nonoverlap_rows'], 1)
            for name in ('plain.jsonl', 'ast_fim.jsonl', 'random_fim.jsonl', 'audit.jsonl'):
                self.assertEqual((root / 'a' / name).read_bytes(), (root / 'b' / name).read_bytes())
            arms = {name: [json.loads(s) for s in (root / 'a' / (name + '.jsonl')).read_text().splitlines()]
                    for name in ('plain', 'ast_fim', 'random_fim')}
            for i in range(2):
                self.assertEqual(len({arms[n][i]['source_id'] for n in arms}), 1)
            for n in ('ast_fim', 'random_fim'):
                self.assertEqual(arms[n][1], arms['plain'][1])
                row = arms[n][0]
                self.assertEqual(row['prefix'] + row['answer'] + row['suffix'], records[0]['answer'])
            with self.assertRaisesRegex(ValueError, 'never overwritten'):
                prepare(root / 'train.jsonl', root / 'val.jsonl', root / 'a', CharacterTokenizer())

    def test_pair_fallback_and_failed_preflight_no_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / 'train.jsonl'
            val = root / 'val.jsonl'
            train.write_text(json.dumps(dict(question='q', answer='p(1).\nq(X):-p(X).')), encoding='utf-8')
            val.write_text(json.dumps(dict(question='v', answer='v(1).')), encoding='utf-8')
            with patch('aspire.training.prepare.match_random', return_value=(None, None, 512)):
                report = prepare(train, val, root / 'a', CharacterTokenizer(), rate=1)
            self.assertEqual(report['fallback_counts'], {'no_exact_length_match': 1})
            self.assertFalse(report['ready_for_smoke'])
            self.assertEqual((root / 'a/plain.jsonl').read_bytes(), (root / 'a/ast_fim.jsonl').read_bytes())
            self.assertEqual((root / 'a/plain.jsonl').read_bytes(), (root / 'a/random_fim.jsonl').read_bytes())
            with self.assertRaisesRegex(ValueError, 'overlength'):
                prepare(train, val, root / 'bad', CharacterTokenizer(), limit=5)
            self.assertFalse((root / 'bad').exists())

    def test_verify_frozen_accepts_every_training_arm(self):
        from aspire.training.common import file_hash
        from aspire.training.frozen import verify_frozen
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, model = root / 'fim', root / 'model'
            model.mkdir()
            (model / 'tokenizer.json').write_text('{}', encoding='utf-8')
            train = root / 'train.jsonl'
            val = root / 'val.jsonl'
            train.write_text(json.dumps(dict(question='q', answer='p(1).\nq(X):-p(X),X<=5.')), encoding='utf-8')
            val.write_text(json.dumps(dict(question='v', answer='v(1).')), encoding='utf-8')
            prepare(train, val, data, CharacterTokenizer(), rate=1, attempts=2000, tokenizer_path=model)
            for arm in ('plain', 'random_fim', 'ast_fim', 'ast_any'):
                self.assertEqual(verify_frozen(data / f'{arm}.jsonl', data / 'validation.jsonl', model),
                                 file_hash(data / 'manifest.json'))
            with self.assertRaisesRegex(ValueError, 'frozen arm'):
                verify_frozen(data / 'audit.jsonl', data / 'validation.jsonl', model)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires torch on server')
class RuntimeTests(unittest.TestCase):
    def test_padding_preserves_eos(self):
        from aspire.training.fim import AnswerCollator
        batch = AnswerCollator(0)([dict(input_ids=[1, 2, 0], labels=[-100, 2, 0]),
                                  dict(input_ids=[1, 0], labels=[-100, 0])])
        self.assertEqual(batch['labels'].tolist(), [[-100, 2, 0], [-100, 0, -100]])
        self.assertEqual(batch['attention_mask'].tolist(), [[1, 1, 1], [1, 1, 0]])


if __name__ == '__main__':
    unittest.main()
