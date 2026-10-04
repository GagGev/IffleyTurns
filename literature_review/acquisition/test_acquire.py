"""Safety properties that matter for a scientific training corpus."""
import importlib.util
import sqlite3
import unittest
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('acquire', Path(__file__).with_name('acquire.py'))
acquire = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acquire)


class AcquisitionTests(unittest.TestCase):
    def test_license_is_explicit_and_conflicts_fail_closed(self):
        template = '<article><front><article-meta><permissions><license>{}</license></permissions></article-meta></front></article>'
        cases = [('https://creativecommons.org/licenses/by/4.0/', True),
                 ('https://creativecommons.org/publicdomain/zero/1.0/', True),
                 ('https://creativecommons.org/licenses/by-nc/4.0/', False),
                 ('https://creativecommons.org/licenses/by-nd/4.0/', False),
                 ('https://creativecommons.org/licenses/by-sa/4.0/', False),
                 ('All rights reserved', False),
                 ('https://creativecommons.org/publicdomain/zero/1.0/ applies to data only', False),
                 ('https://creativecommons.org/licenses/by/4.0/ non-commercial', False)]
        for text, expected in cases:
            with self.subTest(text=text):
                _, allowed = acquire.license_from_xml(ET.fromstring(template.format(text)))
                self.assertEqual(allowed, expected)

    def test_transitive_paper_pair_family_components(self):
        components = acquire.connected_components([
            ('paper:1', 'pair:A'), ('paper:1', 'pair:B'), ('paper:2', 'pair:B'),
            ('paper:2', 'family:X'), ('paper:3', 'family:X'), ('paper:4', 'pair:C')])
        self.assertEqual(components['pair:A'], components['paper:3'])
        self.assertNotEqual(components['pair:A'], components['pair:C'])

    def test_missing_is_not_zero_and_scores_require_passages(self):
        with sqlite3.connect(':memory:') as db:
            db.executescript(acquire.SCHEMA)
            db.execute("INSERT INTO disease_pair_observations(observation_id,ordinal_score,dimension_applicability) VALUES ('missing',NULL,0)")
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute("INSERT INTO disease_pair_observations(observation_id,ordinal_score) VALUES ('bad',0)")
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute("""INSERT INTO disease_pair_observations(observation_id,ordinal_score,dimension_applicability,
                           source_locator,source_passage_or_paraphrase) VALUES ('na-zero',0,0,'p1','evidence')""")
            db.execute("""INSERT INTO disease_pair_observations(observation_id,ordinal_score,dimension_applicability,
                           source_locator,source_passage_or_paraphrase) VALUES ('explicit-zero',0,2,'p1','evidence')""")
            self.assertIsNone(db.execute("SELECT ordinal_score FROM disease_pair_observations WHERE observation_id='missing'").fetchone()[0])

    def test_mention_boundaries_do_not_match_substrings(self):
        self.assertIsNone(acquire.mention('This is a test of ALSA.', ['ALS']))
        self.assertEqual(acquire.mention('Marfan syndrome, compared with X.', ['Marfan syndrome']), 'Marfan syndrome')

    def test_doi_deduplication(self):
        self.assertEqual(acquire.normalize_doi('https://doi.org/10.1234/ABC'), '10.1234/abc')
        self.assertEqual(acquire.normalize_doi(' doi:10.1234/ABC '), '10.1234/abc')

    def test_shared_study_family_blocks_papers_and_quarantines_time_overlap(self):
        with sqlite3.connect(':memory:') as db:
            db.row_factory = sqlite3.Row
            db.executescript(acquire.SCHEMA)
            diseases = [{'orpha_id': x, 'search_terms': ['Disease ' + x]} for x in ('A', 'B', 'C')]
            for d in diseases:
                acquire.insert(db, 'diseases', {'orpha_id': d['orpha_id']})
            acquire.insert(db, 'study_families', dict(study_family_id='family', review_status='reviewed'))
            for pid, a, b, year in [('paper1', 'A', 'B', 2022), ('paper2', 'B', 'C', 2024)]:
                pair_id = a + '|' + b
                acquire.insert(db, 'pairs', dict(pair_id=pair_id, orpha_id_a=a, orpha_id_b=b))
                acquire.insert(db, 'papers', dict(paper_id=pid, study_family_id='family', year=year, metadata_json='{}'))
                acquire.insert(db, 'search_log', dict(query_id=pid, track='A', target_id=pair_id))
                acquire.insert(db, 'discoveries', dict(paper_id=pid, query_id=pid))
            acquire.assign_splits(db, {'diseases': diseases})
            splits = db.execute("SELECT DISTINCT split FROM split_assignments WHERE regime='paper_pair_family'").fetchall()
            self.assertEqual(len(splits), 1)
            times = [r[0] for r in db.execute("SELECT split FROM split_assignments WHERE regime='time'")]
            self.assertEqual(times, ['quarantine', 'quarantine'])

    def test_raw_source_tampering_is_detected(self):
        out = acquire.ROOT / '.data'
        path = out / ('test-response-' + uuid.uuid4().hex)
        try:
            with sqlite3.connect(':memory:') as db:
                db.row_factory = sqlite3.Row
                db.executescript(acquire.SCHEMA)
                path.write_text('altered')
                acquire.insert(db, 'requests', dict(request_id='r', raw_path=path.name, response_sha256=acquire.digest('original')))
                self.assertIn('raw_checksum:r', acquire.validate(db, out))
        finally:
            path.unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()
