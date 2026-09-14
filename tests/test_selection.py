from __future__ import annotations

import gzip
import io
import json
import tarfile
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from influpaintx.data.catalog import CATALOG
from influpaintx.data.selection import (
    SelectedData, NHSN_DELPHI, NHSN_MEASURES, describe, family, measure_columns,
)
from influpaintx.explorer import ExplorerIndex
from influpaintx.data.tables import RawTables


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.specs = []

    def snapshot(self, key, rows=None, members=None):
        spec = CATALOG[key].to_dict()
        self.specs.append(spec)
        (self.root / 'catalog.json').write_text(json.dumps({'datasets': self.specs}))
        folder = self.root / 'raw' / key / 'snapshots' / 's1'
        folder.mkdir(parents=True)
        (folder.parent.parent / 'latest.json').write_text(json.dumps({'snapshot_id': 's1'}))
        if members is not None:
            name = 'hub-data.tar.gz'
            with tarfile.open(folder / name, 'w:gz') as tar:
                for member, content in members.items():
                    data = content.encode()
                    info = tarfile.TarInfo(member)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
        else:
            name = 'data.ndjson.gz'
            with gzip.open(folder / name, 'wt') as f:
                for row in rows:
                    f.write(json.dumps(row) + '\n')
        (folder / 'manifest.json').write_text(json.dumps({
            'files': [{'path': name}], 'retrieved_at': '2026-02-01T12:00:00Z',
        }))

    def test_registry_and_exact_nhsn_counts(self):
        self.assertEqual(len(CATALOG), 25)
        self.assertEqual(len({family(k) for k in CATALOG}), 15)
        self.assertEqual(len(NHSN_MEASURES), 14)
        dropped = ['totalconfflunewadmper100k', 'pctconfflunewadmadult',
                   'totalconfflunewadmcumulativeseasonalsum',
                   'new_future_numeric_field', 'totalconfflunewadmadult',
                   'totalconfflunewadmped', 'numconfflunewadmped0to4',
                   'numconffluhosppatsadult', 'numconffluicupatsped', 'numconfflunewadmunk']
        self.assertEqual(measure_columns('cdc_nhsn_final', [
            'totalconfflunewadm', 'totalconfflunewadmhosprep', 'numinptbeds', *dropped]),
            ('totalconfflunewadm', 'totalconfflunewadmhosprep', 'numinptbeds'))

    def test_geography_filter_precedes_both_consumers(self):
        rows = [
            {'geography': 'New York', 'county': 'All', 'percent_visits': 1},
            {'geography': 'United States', 'percent_visits': 2},
            {'geography': 'New York', 'county': 'Albany', 'percent_visits': 91},
            {'geography': 'New York', 'hsa_nci_id': '123', 'percent_visits': 92},
            {'geography': 'New York', 'site': 'NY1', 'percent_visits': 93},
            {'geography': 'New York', 'geo_type': 'hhs', 'percent_visits': 94},
            {'geography': 'Region 2', 'percent_visits': 95},
        ]
        self.snapshot('cdc_nssp_daily', [dict(date='2026-01-03', **r) for r in rows])
        records = list(SelectedData(self.root).iter_records())
        self.assertEqual([r.values['percent_visits'] for r in records], [1, 2])
        self.assertEqual([r.metadata['geography'] for r in records], ['New York', 'United States'])
        index = ExplorerIndex(self.root)
        index.build(progress=lambda _: None)
        items = index.list_series('NY')['items']
        self.assertEqual(len(items), 2)
        self.assertEqual(sorted(index.data('NY', [i['id']])['series'][0]['points'][0][1]
                                for i in items), [1, 2])

    def test_excluded_partitions_are_not_opened(self):
        self.snapshot('delphi_nssp', [])
        reader = SelectedData(self.root)
        artifact = reader.artifacts()[0]
        for geography in ('county', 'hsa', 'hhs', 'sewershed', 'census_region'):
            with self.subTest(geography=geography), patch.object(
                RawTables, '_table_sources', side_effect=AssertionError('must not open payload')
            ):
                excluded = replace(artifact, relative_path=f'signal=pct_ed_visits_covid/geo_type={geography}/archive.csv.gz')
                self.assertEqual(list(reader._table_sources(excluded)), [])

    def test_site_catalog_with_state_context_is_excluded_before_reading(self):
        self.snapshot('cdc_nwss_wval', [{'state_territory': 'NY', 'site': '1', 'site_wval': 10}])
        with patch.object(RawTables, '_table_sources', side_effect=AssertionError('must not read sites')):
            self.assertEqual(list(SelectedData(self.root).iter_records()), [])

    def test_national_source_without_geography_columns_is_retained(self):
        self.snapshot('cdc_nrevss_national', [{'week_end': '2026-01-03', 'percent_positive': 5}])
        self.assertEqual(len(list(SelectedData(self.root).iter_records())), 1)

    def test_catchment_state_labels_are_not_state_observations(self):
        self.snapshot('hub_rsvnet', members={
            'target-data/2026-01-09_rsvnet_hospitalization.csv':
                'location,date,value\nMN,2026-01-03,99\nUS,2026-01-03,8\n',
        })
        records = list(SelectedData(self.root).iter_records())
        self.assertEqual([r.metadata['location'] for r in records], ['US'])

    def test_finer_hub_rows_are_removed_before_conflict_validation(self):
        self.snapshot('hub_flusight_current', members={
            'target-data/time-series.csv':
                'location,target_end_date,as_of,target,county,observation\n'
                '36,2026-01-03,2026-01-09,wk inc flu hosp,All,8\n'
                '36,2026-01-03,2026-01-09,wk inc flu hosp,Albany,99\n',
        })
        reader = SelectedData(self.root)
        self.assertEqual([r.values['observation'] for r in reader.iter_records()], ['8'])
        self.assertFalse(any(r['status'] == 'quarantined' for r in reader.audit))

    def test_native_records_preserve_metadata_and_bound_snapshot_availability(self):
        self.snapshot('cdc_nhsn_final', [{
            'jurisdiction': 'NY', 'weekendingdate': '2026-01-03',
            'totalconfflunewadm': 17, 'totalconfflunewadmper100k': 2,
            'totalconfflunewadmhosprep': 9,
        }])
        reader = SelectedData(self.root)
        records = list(reader.iter_records(group='nhsn'))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].values, {
            'totalconfflunewadm': 17, 'totalconfflunewadmhosprep': 9,
        })
        self.assertEqual(records[0].metadata['jurisdiction'], 'NY')
        self.assertEqual(records[0].signals['totalconfflunewadm']['spatial_support'], 'native state')
        self.assertNotIn('totalconfflunewadmhosprep', records[0].metadata)
        self.assertEqual(list(reader.iter_records(available_by='2026-01-31T23:59:59Z')), [])
        self.assertEqual(len(list(reader.iter_records(available_by='2026-02-01T12:00:00Z'))), 1)

    def test_delphi_nhsn_only_admission_values_and_all_eligible_revisions(self):
        self.snapshot('delphi_nhsn', [
            dict(signal=signal, geo_type='state', geo_value='ny', reference_time='2026-01-03',
                 report_time=release, fill_method='', value=value)
            for signal, release, value in [
                ('confirmed_admissions_flu_ew', '2026-01-09 09:00:00', 8),
                ('confirmed_admissions_flu_ew', '2026-01-09 18:00:00', 12),
                ('inpatient_beds_ew', '2026-01-09 09:00:00', 100),
                ('hosprep_confirmed_admissions_flu_ew', '2026-01-09 09:00:00', 4),
            ]])
        records = list(SelectedData(self.root).iter_records())
        self.assertEqual([r.values['value'] for r in records], [8, 12, 100, 4])
        earlier = list(SelectedData(self.root).iter_records(available_by='2026-01-09T12:00:00Z'))
        self.assertEqual([r.values['value'] for r in earlier], [8, 100, 4])

    def test_hub_uses_canonical_observations_only_and_handles_retractions(self):
        self.snapshot('hub_flusight_current', members={
            'target-data/time-series.csv':
                'location,target_end_date,as_of,target,observation\n'
                '36,2026-01-03,2026-01-09,wk inc flu hosp,8\n'
                '36,2025-12-27,2026-01-09,wk inc flu hosp,9\n'
                '36,2026-01-03,2026-01-16,wk inc flu hosp,12\n',
            'target-data/oracle-output.csv':
                'location,target_end_date,target,oracle_value\n36,2026-01-03,wk inc flu hosp,999\n',
            'target-data/target-hospital-admissions.csv':
                'location,target_end_date,value\n36,2026-01-03,777\n',
        })
        reader = SelectedData(self.root)
        records = list(reader.iter_records())
        self.assertEqual([r.values for r in records], [{'observation': '8'},
                                                     {'observation': '9'}, {'observation': '12'}])
        index = ExplorerIndex(self.root)
        index.build(progress=lambda _: None)
        items = index.list_series('NY')['items']
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]['signal_key'], 'nhsn:totalconfflunewadm')
        self.assertEqual(index.data('NY', [items[0]['id']])['series'][0]['points'], [['2026-01-03', 12.0, 1]])
        before = index.data('NY', [items[0]['id']], as_of='2026-01-09')['series'][0]['points']
        self.assertEqual(len(before), 2)
        self.assertTrue(any(w['status'] == 'excluded' for w in index.overview()['warnings']))

    def test_conflicts_are_missing_without_losing_other_truth_or_release_dates(self):
        self.snapshot('hub_flusight_current', members={
            'target-data/time-series.csv':
                'location,target_end_date,as_of,target,observation\n'
                '36,2026-01-03,2026-01-09,wk inc flu hosp,8\n'
                '36,2026-01-03,2026-01-16,wk inc flu hosp,12\n'
                '36,2026-01-10,2026-01-16,wk inc flu hosp,20\n'
                '36,2026-01-10,2026-01-16,wk inc flu hosp,20\n'
                '36,2026-01-03,2026-01-16,wk inc flu hosp,99\n',
        })
        reader = SelectedData(self.root)
        rows = list(reader.iter_records())
        self.assertEqual([r.values['observation'] for r in rows], ['8', None, '20'])
        self.assertTrue(rows[1].metadata['_selection_conflict'])
        self.assertEqual(len(reader.audit), 1)
        index = ExplorerIndex(self.root)
        index.build(progress=lambda _: None)
        item = index.list_series('NY')['items'][0]
        latest = index.data('NY', [item['id']])['series'][0]['points']
        self.assertEqual(latest, [['2026-01-10', 20.0, 1]])
        previous = index.data('NY', [item['id']], as_of='2026-01-09')['series'][0]['points']
        self.assertEqual(previous, [['2026-01-03', 8.0, 1]])

    def test_lfs_truth_unavailable_without_provider_substitution(self):
        self.snapshot('hub_covid_legacy', members={
            'data-truth/truth-Incident Cases.csv': 'version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 12\n',
            'data-truth/usafacts/truth_usafacts-Incident Cases.csv':
                'location,date,value\n36,2026-01-03,999\n',
        })
        with self.assertRaisesRegex(ValueError, 'Git LFS pointer'):
            list(SelectedData(self.root).iter_records())
        index = ExplorerIndex(self.root)
        index.build(progress=lambda _: None)
        self.assertEqual(index.list_series('NY')['items'], [])
        warnings = index.overview()['warnings']
        self.assertTrue(any('Git LFS pointer' in str(w['message']) for w in warnings))
        self.assertEqual(sum(w['status'] == 'unavailable' for w in warnings), 5)

    def test_family_signals_group_variants_without_erasing_their_definitions(self):
        native = describe('cdc_nhsn_final', 'totalconfflunewadm', 'data.ndjson.gz', {})
        archive = describe('delphi_nhsn', 'value', 'signal=confirmed_admissions_flu_ew/data.csv',
                           {'spatial_support': 'HHS parent broadcast'})
        self.assertEqual(native['signal_key'], archive['signal_key'])
        self.assertNotEqual(native['variant_label'], archive['variant_label'])
        reported = describe('cdc_nssp_trajectories', 'percent_visits_influenza', '', {})
        smooth = describe('cdc_nssp_trajectories', 'percent_visits_smoothed_1', '', {})
        self.assertNotEqual(reported['signal_key'], smooth['signal_key'])
        self.assertIn('Smoothed', smooth['variant_label'])
        columns = ['pcr_target_avg_conc', 'pcr_target_flowpop_lin', 'flow_rate', 'population_served', 'lod_sewage']
        self.assertEqual(measure_columns('cdc_nwss_covid_raw', columns), tuple(columns[:2]))
        concentration = describe('cdc_nwss_covid_raw', 'pcr_target_avg_conc', '', {})
        wval = describe('cdc_nwss_wval', 'site_wval', '', {'pathogen_target': 'SARS-CoV-2'})
        self.assertNotEqual(concentration['signal_key'], wval['signal_key'])

    def test_delphi_signals_map_to_their_cdc_measure_columns(self):
        self.assertEqual(NHSN_DELPHI, {
            'confirmed_admissions_covid_ew': 'totalconfc19newadm',
            'confirmed_admissions_flu_ew': 'totalconfflunewadm',
            'confirmed_admissions_rsv_ew': 'totalconfrsvnewadm',
            'hosprep_confirmed_admissions_covid_ew': 'totalconfc19newadmhosprep',
            'hosprep_confirmed_admissions_flu_ew': 'totalconfflunewadmhosprep',
            'hosprep_confirmed_admissions_rsv_ew': 'totalconfrsvnewadmhosprep',
            'inpatient_beds_ew': 'numinptbeds',
            'inpatient_beds_occupied_pct_ew': 'numinptbedsocc',
        })
        self.assertEqual(
            describe('delphi_nhsn', 'value', 'signal=inpatient_beds_ew/data.csv', {})['signal_key'],
            'nhsn:numinptbeds',
        )
        self.assertEqual(
            describe('delphi_nwss', 'value', 'signal=covid_avg_conc/data.csv', {})['signal_key'],
            'nwss:covid_concentration',
        )

    def test_current_hubs_and_delphi_join_exact_cdc_columns(self):
        for hub, disease, pathogen in [('hub_flusight_current', 'flu', 'influenza'),
                                       ('hub_covid_current', 'c19', 'covid'),
                                       ('hub_rsv_current', 'rsv', 'rsv')]:
            target_disease = 'covid' if disease == 'c19' else disease
            hosp = describe(hub, 'observation', '', {'target': f'wk inc {target_disease} hosp'})
            cdc = describe('cdc_nhsn_final', f'totalconf{disease}newadm', '', {})
            self.assertEqual(hosp['signal_key'], cdc['signal_key'])
            ed = describe(hub, 'observation', '', {'target': f'wk inc {target_disease} prop ed visits'})
            direct = describe('cdc_nssp_trajectories', f'percent_visits_{pathogen}', '', {})
            archive = describe('delphi_nssp', 'value', '', {'signal': f'pct_ed_visits_{pathogen}'})
            self.assertEqual(ed['signal_key'], direct['signal_key'])
            self.assertEqual(archive['signal_key'], direct['signal_key'])
            self.assertIn('Proportion (0–1)', ed['variant_label'])
            self.assertEqual(archive['measure_id'], f'pct_ed_visits_{pathogen}')

    def test_saved_cdc_api_labels_used_for_mirrors_and_search(self):
        self.snapshot('cdc_nhsn_final', [{'jurisdiction': 'NY', 'weekendingdate': '2026-01-03',
                                        'totalconfflunewadm': 17, 'totalconfflunewadmadult': 15}])
        # Older downloads only have metadata.json, before the new columns.json sidecar.
        path = self.root / 'raw/cdc_nhsn_final/snapshots/s1/metadata.json'
        path.write_text(json.dumps({'columns': [{'fieldName': 'totalconfflunewadm',
                                               'name': 'CDC authoritative influenza admissions',
                                               'description': 'Publisher definition'}]}))
        self.snapshot('hub_flusight_current', members={
            'target-data/time-series.csv':
                'location,target_end_date,as_of,target,observation\n'
                '36,2026-01-03,2026-01-09,wk inc flu hosp,8\n'
                '36,2026-01-03,2026-01-09,wk inc flu prop ed visits,0.01\n'})
        reader = SelectedData(self.root)
        records = list(reader.iter_records(group='nhsn'))
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r.source_group == 'nhsn' for r in records))
        self.assertEqual(next(iter(records[1].signals.values()))['signal_title'], 'CDC authoritative influenza admissions')
        index = ExplorerIndex(self.root)
        index.build(progress=lambda _: None)
        items = index.list_series('NY', query='authoritative')['items']
        self.assertEqual(len(items), 2)
        self.assertEqual({i['signal_title'] for i in items}, {'CDC authoritative influenza admissions'})
        self.assertEqual({i['origin_column'] for i in items}, {'totalconfflunewadm'})
        self.assertEqual({i['column_description'] for i in items}, {'Publisher definition'})
        self.assertFalse(any('adult' in i['value_column'] for i in index.list_series('NY')['items']))

    def test_explorer_excludes_rates_and_county_hsa_copies(self):
        self.snapshot('cdc_nhsn_final', [{
            'jurisdiction': 'NY', 'weekendingdate': '2026-01-03',
            'totalconfflunewadm': 17, 'totalconfflunewadmper100k': 2, 'numinptbeds': 20}])
        self.snapshot('cdc_nssp_trajectories', [
            {'geography': 'New York', 'county': county, 'week_end': '2026-01-03',
             'percent_visits_influenza': value, 'hsa_nci_id': hsa}
            for county, value, hsa in [('All', 1, 'All'), ('Albany', 99, '1')]])
        index = ExplorerIndex(self.root)
        index.build(progress=lambda _: None)
        items = index.list_series('NY')['items']
        self.assertEqual(len(items), 3)
        nssp = next(i for i in items if i['source_group'] == 'nssp')
        self.assertEqual(index.data('NY', [nssp['id']])['series'][0]['points'][0][1], 1)


if __name__ == '__main__':
    unittest.main()
