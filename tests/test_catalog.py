from __future__ import annotations

import unittest

from tapestry.data.catalog import CATALOG, specs_in_group
from tapestry.data.models import RevisionMode


class CatalogTests(unittest.TestCase):
    def test_catalog_has_expected_source_families(self):
        self.assertEqual(len(CATALOG), 25)
        self.assertIn("cdc_nhsn_initial_release", CATALOG)
        self.assertIn("delphi_nwss", CATALOG)
        self.assertEqual(
            {spec.config["source"] for spec in specs_in_group("delphi")},
            {"nhsn", "nssp", "nwss", "claims_inpatient", "claims_outpatient"},
        )
        self.assertTrue(all(spec.fetcher == "delphi_v5" for spec in specs_in_group("delphi")))
        self.assertFalse({"delphi_fluview", "delphi_fluview_clinical", "delphi_flusurv"} & CATALOG.keys())
        self.assertIn("hub_flusight_current", CATALOG)
        self.assertIn("hub_covid_legacy", CATALOG)
        self.assertGreaterEqual(len(specs_in_group("core")), 10)

    def test_revision_semantics_drive_versioned_flag(self):
        self.assertFalse(CATALOG["cdc_nhsn_final"].versioned)
        self.assertTrue(CATALOG["cdc_nhsn_initial_release"].versioned)
        self.assertTrue(CATALOG["delphi_nhsn"].versioned)
        self.assertEqual(
            CATALOG["hub_flusight_current"].revision_mode,
            RevisionMode.AS_OF_COLUMN,
        )

    def test_every_spec_has_class_ready_metadata(self):
        for key, spec in CATALOG.items():
            with self.subTest(key=key):
                self.assertEqual(key, spec.key)
                self.assertTrue(spec.natural_key)
                self.assertIsNotNone(spec.event_date_column)
                self.assertTrue(spec.geographic_resolutions)
                self.assertTrue(spec.measures)


if __name__ == "__main__":
    unittest.main()
