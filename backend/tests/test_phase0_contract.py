"""Static contract checks for the Phase 0 migration and seed fixture."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

from scripts.seed_demo_data import DEMO_MEMBERS, meal_payload


BACKEND_ROOT = Path(__file__).resolve().parents[1]
MIGRATION = BACKEND_ROOT / "alembic" / "versions" / "20260826_0001_initial_schema.py"
SEED_SCRIPT = BACKEND_ROOT / "scripts" / "seed_demo_data.py"


class PhaseZeroContractTests(unittest.TestCase):
    """Protect the tables and fixture size promised by the Phase 0 scaffold."""

    def test_initial_migration_declares_all_phase_zero_tables(self) -> None:
        source = MIGRATION.read_text(encoding="utf-8")
        expected_tables = {
            "members", "subscriptions", "member_profiles", "member_nutrition_targets",
            "bodyweight_logs", "food_logs", "exercises", "workout_plans", "workout_logs",
            "member_trend_flags", "audit_logs",
        }
        for table in expected_tables:
            self.assertIn(f'"{table}"', source)
        self.assertIn("CREATE EXTENSION IF NOT EXISTS pgcrypto", source)
        self.assertIn("CREATE EXTENSION IF NOT EXISTS vector", source)

    def test_seed_catalog_has_about_one_hundred_exercises(self) -> None:
        tree = ast.parse(SEED_SCRIPT.read_text(encoding="utf-8"))
        catalog = next(
            node.value
            for node in tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "EXERCISE_CATALOG"
        )
        self.assertIsInstance(catalog, ast.Tuple)
        assert isinstance(catalog, ast.Tuple)
        self.assertGreaterEqual(len(catalog.elts), 100)

    def test_seed_meal_payload_matches_food_json_contract(self) -> None:
        expected_totals = {"calories", "protein_g", "carbs_g", "fat_g"}

        for member in DEMO_MEMBERS:
            for meal_type in ("breakfast", "lunch", "dinner"):
                with self.subTest(member=member["email"], meal_type=meal_type):
                    items, totals = meal_payload(member, 0, meal_type)

                    self.assertEqual(set(totals), expected_totals)
                    self.assertEqual(len(items), 1)
                    self.assertTrue({"name", "qty", "unit"} <= set(items[0]))
                    for nutrient in expected_totals:
                        self.assertEqual(items[0][nutrient], totals[nutrient])


if __name__ == "__main__":
    unittest.main()
