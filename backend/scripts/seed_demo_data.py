"""Seed a fresh Barbarik Fitness Pal database with reproducible Phase 0 demo data.

Run after ``alembic upgrade head``. The script exits without changes when its demo
members already exist, so it is safe to run repeatedly.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.dates import local_date  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.models import (  # noqa: E402
    AuditLog,
    BodyweightLog,
    Exercise,
    FoodLog,
    Member,
    MemberNutritionTarget,
    MemberProfile,
    MemberTrendFlag,
    Subscription,
    WorkoutLog,
    WorkoutPlan,
)


DemoMember = dict[str, Any]
EXERCISE_CATALOG: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Barbell Back Squat", ("back squat", "squat", "barbell squat")),
    ("Front Squat", ("front barbell squat",)),
    ("Goblet Squat", ("dumbbell goblet squat",)),
    ("Leg Press", ("45 degree leg press",)),
    ("Hack Squat", ("machine hack squat",)),
    ("Bulgarian Split Squat", ("rear foot elevated split squat", "RFESS")),
    ("Walking Lunge", ("walking lunges",)),
    ("Reverse Lunge", ("backward lunge",)),
    ("Romanian Deadlift", ("RDL", "romanian DL")),
    ("Conventional Deadlift", ("deadlift", "barbell deadlift")),
    ("Sumo Deadlift", ("sumo pull",)),
    ("Trap Bar Deadlift", ("hex bar deadlift",)),
    ("Hip Thrust", ("barbell hip thrust", "glute bridge")),
    ("Cable Pull Through", ("pull through",)),
    ("Leg Curl", ("hamstring curl", "lying leg curl")),
    ("Seated Leg Curl", ("seated hamstring curl",)),
    ("Leg Extension", ("quad extension",)),
    ("Standing Calf Raise", ("calf raise",)),
    ("Seated Calf Raise", ("seated calves",)),
    ("Dumbbell Bench Press", ("DB bench", "dumbbell press")),
    ("Barbell Bench Press", ("bench press", "flat bench")),
    ("Incline Dumbbell Press", ("incline DB press",)),
    ("Incline Barbell Press", ("incline bench",)),
    ("Decline Bench Press", ("decline bench",)),
    ("Push Up", ("pushup", "press up")),
    ("Chest Dip", ("dips", "parallel bar dip")),
    ("Cable Fly", ("cable crossover",)),
    ("Pec Deck", ("chest fly machine",)),
    ("Machine Chest Press", ("chest press",)),
    ("Overhead Press", ("OHP", "military press")),
    ("Dumbbell Shoulder Press", ("DB shoulder press",)),
    ("Arnold Press", ("arnold dumbbell press",)),
    ("Lateral Raise", ("side raise",)),
    ("Front Raise", ("anterior raise",)),
    ("Rear Delt Fly", ("reverse fly", "rear delt raise")),
    ("Face Pull", ("cable face pull",)),
    ("Upright Row", ("barbell upright row",)),
    ("Pull Up", ("pullup", "bodyweight pull up")),
    ("Chin Up", ("chinup", "underhand pull up")),
    ("Lat Pulldown", ("pulldown", "wide grip pulldown")),
    ("Barbell Row", ("bent over row", "BB row")),
    ("Dumbbell Row", ("one arm row", "single arm dumbbell row")),
    ("Seated Cable Row", ("cable row",)),
    ("Chest Supported Row", ("incline bench row",)),
    ("T-Bar Row", ("tbar row",)),
    ("Straight Arm Pulldown", ("lat prayer",)),
    ("Shrug", ("barbell shrug", "dumbbell shrug")),
    ("Barbell Curl", ("bicep curl", "BB curl")),
    ("Dumbbell Curl", ("DB curl",)),
    ("Hammer Curl", ("neutral grip curl",)),
    ("Preacher Curl", ("preacher bench curl",)),
    ("Cable Curl", ("cable bicep curl",)),
    ("Triceps Pushdown", ("rope pushdown", "cable pushdown")),
    ("Skull Crusher", ("lying triceps extension",)),
    ("Overhead Triceps Extension", ("overhead cable extension",)),
    ("Close Grip Bench Press", ("CGBP",)),
    ("Diamond Push Up", ("diamond pushup",)),
    ("Plank", ("forearm plank",)),
    ("Side Plank", ("lateral plank",)),
    ("Dead Bug", ("deadbug",)),
    ("Bird Dog", ("birddog",)),
    ("Hanging Knee Raise", ("knee raise",)),
    ("Hanging Leg Raise", ("leg raise",)),
    ("Cable Crunch", ("kneeling cable crunch",)),
    ("Ab Wheel Rollout", ("ab rollout",)),
    ("Russian Twist", ("seated russian twist",)),
    ("Farmer Carry", ("farmer walk",)),
    ("Suitcase Carry", ("one arm carry",)),
    ("Kettlebell Swing", ("KB swing",)),
    ("Kettlebell Clean", ("KB clean",)),
    ("Kettlebell Press", ("KB press",)),
    ("Battle Rope Waves", ("battle ropes",)),
    ("Sled Push", ("prowler push",)),
    ("Box Jump", ("jump box",)),
    ("Burpee", ("burpees",)),
    ("Mountain Climber", ("mountain climbers",)),
    ("Jump Rope", ("skipping", "rope skipping")),
    ("Treadmill Run", ("running", "treadmill jogging")),
    ("Stationary Bike", ("exercise bike", "cycling")),
    ("Rowing Machine", ("erg row",)),
    ("Surya Namaskar", ("sun salutation",)),
    ("Dand", ("Hindu push up", "hindu pushup")),
    ("Baithak", ("Hindu squat", "bethak")),
    ("Hanuman Dand", ("Hanuman push up",)),
    ("Mallakhamb Pull", ("mallakhamb",)),
    ("Naukasana", ("boat pose",)),
    ("Bhujangasana", ("cobra pose",)),
    ("Utkatasana", ("chair pose",)),
    ("Virabhadrasana", ("warrior pose",)),
    ("Pawanmuktasana", ("wind relieving pose",)),
    ("Kapalbhati", ("kapalabhati breathing",)),
    ("Bhastrika", ("bellows breath",)),
    ("Pranayama", ("breathing practice",)),
    ("Resistance Band Row", ("band row",)),
    ("Resistance Band Press", ("band chest press",)),
    ("Resistance Band Squat", ("band squat",)),
    ("TRX Row", ("suspension row",)),
    ("TRX Push Up", ("suspension push up",)),
    ("Step Up", ("box step up",)),
    ("Glute Bridge", ("bodyweight glute bridge",)),
    ("Good Morning", ("barbell good morning",)),
    ("Nordic Curl", ("nordic hamstring curl",)),
    ("Zercher Squat", ("zercher",)),
)

DEMO_MEMBERS: tuple[DemoMember, ...] = (
    {
        "email": "rajesh.demo@barbarik.local", "phone": "+919810000001", "full_name": "Rajesh Kumar",
        "dob": date(1992, 5, 14), "sex": "male", "height": Decimal("174.00"), "goal": "fat_loss",
        "activity": "moderate", "experience": "intermediate", "days": 4, "split": "upper_lower",
        "diet": "Vegetarian; enjoys dal, paneer, and home-cooked North Indian meals.", "tier": "quarterly",
        "timezone": "Asia/Kolkata",
        "status": "active", "bmr": Decimal("1748.00"), "tdee": Decimal("2710.00"), "calories": 2200,
        "protein": 150, "fat": 65, "carbs": 255, "start_weight": Decimal("92.40"), "daily_delta": Decimal("-0.055"),
        "workouts_per_week": 3,
    },
    {
        "email": "priya.demo@barbarik.local", "phone": "+919810000002", "full_name": "Priya Shah",
        "dob": date(1996, 9, 2), "sex": "female", "height": Decimal("162.00"), "goal": "muscle_gain",
        "activity": "active", "experience": "intermediate", "days": 5, "split": "push_pull_legs",
        "diet": "Eggetarian; prefers Indian meals and easy high-protein snacks.", "tier": "annual",
        "timezone": "Asia/Kolkata",
        "status": "active", "bmr": Decimal("1370.00"), "tdee": Decimal("2220.00"), "calories": 2400,
        "protein": 125, "fat": 70, "carbs": 310, "start_weight": Decimal("58.20"), "daily_delta": Decimal("0.018"),
        "workouts_per_week": 4,
    },
    {
        "email": "amit.demo@barbarik.local", "phone": "+919810000003", "full_name": "Amit Verma",
        "dob": date(1988, 1, 21), "sex": "male", "height": Decimal("179.00"), "goal": "strength",
        "activity": "light", "experience": "advanced", "days": 3, "split": "full_body",
        "diet": "Non-vegetarian; prefers simple Indian food and tracks protein inconsistently.", "tier": "monthly",
        "timezone": "Asia/Kolkata",
        "status": "past_due", "bmr": Decimal("1810.00"), "tdee": Decimal("2490.00"), "calories": 2550,
        "protein": 165, "fat": 75, "carbs": 305, "start_weight": Decimal("81.00"), "daily_delta": Decimal("0.006"),
        "workouts_per_week": 1,
    },
)


def classify_exercise(name: str) -> tuple[str, str, str, bool]:
    """Give the reference library useful, deterministic starter metadata."""

    lowered = name.lower()
    if any(word in lowered for word in ("squat", "lunge", "leg press", "step up", "baithak")):
        return "quadriceps", "barbell or bodyweight", "squat", True
    if any(word in lowered for word in ("deadlift", "thrust", "curl", "good morning", "swing")):
        return "hamstrings", "barbell or machine", "hinge", True
    if any(word in lowered for word in ("press", "push", "dip", "dand")):
        return "chest", "barbell, dumbbell, cable, or bodyweight", "push", True
    if any(word in lowered for word in ("row", "pull", "chin", "shrug")):
        return "back", "cable, machine, or free weights", "pull", True
    if any(word in lowered for word in ("plank", "crunch", "raise", "twist", "naukasana")):
        return "core", "bodyweight or cable", "core", False
    return "conditioning", "bodyweight", "carry", False


def make_item(
    name: str,
    calories: int,
    protein_g: int,
    carbs_g: int,
    fat_g: int,
) -> dict[str, Any]:
    """Return the JSON shape used by the food-log schema."""

    return {
        "name": name,
        "qty": 1,
        "unit": "serving",
        "calories": calories,
        "protein_g": protein_g,
        "carbs_g": carbs_g,
        "fat_g": fat_g,
    }


def meal_payload(member: DemoMember, day_offset: int, meal_type: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Create varied Indian-friendly meals while retaining reproducible totals."""

    base: dict[str, tuple[str, int, int, int, int]] = {
        "breakfast": ("Paneer bhurji with roti", 520, 29, 48, 22),
        "lunch": ("Dal, rice, sabzi and curd", 710, 28, 108, 17),
        "dinner": ("Chicken curry with roti", 640, 42, 62, 19),
    }
    if member["goal"] == "muscle_gain":
        base["breakfast"] = ("Egg bhurji, oats and banana", 610, 34, 72, 21)
        base["lunch"] = ("Rajma rice with paneer", 790, 36, 112, 23)
        base["dinner"] = ("Paneer tikka with jeera rice", 740, 40, 82, 26)
    if member["goal"] == "strength":
        base["breakfast"] = ("Poha with eggs", 490, 20, 61, 17)
        base["lunch"] = ("Chicken biryani", 760, 33, 95, 25)
        base["dinner"] = ("Dal roti and salad", 520, 19, 70, 14)
    name, calories, protein, carbs, fat = base[meal_type]
    variation = (day_offset % 3) * 10
    if member["goal"] == "strength":
        protein -= 4
    totals = {"calories": calories + variation, "protein_g": protein, "carbs_g": carbs + variation // 2, "fat_g": fat}
    return [make_item(name, **totals)], totals


def add_members(session: Session, now: datetime) -> dict[str, Member]:
    """Insert members and their one-to-one Phase 0 records."""

    members: dict[str, Member] = {}
    for profile in DEMO_MEMBERS:
        member = Member(
            email=profile["email"], phone=profile["phone"], password_hash="$demo-not-for-production$phase-1-auth-pending",
            full_name=profile["full_name"], date_of_birth=profile["dob"], sex=profile["sex"], height_cm=profile["height"],
        )
        session.add(member)
        session.flush()
        members[profile["email"]] = member
        session.add_all([
            Subscription(member_id=member.id, tier=profile["tier"], status=profile["status"], started_at=now - timedelta(days=60), expires_at=now + timedelta(days=30), payment_provider="manual", payment_ref=f"DEMO-{member.id.hex[:8]}"),
            MemberProfile(member_id=member.id, goal=profile["goal"], activity_level=profile["activity"], training_experience=profile["experience"], injuries_limitations=None, preferred_days_per_week=profile["days"], preferred_split=profile["split"], dietary_preferences=profile["diet"], timezone=profile["timezone"], whatsapp_linked=True, whatsapp_phone=profile["phone"], whatsapp_verified_at=now, onboarding_completed_at=now - timedelta(days=60)),
            MemberNutritionTarget(member_id=member.id, bmr_kcal=profile["bmr"], tdee_kcal=profile["tdee"], calorie_target_kcal=profile["calories"], recommended_protein_target_g=None, active_protein_target_g=None, fat_target_g=profile["fat"], carb_target_g=profile["carbs"], calculated_at=now),
            AuditLog(member_id=member.id, action="seed_member", entity_type="member", entity_id=member.id, after={"email": member.email, "goal": profile["goal"]}, source="system", actor="system"),
        ])
    return members


def add_exercises(session: Session) -> dict[str, Exercise]:
    """Insert the 100-exercise mixed Indian and international library."""

    exercises: dict[str, Exercise] = {}
    for name, aliases in EXERCISE_CATALOG:
        muscle, equipment, pattern, compound = classify_exercise(name)
        exercise = Exercise(name=name, aliases=list(aliases), primary_muscle=muscle, secondary_muscles=[], equipment=equipment, movement_pattern=pattern, is_compound=compound)
        session.add(exercise)
        exercises[name] = exercise
    session.flush()
    return exercises


def exercise_target(exercise: Exercise, sets: int = 3, reps: str = "8-12") -> dict[str, Any]:
    """Return a compact plan exercise object."""

    return {"exercise_id": str(exercise.id), "name": exercise.name, "target_sets": sets, "target_reps": reps}


def add_workout_plans(session: Session, members: dict[str, Member], exercises: dict[str, Exercise]) -> dict[str, WorkoutPlan]:
    """Add two plans for each supported split type and expose active plans by member."""

    r = members["rajesh.demo@barbarik.local"]
    p = members["priya.demo@barbarik.local"]
    a = members["amit.demo@barbarik.local"]
    templates: tuple[tuple[Member, str, str, int, bool, dict[str, list[str]]], ...] = (
        (a, "Full Body Strength A", "full_body", 3, True, {"Monday": ["Barbell Back Squat", "Barbell Bench Press", "Barbell Row"], "Wednesday": ["Conventional Deadlift", "Overhead Press", "Lat Pulldown"], "Friday": ["Front Squat", "Incline Dumbbell Press", "Pull Up"]}),
        (a, "Full Body Strength B", "full_body", 3, False, {"Monday": ["Goblet Squat", "Dumbbell Bench Press", "Seated Cable Row"], "Wednesday": ["Romanian Deadlift", "Dumbbell Shoulder Press", "Chin Up"], "Friday": ["Leg Press", "Push Up", "T-Bar Row"]}),
        (r, "Upper Lower Fat Loss A", "upper_lower", 4, True, {"Upper A": ["Barbell Bench Press", "Lat Pulldown", "Lateral Raise"], "Lower A": ["Barbell Back Squat", "Romanian Deadlift", "Standing Calf Raise"], "Upper B": ["Incline Dumbbell Press", "Seated Cable Row", "Triceps Pushdown"], "Lower B": ["Leg Press", "Walking Lunge", "Leg Curl"]}),
        (r, "Upper Lower Fat Loss B", "upper_lower", 4, False, {"Upper A": ["Dumbbell Bench Press", "Dumbbell Row", "Face Pull"], "Lower A": ["Front Squat", "Hip Thrust", "Seated Calf Raise"], "Upper B": ["Machine Chest Press", "Chest Supported Row", "Hammer Curl"], "Lower B": ["Hack Squat", "Cable Pull Through", "Leg Extension"]}),
        (p, "Push Pull Legs Hypertrophy A", "push_pull_legs", 5, True, {"Push": ["Incline Dumbbell Press", "Dumbbell Shoulder Press", "Triceps Pushdown"], "Pull": ["Pull Up", "Barbell Row", "Dumbbell Curl"], "Legs": ["Barbell Back Squat", "Hip Thrust", "Leg Curl"]}),
        (p, "Push Pull Legs Hypertrophy B", "push_pull_legs", 5, False, {"Push": ["Machine Chest Press", "Arnold Press", "Skull Crusher"], "Pull": ["Lat Pulldown", "Seated Cable Row", "Preacher Curl"], "Legs": ["Leg Press", "Bulgarian Split Squat", "Standing Calf Raise"]}),
        (r, "Custom Home Strength A", "custom", 3, False, {"Day 1": ["Dand", "Baithak", "Plank"], "Day 2": ["Kettlebell Swing", "Resistance Band Row", "Dead Bug"], "Day 3": ["Surya Namaskar", "Push Up", "Farmer Carry"]}),
        (p, "Custom Yoga Conditioning B", "custom", 3, False, {"Day 1": ["Surya Namaskar", "Utkatasana", "Naukasana"], "Day 2": ["Virabhadrasana", "Bhujangasana", "Side Plank"], "Day 3": ["Jump Rope", "Mountain Climber", "Russian Twist"]}),
    )
    active_by_member: dict[str, WorkoutPlan] = {}
    for member, name, split, days, active, schedule in templates:
        serialized = {day: [exercise_target(exercises[item]) for item in items] for day, items in schedule.items()}
        plan = WorkoutPlan(member_id=member.id, name=name, split_type=split, days_per_week=days, schedule=serialized, is_active=active)
        session.add(plan)
        if active:
            active_by_member[str(member.id)] = plan
    session.flush()
    return active_by_member


def add_history(session: Session, members: dict[str, Member], plans: dict[str, WorkoutPlan], exercises: dict[str, Exercise], now: datetime) -> None:
    """Create 28 days of bodyweight, food, and workout history for every demo user."""

    workout_names = ("Barbell Back Squat", "Barbell Bench Press", "Barbell Row")
    for profile in DEMO_MEMBERS:
        today = local_date(now, profile["timezone"])
        member_timezone = ZoneInfo(profile["timezone"])
        member = members[profile["email"]]
        plan = plans[str(member.id)]
        for day_offset in range(28):
            log_date = today - timedelta(days=27 - day_offset)
            weight = profile["start_weight"] + profile["daily_delta"] * Decimal(day_offset)
            recorded_at = datetime.combine(
                log_date,
                time(hour=12),
                tzinfo=member_timezone,
            ).astimezone(timezone.utc)
            session.add(
                BodyweightLog(
                    member_id=member.id,
                    log_date=log_date,
                    weight_kg=weight.quantize(Decimal("0.01")),
                    recorded_at=recorded_at,
                    source="app",
                )
            )
            for meal_type in ("breakfast", "lunch", "dinner"):
                items, totals = meal_payload(profile, day_offset, meal_type)
                meal_hour = {"breakfast": 8, "lunch": 13, "dinner": 20}[meal_type]
                meal_recorded_at = datetime.combine(
                    log_date,
                    time(hour=meal_hour),
                    tzinfo=member_timezone,
                ).astimezone(timezone.utc)
                session.add(
                    FoodLog(
                        member_id=member.id,
                        log_date=log_date,
                        meal_type=meal_type,
                        items=items,
                        totals=totals,
                        recorded_at=meal_recorded_at,
                        source="whatsapp",
                    )
                )
            weekday = log_date.weekday()
            frequency = profile["workouts_per_week"]
            should_train = weekday in ({1: {2}, 3: {0, 2, 4}, 4: {0, 1, 3, 5}}[frequency])
            if should_train:
                workout_recorded_at = datetime.combine(
                    log_date,
                    time(hour=18),
                    tzinfo=member_timezone,
                ).astimezone(timezone.utc)
                performance: list[dict[str, Any]] = []
                for index, name in enumerate(workout_names):
                    base_weight = 45 + index * 15 + day_offset // 7 * 2
                    performance.append({"exercise_id": str(exercises[name].id), "name": name, "sets": [{"weight_kg": base_weight, "reps": 8 + index, "rpe": 7.5, "note": None}], "superset_group": None})
                session.add(WorkoutLog(member_id=member.id, log_date=log_date, plan_id=plan.id, exercises=performance, recorded_at=workout_recorded_at, source="whatsapp"))


def add_trend_flags(session: Session, members: dict[str, Member], now: datetime) -> None:
    """Add meaningful precomputed flags for dashboard/demo states."""

    flags = (
        ("rajesh.demo@barbarik.local", "rapid_weight_loss", "warning", {"weekly_change_pct": -1.2, "window_days": 7}),
        ("rajesh.demo@barbarik.local", "protein_consistently_low", "info", {"average_protein_g": 118, "target_g": 150, "window_days": 14}),
        ("priya.demo@barbarik.local", "rapid_weight_gain", "warning", {"weekly_change_pct": 0.7, "window_days": 7}),
        ("amit.demo@barbarik.local", "low_training_frequency", "warning", {"sessions_per_week": 1.0, "target": 3}),
        ("amit.demo@barbarik.local", "plan_adherence_problem", "warning", {"adherence_pct": 33, "target": 70, "window_days": 28}),
        ("amit.demo@barbarik.local", "strength_decline", "info", {"estimated_1rm_slope_kg_per_week": -0.8, "sessions": 4}),
    )
    for email, flag_type, severity, details in flags:
        session.add(MemberTrendFlag(member_id=members[email].id, flag_type=flag_type, severity=severity, details=details, detected_at=now, is_active=True))


def seed_demo_data() -> None:
    """Seed all requested Phase 0 demo entities in one transaction."""

    now = datetime.now(timezone.utc)
    with SessionLocal() as session:
        existing = session.scalar(select(Member.id).where(Member.email.in_([entry["email"] for entry in DEMO_MEMBERS])).limit(1))
        if existing is not None:
            print("Demo data already exists; no changes made.")
            return
        members = add_members(session, now)
        exercises = add_exercises(session)
        plans = add_workout_plans(session, members, exercises)
        add_history(session, members, plans, exercises, now)
        add_trend_flags(session, members, now)
        session.commit()
    print(f"Seeded {len(DEMO_MEMBERS)} demo members, {len(EXERCISE_CATALOG)} exercises, 8 plans, and 28 days of history.")


if __name__ == "__main__":
    seed_demo_data()
