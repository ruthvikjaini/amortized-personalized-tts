from persontts.controllers import Controller
from persontts.discovery import discover, exhaustive_oracle
from persontts.experience_bank import ExperienceBank, HistoryEntry
from persontts.guide import distill_guide
from persontts.problems import generate_problems, split_problems
from persontts.profiles import UserProfile
from persontts.trajectories import build_bank


def _setup(seed=2):
    problems = generate_problems(n_problems=36, seed=seed)
    discovery, heldout = split_problems(problems, seed=seed)
    bank = build_bank(problems, n_trajectories=16, seed=seed)
    return discovery, heldout, bank


def test_retrieval_returns_nearest_profile():
    bank = ExperienceBank()
    near = UserProfile("near", 0.80, 250.0, 0.0015)
    far = UserProfile("far", 0.70, 600.0, 0.0060)
    bank.add(HistoryEntry(far, Controller("small", 1, 1, False), 0.5))
    bank.add(HistoryEntry(near, Controller("large", 4, 2, False), 0.9))
    target = UserProfile("target", 0.81, 240.0, 0.0014)
    assert bank.retrieve(target).profile.name == "near"
    assert ExperienceBank().retrieve(target) is None


def test_guide_recommendation_respects_tight_profile():
    bank = ExperienceBank()
    for acc, lat, cost, controller in [
        (0.90, 200.0, 0.0010, Controller("large", 2, 2, True)),
        (0.90, 200.0, 0.0010, Controller("large", 2, 2, True)),
        (0.70, 500.0, 0.0050, Controller("small", 8, 1, False)),
    ]:
        bank.add(
            HistoryEntry(
                UserProfile("s", acc, lat, cost), controller, 0.8
            )
        )
    guide = distill_guide(bank)
    assert guide.rules  # Rules carry the counts behind them.
    recommended = guide.recommend(UserProfile("t", 0.90, 200.0, 0.0010))
    assert recommended.tier == "large"
    assert recommended.early_stop is True


def test_discovery_variants_respect_budget():
    discovery, _, bank = _setup()
    profile = UserProfile("target", 0.80, 250.0, 0.0015)
    for variant in ("from_scratch", "warm_start", "guide_only", "full"):
        result = discover(
            profile, discovery, bank, variant=variant, budget=12, n_seeds=2, seed=1
        )
        assert result.n_candidates <= 12
        assert 0.0 <= result.joint_satisfaction_rate <= 1.0


def test_candidates_to_best_marks_first_discovery_of_best():
    discovery, _, bank = _setup()
    profile = UserProfile("target", 0.80, 250.0, 0.0015)
    result = discover(profile, discovery, bank, variant="from_scratch", budget=12, n_seeds=2, seed=1)
    assert 1 <= result.candidates_to_best <= result.n_candidates
    assert result.history[result.candidates_to_best - 1]["controller"] == result.best_controller.label()


def test_warm_start_begins_from_retrieved_controller():
    discovery, _, bank = _setup()
    experience = ExperienceBank()
    source = UserProfile("source", 0.80, 250.0, 0.0015)
    experience.add(
        HistoryEntry(source, Controller("small", 1, 1, True), 0.5)
    )
    profile = UserProfile("target", 0.80, 250.0, 0.0015)
    result = discover(
        profile, discovery, bank, variant="warm_start",
        bank_experience=experience, budget=1, n_seeds=2, seed=1,
    )
    assert result.history[0]["controller"] == "small/w1/p1+stop"


def test_discovery_never_worse_than_its_start_on_discovery_set():
    from persontts.controllers import evaluate_controller, summarise
    from persontts.discovery import DEFAULT_CONTROLLER

    discovery, _, bank = _setup()
    profile = UserProfile("target", 0.80, 300.0, 0.0020)
    result = discover(profile, discovery, bank, variant="from_scratch", budget=20, n_seeds=2, seed=1)
    # Hill-climbing only moves to a strictly better score, so the
    # returned satisfaction is at least the default controller's.
    baseline = summarise(
        evaluate_controller(DEFAULT_CONTROLLER, discovery, bank, n_seeds=2, seed=1), profile
    )
    assert result.joint_satisfaction_rate >= baseline["joint_satisfaction_rate"]
