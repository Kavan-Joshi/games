import pytest
from environment.models import ResetRequest, InspectorAction, TaskType


class TestEnvReset:
    def test_reset_returns_observation(self):
        from environment import FleetAIEnv
        env = FleetAIEnv()
        result = env.reset(ResetRequest(task="inspection_easy", seed=42))
        assert result.observation is not None
        assert result.observation.ticket.id != ""
        assert result.done is False
        assert result.reward == 0.0

    def test_reset_with_each_task(self):
        from environment import FleetAIEnv
        env = FleetAIEnv()
        for task in ["inspection_easy", "inspection_hard", "inspection_adversarial"]:
            result = env.reset(ResetRequest(task=task, seed=1))
            assert result.observation.task_type == task

    def test_reset_invalid_task_defaults_to_easy(self):
        from environment import FleetAIEnv
        env = FleetAIEnv()
        result = env.reset(ResetRequest(task="nonexistent_task", seed=1))
        assert result.observation.task_type == "inspection_easy"

    def test_reset_seed_reproducibility(self):
        from environment import FleetAIEnv
        env1 = FleetAIEnv()
        env2 = FleetAIEnv()
        r1 = env1.reset(ResetRequest(task="inspection_easy", seed=99))
        r2 = env2.reset(ResetRequest(task="inspection_easy", seed=99))
        assert r1.observation.ticket.id == r2.observation.ticket.id

    def test_reset_info_has_ground_truth(self):
        from environment import FleetAIEnv
        env = FleetAIEnv()
        result = env.reset(ResetRequest(task="inspection_easy", seed=5))
        assert "ground_truth" in result.info
        assert "category" in result.info["ground_truth"]
        assert "priority" in result.info["ground_truth"]
        assert "department" in result.info["ground_truth"]

    def test_reset_max_steps_is_3(self):
        from environment import FleetAIEnv
        env = FleetAIEnv()
        result = env.reset(ResetRequest(task="inspection_easy", seed=1))
        assert result.observation.max_steps == 3


class TestEnvStep:
    def test_step_returns_reward(self):
        from environment import FleetAIEnv
        env = FleetAIEnv()
        env.reset(ResetRequest(task="inspection_easy", seed=42))
        action = InspectorAction(flagged=False, confidence=0.5)
        result = env.step(action)
        assert isinstance(result.reward, float)
        assert 0.0 <= result.reward <= 1.0

    def test_step_with_correct_flag(self):
        from environment import FleetAIEnv
        env = FleetAIEnv()
        r = env.reset(ResetRequest(task="inspection_easy", seed=1))
        has_errors = r.info.get("has_injected_errors", False)
        if has_errors:
            action = InspectorAction(flagged=True, confidence=0.7)
            result = env.step(action)
            assert result.reward > 0.0

    def test_step_without_reset_returns_error(self):
        from environment import FleetAIEnv
        env = FleetAIEnv()
        action = InspectorAction(flagged=False, confidence=0.5)
        result = env.step(action)
        assert "error" in result.info

    def test_multi_step_allows_retry(self):
        from environment import FleetAIEnv
        env = FleetAIEnv()
        for seed in range(50):
            r = env.reset(ResetRequest(task="inspection_hard", seed=seed))
            if r.info.get("has_injected_errors"):
                bad = InspectorAction(flagged=False, confidence=0.9)
                result1 = env.step(bad)
                if not result1.done:
                    assert result1.info.get("can_retry") is True
                    assert "hints" in result1.info
                    return
        pytest.skip("No error ticket found in 50 seeds")

    def test_best_reward_preserved_across_retries(self):
        from environment import FleetAIEnv
        env = FleetAIEnv()
        for seed in range(50):
            r = env.reset(ResetRequest(task="inspection_hard", seed=seed))
            if r.info.get("has_injected_errors"):
                bad = InspectorAction(flagged=False, confidence=0.9)
                result1 = env.step(bad)
                if not result1.done:
                    better = InspectorAction(flagged=True, flagged_fields=r.info.get("injected_error_fields", []), confidence=0.7)
                    result2 = env.step(better)
                    assert result2.info.get("best_reward", 0) >= result1.reward
                    return
        pytest.skip("No error ticket found in 50 seeds")

    def test_reward_floor_prevents_zero(self):
        from environment import FleetAIEnv
        env = FleetAIEnv()
        for seed in range(50):
            r = env.reset(ResetRequest(task="inspection_adversarial", seed=seed))
            if r.info.get("has_injected_errors"):
                terrible = InspectorAction(flagged=False, confidence=0.99)
                result = env.step(terrible)
                assert result.reward >= 0.05
                return
        pytest.skip("No error ticket found in 50 seeds")


class TestEnvTimeout:
    def test_timeout_returns_penalty(self):
        import time
        from environment import FleetAIEnv
        env = FleetAIEnv()
        env.reset(ResetRequest(task="inspection_easy", seed=42))
        env._episode_start_time = time.time() - 31.0
        action = InspectorAction(flagged=False, confidence=0.5)
        result = env.step(action)
        assert result.info.get("penalty") == "timeout"

    def test_normal_step_no_timeout(self):
        from environment import FleetAIEnv
        env = FleetAIEnv()
        env.reset(ResetRequest(task="inspection_easy", seed=42))
        action = InspectorAction(flagged=False, confidence=0.5)
        result = env.step(action)
        assert result.info.get("penalty") != "timeout"


class TestEnvState:
    def test_state_returns_observation(self):
        from environment import FleetAIEnv
        env = FleetAIEnv()
        env.reset(ResetRequest(task="inspection_easy", seed=1))
        state = env.state()
        assert state.ticket.id != ""

    def test_state_without_reset(self):
        from environment import FleetAIEnv
        env = FleetAIEnv()
        state = env.state()
        assert state.task_id == ""
