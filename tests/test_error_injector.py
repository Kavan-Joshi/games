import pytest
from environment.models import ErrorType
from environment.tickets import TICKETS


class TestErrorInjector:
    def test_generates_worker_response(self):
        from environment.error_injector import ErrorInjector
        import random
        injector = ErrorInjector(rng=random.Random(42))
        ticket = TICKETS[0]
        error_mix = {"clean": 0.30, "obvious": 0.40, "subtle": 0.20, "multi": 0.10}
        worker, errors = injector.generate_worker_response(ticket, error_mix=error_mix)
        assert worker is not None
        assert worker.classification is not None or worker.priority is not None

    def test_clean_response_has_no_errors(self):
        from environment.error_injector import ErrorInjector
        import random
        injector = ErrorInjector(rng=random.Random(42))
        ticket = TICKETS[0]
        worker, errors = injector.generate_worker_response(ticket, error_mix={"clean": 1.0})
        assert len(errors) == 0

    def test_obvious_errors_have_low_subtlety(self):
        from environment.error_injector import ErrorInjector
        import random
        injector = ErrorInjector(rng=random.Random(42))
        ticket = TICKETS[0]
        worker, errors = injector.generate_worker_response(ticket, error_mix={"obvious": 1.0})
        if errors:
            for e in errors:
                assert e.subtlety <= 2

    def test_subtle_errors_have_high_subtlety(self):
        from environment.error_injector import ErrorInjector
        import random
        injector = ErrorInjector(rng=random.Random(42))
        ticket = TICKETS[0]
        worker, errors = injector.generate_worker_response(ticket, error_mix={"subtle": 1.0})
        if errors:
            for e in errors:
                assert e.subtlety >= 2

    def test_multi_error_produces_multiple_errors(self):
        from environment.error_injector import ErrorInjector
        import random
        multi_error_count = 0
        for seed in range(20):
            injector = ErrorInjector(rng=random.Random(seed))
            ticket = TICKETS[seed % len(TICKETS)]
            worker, errors = injector.generate_worker_response(ticket, error_mix={"multi": 1.0})
            if len(errors) > 1:
                multi_error_count += 1
        assert multi_error_count > 0

    def test_seed_reproducibility(self):
        from environment.error_injector import ErrorInjector
        import random
        ticket = TICKETS[0]
        error_mix = {"clean": 0.30, "obvious": 0.40, "subtle": 0.20, "multi": 0.10}
        inj1 = ErrorInjector(rng=random.Random(42))
        w1, e1 = inj1.generate_worker_response(ticket, error_mix=error_mix)
        inj2 = ErrorInjector(rng=random.Random(42))
        w2, e2 = inj2.generate_worker_response(ticket, error_mix=error_mix)
        assert w1.classification == w2.classification
        assert w1.priority == w2.priority
        assert len(e1) == len(e2)

    def test_error_fields_are_valid(self):
        from environment.error_injector import ErrorInjector
        import random
        valid_fields = {"classification", "priority", "department", "response", "resolution_actions"}
        injector = ErrorInjector(rng=random.Random(42))
        ticket = TICKETS[0]
        for _ in range(10):
            worker, errors = injector.generate_worker_response(ticket, error_mix={"obvious": 0.5, "subtle": 0.5})
            for e in errors:
                assert e.field in valid_fields
