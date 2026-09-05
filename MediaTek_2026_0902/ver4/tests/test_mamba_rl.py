import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from src.data_mamba_rl import amazon_item_group, amazon_subset, load_movielens, load_recommendation_data
    from src.model_mamba_rl import FullRankUpdate, LoRAUpdate, MultiAgentMambaRecommender
    from src.train_mamba_rl import (
        build_transitions, candidate_slates, evaluate, evaluation_interval, history_batch,
        preference_auxiliary_losses, future_target_batch,
        preference_contrastive_loss, soft_target_ranking_loss,
    )


@unittest.skipUnless(HAS_TORCH, "requires torch")
class MultiAgentMambaRLTest(unittest.TestCase):
    def test_agents_have_disjoint_lora_parameters_and_full_catalog_scores(self):
        edges = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        model = MultiAgentMambaRecommender(
            torch.randn(12, 16), dim=8, lora_rank=2, short_window=2,
            graph_edges=edges, graph_users=4,
        )
        histories = torch.tensor([[1, 2, 3], [3, 4, 0]])
        lengths = torch.tensor([3, 2])
        candidates = torch.tensor([[4, 5, 6], [5, 6, 7]])
        output = model(histories, lengths, candidates)
        self.assertEqual(output["coordinator"].shape, (2, 3))
        self.assertEqual(output["preference"].shape, (2, 3))
        self.assertEqual(output["preference_current"].shape, (2, 64))
        self.assertTrue(torch.allclose(output["preference_next"].sum(-1), torch.ones(2)))
        self.assertTrue(torch.all((output["preference_change"] >= 0) & (output["preference_change"] <= 1)))
        self.assertEqual(output["preference_weight"].shape, (2, 1))
        self.assertEqual(output["preference_uncertainty"].shape, (2,))
        self.assertIsInstance(model.preference_agent.transition_lora, LoRAUpdate)
        self.assertEqual(model.preference_agent.transition_lora.a.out_features, 2)
        self.assertEqual(model.preference_agent.state_lora.a.out_features, 2)
        self.assertEqual(model.preference_agent.transition_lora.scale, 8.0)
        self.assertEqual(model.full_catalog_scores(histories, lengths)["coordinator"].shape, (2, 12))
        self.assertEqual(model.adaptation_mode, "multi_lora")
        agents = {
            "long": model.long_agent,
            "short": model.short_agent,
            "preference": model.preference_agent,
            "coordinator": model.coordinator,
        }
        adapter_parameter_sets = []
        for agent_name, adapter_names in model.adapter_routes().items():
            adapters = [getattr(agents[agent_name], name) for name in adapter_names]
            self.assertTrue(all(isinstance(adapter, LoRAUpdate) for adapter in adapters))
            adapter_parameter_sets.append({
                parameter for adapter in adapters for parameter in adapter.parameters()
            })
        for left_index, left in enumerate(adapter_parameter_sets):
            for right in adapter_parameter_sets[left_index + 1:]:
                self.assertTrue(left.isdisjoint(right))

    def test_disabling_lora_preserves_full_rank_agent_updates(self):
        model = MultiAgentMambaRecommender(
            torch.randn(12, 16), dim=8, lora_rank=2,
            use_graph_embeddings=False, enable_lora=False,
        )
        self.assertEqual(model.adaptation_mode, "full_rank")
        agents = {
            "long": model.long_agent,
            "short": model.short_agent,
            "preference": model.preference_agent,
            "coordinator": model.coordinator,
        }
        for agent_name, adapter_names in model.adapter_routes().items():
            self.assertTrue(all(
                isinstance(getattr(agents[agent_name], name), FullRankUpdate)
                for name in adapter_names
            ))
        self.assertFalse(any(isinstance(module, LoRAUpdate) for module in model.modules()))

    def test_evaluation_interval_is_capped_by_steps_per_epoch(self):
        self.assertEqual(evaluation_interval(12000, 5000), 5000)
        self.assertEqual(evaluation_interval(1000, 5000), 1000)
        self.assertEqual(evaluation_interval(0, 5000), 0)

    def test_graph_embeddings_can_be_disabled(self):
        model = MultiAgentMambaRecommender(
            torch.randn(12, 16), dim=8, lora_rank=2, use_graph_embeddings=False,
        )
        model.set_stage("specialists")
        model.set_stage("joint")
        histories = torch.tensor([[1, 2, 3]])
        lengths = torch.tensor([3])
        candidates = torch.tensor([[4, 5, 6]])
        self.assertEqual(model(histories, lengths, candidates)["coordinator"].shape, (1, 3))

    def test_each_training_stage_selects_expected_parameters(self):
        edges = torch.tensor([[0, 1], [4, 5]])
        model = MultiAgentMambaRecommender(torch.randn(8, 6), dim=4, lora_rank=2,
                                           graph_edges=edges, graph_users=4)
        model.set_stage("coordinator")
        self.assertTrue(all(parameter.requires_grad for parameter in model.coordinator.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.long_agent.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.graph.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.preference_agent.parameters()))
        model.set_stage("joint")
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_preference_agent_has_finite_auxiliary_losses_and_gradients(self):
        model = MultiAgentMambaRecommender(
            torch.randn(12, 8), dim=6, lora_rank=2, preference_count=4,
            preference_hidden=8, use_graph_embeddings=False,
        )
        model.set_stage("specialists")
        histories = torch.tensor([[1, 2, 3], [2, 4, 0]])
        lengths = torch.tensor([3, 2])
        candidates = torch.tensor([[4, 5, 6], [5, 6, 7]])
        vectors = model.project_ids(candidates)
        output = model.logits_from_states(model.encode_states(histories, lengths), vectors)
        losses = preference_auxiliary_losses(model, output, vectors[:, 0])
        loss = sum(losses) + torch.nn.functional.cross_entropy(
            output["preference"], torch.zeros(2, dtype=torch.long)
        )
        self.assertTrue(all(torch.isfinite(term) for term in losses))
        loss.backward()
        self.assertTrue(any(
            parameter.grad is not None for parameter in model.preference_agent.parameters()
        ))
        self.assertIsNotNone(model.preference_agent.transition_lora.b.weight.grad)
        self.assertIsNotNone(model.preference_agent.state_lora.b.weight.grad)

    def test_graph_embeddings_receive_joint_training_gradients(self):
        edges = torch.tensor([[0, 1], [4, 5]])
        model = MultiAgentMambaRecommender(torch.randn(8, 6), dim=4, lora_rank=2,
                                           graph_edges=edges, graph_users=4)
        model.set_stage("joint")
        output = model(
            torch.tensor([[1, 2, 3]]), torch.tensor([3]), torch.tensor([[4, 5, 6]]),
        )
        output["coordinator"].sum().backward()
        self.assertTrue(all(parameter.requires_grad for parameter in model.graph.parameters()))
        self.assertTrue(all(parameter.grad is not None for parameter in model.graph.parameters()))

    def test_synthetic_transition_batch(self):
        data = load_recommendation_data("synthetic", None, "/tmp", min_user_events=5)
        transitions = build_transitions(data, maximum=10, seed=1)
        histories, lengths, targets = history_batch(data, transitions[:4], 10, "cpu")
        slates = candidate_slates(targets, data.num_items, 4)
        self.assertEqual(histories.size(0), 4)
        self.assertTrue(torch.all(lengths >= 1))
        self.assertTrue(torch.equal(slates[:, 0], targets))

    def test_transition_budget_prioritises_user_coverage(self):
        data = load_recommendation_data("synthetic", None, "/tmp", min_user_events=5)
        eligible_users = {user for user, history in data.train_by_user.items() if len(history) > 1}
        transitions = build_transitions(data, maximum=len(eligible_users), seed=7)
        self.assertEqual({row.user for row in transitions}, eligible_users)

    def test_future_soft_targets_use_training_sequence_only(self):
        data = load_recommendation_data("synthetic", None, "/tmp", min_user_events=5)
        transition = build_transitions(data, maximum=1, seed=3)
        ids, weights = future_target_batch(data, transition, 3, 0.5, "cpu")
        row = transition[0]
        expected = data.train_by_user[row.user][row.end:row.end + 3]
        self.assertEqual(ids[0, :len(expected)].tolist(), expected)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        data.valid_target[row.user] = data.num_items + 100
        data.test_target[row.user] = data.num_items + 200
        repeated_ids, repeated_weights = future_target_batch(
            data, transition, 3, 0.5, "cpu"
        )
        self.assertTrue(torch.equal(ids, repeated_ids))
        self.assertTrue(torch.equal(weights, repeated_weights))

    def test_soft_ranking_and_multi_positive_contrastive_losses_are_finite(self):
        logits = torch.tensor([[2.0, 1.0, -1.0], [0.5, 1.5, -0.5]])
        positions = torch.tensor([[0, 1], [0, 1]])
        weights = torch.tensor([[0.75, 0.25], [0.75, 0.25]])
        ranking = soft_target_ranking_loss(logits, positions, weights)
        contrastive = preference_contrastive_loss(
            torch.randn(3, 4), torch.randn(3, 4), torch.tensor([1, 1, 2])
        )
        self.assertTrue(torch.isfinite(ranking))
        self.assertTrue(torch.isfinite(contrastive))

    def test_amazon_category_config_name(self):
        self.assertEqual(amazon_subset("amazon-all-beauty"), "raw_review_All_Beauty")
        self.assertEqual(amazon_subset("amazon-movies-and-tv"), "raw_review_Movies_and_TV")
        self.assertEqual(amazon_subset("amazon-games"), "raw_review_Toys_and_Games")
        self.assertEqual(amazon_subset("amazon-toys"), "raw_review_Toys_and_Games")
        self.assertEqual(amazon_item_group("amazon-games"), "games")
        self.assertEqual(amazon_item_group("amazon-toys"), "toys")
        self.assertIsNone(amazon_item_group("amazon-toys-and-games"))

    def test_evaluation_reports_hit_at_5_and_10(self):
        data = load_recommendation_data("synthetic", None, "/tmp", min_user_events=5)
        model = MultiAgentMambaRecommender(
            torch.randn(data.num_items, 8), dim=4, lora_rank=2, use_graph_embeddings=False,
        )
        metrics, _ = evaluate(model, data, "test", 8, 10, "cpu")
        self.assertEqual(metrics["hit@5"], metrics["recall@5"])
        self.assertEqual(metrics["hit@10"], metrics["recall@10"])
        self.assertIn("preference_change_probability", metrics)

    def test_movielens_csv_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "movies.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("movieId", "title", "genres"))
                writer.writerow(("10", "A Film", "Drama"))
            with (root / "ratings.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("userId", "movieId", "rating", "timestamp"))
                writer.writerow(("u1", "10", "5", "123"))
                writer.writerow(("u1", "10", "2", "124"))
            rows = load_movielens("movielens-1m", str(root), directory, None, 4.0)
            self.assertEqual(rows, [("u1", "10", 123, "A Film Genres: Drama")])


if __name__ == "__main__":
    unittest.main()
