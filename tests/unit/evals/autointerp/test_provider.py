import json
from functools import partial
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pytest
import torch
from openai import OpenAI, RateLimitError

import sae_bench.evals.autointerp.main as autointerp
from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig


@pytest.fixture
def interpreter():
    return autointerp.AutoInterp(
        cfg=AutoInterpEvalConfig(n_latents=None, override_latents=[0]),
        model=Mock(),
        sae=Mock(),
        tokenized_dataset=torch.zeros((1, 2), dtype=torch.long),
        sparsity=torch.ones(1),
        device="cpu",
        api_key="test-key",
    )


def completion_response(contents: list[str]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "test-completion",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": i,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
                for i, content in enumerate(contents)
            ],
        },
    )


@pytest.mark.parametrize(
    "judge_model_name,judge_base_url,environment_url,expected_url",
    [
        ("gpt-4o-mini", None, None, "https://api.openai.com/v1/chat/completions"),
        (
            "local-model",
            "http://localhost:8000/v1",
            None,
            "http://localhost:8000/v1/chat/completions",
        ),
        (
            "gpt-4o-mini",
            None,
            "http://localhost:8001/v1",
            "http://localhost:8001/v1/chat/completions",
        ),
        (
            "local-model",
            "http://localhost:8000/v1",
            "http://localhost:8001/v1",
            "http://localhost:8000/v1/chat/completions",
        ),
    ],
)
def test_judge_request(
    interpreter: autointerp.AutoInterp,
    monkeypatch: pytest.MonkeyPatch,
    judge_model_name: str,
    judge_base_url: str | None,
    environment_url: str | None,
    expected_url: str,
):
    interpreter.cfg.judge_model_name = judge_model_name
    interpreter.cfg.judge_base_url = judge_base_url
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    if environment_url is not None:
        monkeypatch.setenv("OPENAI_BASE_URL", environment_url)

    requests = []

    def handle(request: httpx.Request):
        requests.append(request)
        return completion_response([" first ", " second "])

    messages: autointerp.Messages = [
        {"role": "user", "content": "Explain this feature"}
    ]
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with patch.object(autointerp, "OpenAI", partial(OpenAI, http_client=client)):
            response, logs = interpreter.get_api_response(messages, 25, n_completions=2)

    assert response == ["first", "second"]
    assert "first" in logs
    assert len(requests) == 1
    assert str(requests[0].url) == expected_url
    assert requests[0].headers["authorization"] == "Bearer test-key"
    assert json.loads(requests[0].content) == {
        "model": judge_model_name,
        "messages": messages,
        "n": 2,
        "max_tokens": 25,
        "stream": False,
    }


def test_provider_error_is_propagated(interpreter: autointerp.AutoInterp):
    interpreter.cfg.judge_base_url = "http://localhost:8000/v1"
    transport = httpx.MockTransport(
        lambda request: httpx.Response(429, json={"error": {"message": "Rate limited"}})
    )
    with httpx.Client(transport=transport) as client:
        with patch.object(
            autointerp, "OpenAI", partial(OpenAI, http_client=client, max_retries=0)
        ):
            with pytest.raises(RateLimitError, match="Rate limited"):
                interpreter.get_api_response(
                    [{"role": "user", "content": "Explain"}], 25
                )


@pytest.mark.parametrize("custom_judge", [False, True])
def test_cli_configures_judge(custom_judge: bool):
    arguments = [
        "--model_name",
        "pythia-70m-deduped",
        "--sae_regex_pattern",
        "test-release",
        "--sae_block_pattern",
        "test-sae",
    ]
    if custom_judge:
        arguments += [
            "--judge_model_name",
            "local-model",
            "--judge_base_url",
            "http://localhost:8000/v1",
        ]
    args = autointerp.arg_parser().parse_args(arguments)
    with patch.object(
        autointerp, "get_saes_from_regex", return_value=[("release", "sae")]
    ):
        config, _ = autointerp.create_config_and_selected_saes(args)

    assert config.model_name == "pythia-70m-deduped"
    assert config.judge_model_name == ("local-model" if custom_judge else "gpt-4o-mini")
    assert config.judge_base_url == (
        "http://localhost:8000/v1" if custom_judge else None
    )


def test_run_eval_uses_judge_for_both_phases_and_saves_config(tmp_path: Path):
    config = AutoInterpEvalConfig(
        model_name="target-model",
        judge_model_name="local-model",
        judge_base_url="http://localhost:8000/v1",
        n_latents=None,
        override_latents=[0, 1],
        n_top_ex_for_scoring=1,
        n_iw_sampled_ex_for_scoring=0,
        n_random_ex_for_scoring=1,
        llm_dtype="float32",
    )
    model = Mock()
    model.to_str_tokens.return_value = ["cat"]
    active = autointerp.Example([1], [1.0], 0.5, model)
    inactive = autointerp.Example([1], [0.0], 0.5, model)
    generation = autointerp.Examples([active])
    scoring = autointerp.Examples([active, inactive])
    sae = Mock()
    sae.to.return_value = sae
    sae.cfg.to_dict.return_value = {}
    requests = []

    def handle(request: httpx.Request):
        requests.append(request)
        body = json.loads(request.content)
        content = (
            "This neuron activates on cats."
            if body["max_tokens"] == config.max_tokens_in_explanation
            else "1"
        )
        return completion_response([content])

    with (
        httpx.Client(transport=httpx.MockTransport(handle)) as client,
        patch.object(autointerp, "OpenAI", partial(OpenAI, http_client=client)),
        patch.object(
            autointerp.HookedTransformer,
            "from_pretrained_no_processing",
            return_value=model,
        ) as load_model,
        patch.object(
            autointerp.general_utils,
            "load_and_format_sae",
            return_value=("test-sae", sae, torch.ones(2)),
        ),
        patch.object(
            autointerp.dataset_utils,
            "load_and_tokenize_dataset",
            return_value=torch.zeros((1, 2), dtype=torch.long),
        ),
        patch.object(
            autointerp.AutoInterp,
            "gather_data",
            return_value=({0: generation, 1: generation}, {0: scoring, 1: scoring}),
        ),
    ):
        results = autointerp.run_eval(
            config,
            [("test-release", "test-sae")],
            "cpu",
            "test-key",
            str(tmp_path / "results"),
            artifacts_path=str(tmp_path / "artifacts"),
        )

    load_model.assert_called_once_with(
        "target-model", device="cpu", dtype=torch.float32
    )
    assert len(requests) == 4
    assert {str(request.url) for request in requests} == {
        "http://localhost:8000/v1/chat/completions"
    }
    bodies = [json.loads(request.content) for request in requests]
    assert {body["model"] for body in bodies} == {"local-model"}
    assert sorted(body["max_tokens"] for body in bodies) == sorted(
        [config.max_tokens_in_explanation, config.max_tokens_in_prediction] * 2
    )
    saved_results = list((tmp_path / "results").glob("*.json"))
    assert len(saved_results) == 1
    for result in [
        results["test-release_test-sae"],
        json.loads(saved_results[0].read_text()),
    ]:
        assert result["eval_config"]["model_name"] == "target-model"
        assert result["eval_config"]["judge_model_name"] == "local-model"
        assert result["eval_config"]["judge_base_url"] == "http://localhost:8000/v1"
        assert result["eval_result_metrics"]["autointerp"]["autointerp_score"] == 1.0
        assert "test-key" not in json.dumps(result)
