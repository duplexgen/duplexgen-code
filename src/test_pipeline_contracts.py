"""Contracts between what the pipeline declares and what it actually reads.

Each test here pins a regression in which a declaration and its use drifted
apart: `CONFIG` keys nothing reads (and reads of keys `CONFIG` never defines),
argparse flags a stage consumes without declaring, dataset registries that
disagree between stages, corpus paths that resolve into another stage's output
directory, and vLLM-only `extra_body` fields hardcoded onto OpenAI calls.
"""

import argparse
import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_config_declares_every_key_it_reads():
    from src.synthesis.core import CONFIG

    src = (ROOT / "src/synthesis/core.py").read_text()
    tree = ast.parse(src)
    read_keys = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "get"
            and getattr(node.func.value, "id", "") == "CONFIG"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            read_keys.add(node.args[0].value)
    missing = sorted(read_keys - set(CONFIG))
    assert not missing, f"CONFIG.get() reads keys CONFIG never defines: {missing}"


def test_config_declares_no_unread_keys():
    from src.synthesis.core import CONFIG

    src = (ROOT / "src").rglob("*.py")
    body = "\n".join(p.read_text() for p in src)
    unread = [k for k in CONFIG if body.count(f'"{k}"') + body.count(f"'{k}'") <= 1]
    assert not unread, f"CONFIG declares keys nothing reads: {unread}"


def test_load_in_4bit_rejects_the_string_False():
    """type=bool makes bool('False') == True, silently inverting the flag."""
    src = (ROOT / "src/train_turntaking_hf.py").read_text()
    assert (
        'add_argument("--load_in_4bit", type=bool' not in src
    ), "type=bool means --load_in_4bit False enables 4-bit quantization"
    # And the replacement must actually work:
    p = argparse.ArgumentParser()
    p.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    assert p.parse_args([]).load_in_4bit is True
    assert p.parse_args(["--no-load_in_4bit"]).load_in_4bit is False


def test_no_duplicated_developer_prefix_in_prompts():
    src = (ROOT / "src/speechify_prompts.py").read_text()
    assert "Developer: Developer:" not in src


def _declared_arg_dests(tree: ast.AST) -> set:
    """Every attribute name argparse will put on the Namespace."""
    dests = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"):
            continue
        explicit = [k.value.value for k in node.keywords if k.arg == "dest"]
        if explicit:
            dests.add(explicit[0])
            continue
        flags = [a.value for a in node.args if isinstance(a, ast.Constant)]
        longest = max((f for f in flags if f.startswith("--")), key=len, default=None)
        if longest:
            dests.add(longest.lstrip("-").replace("-", "_"))
    return dests


def test_speechify_run_reads_only_flags_it_declares():
    """args.max_samples was never a flag: 5 of 7 datasets died on AttributeError."""
    tree = ast.parse((ROOT / "src/speechify_run.py").read_text())
    declared = _declared_arg_dests(tree)
    read = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "args"
    }
    missing = sorted(read - declared)
    assert not missing, f"speechify_run.py reads args.{{{','.join(missing)}}}, never declared"


def test_data_root_is_optional_for_hub_backed_datasets():
    """interviewer/soda load from the Hub and persuader takes --input_path."""
    from src.speechify_run import DATA_ROOT_DATASETS

    assert DATA_ROOT_DATASETS.isdisjoint({"interviewer", "soda", "persuader"})
    tree = ast.parse((ROOT / "src/speechify_run.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            flags = [a.value for a in node.args if isinstance(a, ast.Constant)]
            if "--data-root" in flags:
                required = [k.value.value for k in node.keywords if k.arg == "required"]
                assert required in ([], [False]), "--data-root must not be globally required"


def test_soda_filter_takes_the_model_it_uses():
    """A default model here silently diverges from --llm_model_name."""
    import inspect

    from src.speechify_datasets import is_competitive_scenario, iter_soda

    for fn in (is_competitive_scenario, iter_soda):
        model = inspect.signature(fn).parameters["model"]
        assert model.default is inspect.Parameter.empty, f"{fn.__name__} defaults its model"
    assert 'model="gpt-4.1"' not in (ROOT / "src/speechify_datasets.py").read_text()


def _budget(train, test):
    from src.speechify_run import SampleBudget

    return SampleBudget(train=train, test=test)


def test_split_in_halves_honours_both_budgets():
    from src.speechify_run import split_in_halves

    items = list(range(1000))
    (_, train), (_, test) = split_in_halves(items, _budget(3, 7))
    assert len(train) == 3 and len(test) == 7
    # Evenly spaced across the whole source, not the first ten records.
    assert train + test == [items[i] for i in range(0, 1000, 100)]


def test_split_in_halves_is_unaffected_by_variant_expansion():
    """Budgets count dialogues, so --max_variants must not inflate the test split."""
    from src.speechify_run import split_in_halves

    for variants in (1, 2, 5):
        expanded = [(i, v) for i in range(200) for v in range(variants)]
        (_, train), (_, test) = split_in_halves(expanded, _budget(5, 9))
        assert (len(train), len(test)) == (5, 9), f"broke at max_variants={variants}"


def test_zero_means_no_limit():
    from src.speechify_run import SampleBudget, take

    args = argparse.Namespace(max_train_samples=0, max_test_samples=4)
    budget = SampleBudget.from_args(args)
    assert budget.train is None and budget.test == 4
    assert budget.total is None, "an unlimited split makes the combined budget unlimited"
    assert take(list(range(50)), budget.for_split("train")) == list(range(50))


def test_stage1_offers_the_same_scenarios_as_the_downstream_stages():
    """A scenario Stage 1 can convert but Stage 4 rejects is a dead end."""
    from src.prepare_corpus import DATASET_CHOICES as PREPARE_CHOICES
    from src.speechify_run import DATASET_CHOICES as STAGE1_CHOICES
    from src.synthesis.run import DATASET_CHOICES as STAGE4_CHOICES

    assert set(STAGE1_CHOICES) == set(STAGE4_CHOICES) == set(PREPARE_CHOICES)
    assert len(STAGE1_CHOICES) == 6


def test_topiocqa_is_gone():
    """Explored during development, absent from every released artifact."""
    for path in ("src/speechify_run.py", "src/speechify_datasets.py"):
        assert "topiocqa" not in (ROOT / path).read_text().lower()


def test_every_local_corpus_has_a_declared_path():
    from src.speechify_run import DATA_ROOT_DATASETS, LOCAL_CORPUS_PATHS, SEPARATE_FILE_LOADERS

    assert DATA_ROOT_DATASETS == set(LOCAL_CORPUS_PATHS)
    assert DATA_ROOT_DATASETS.isdisjoint({"interviewer", "soda", "persuader"})
    for name in SEPARATE_FILE_LOADERS:
        assert set(LOCAL_CORPUS_PATHS[name]) == {"train", "test"}


def test_soda_filter_does_not_dump_every_accepted_dialogue():
    """The classifier printed each accepted dialogue and its verdict, which
    interleaves with its own tqdm bar over all 149k SODA records."""
    src = (ROOT / "src/speechify_datasets.py").read_text()
    tree = ast.parse(src)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "is_competitive_scenario"
    )
    printed = [
        n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "print"
    ]
    # The one surviving print reports a classification error; nothing else.
    assert len(printed) == 1, f"is_competitive_scenario has {len(printed)} print() calls"


def test_test_parse_dumps_land_clear_of_converted_output():
    """--test-parse wrote to the very path a real run writes to, so the README's
    own order -- smoke-test into results/, then convert into results/ -- left
    every dialogue already on disk. The real run skipped all 25, printed a full
    progress bar, and exited 0 having converted nothing."""
    from src.speechify_run import split_output_dir

    real = split_output_dir("results/", "interviewer", "train", test_parse=False)
    dumped = split_output_dir("results/", "interviewer", "train", test_parse=True)

    assert real != dumped
    assert real.parent.name == "text_dialogue_interviewer"
    # Stage 3 discovers scenarios by globbing text_dialogue_* under a root
    # (train_turntaking_hf.py:267), so a dump wearing that prefix would show up
    # as a scenario of its own.
    assert not any(part.startswith("text_dialogue_") for part in dumped.parts)


def test_resume_skips_only_finished_conversions(tmp_path):
    """Resume treats any existing file as done. A parse dump left by an older
    version, or a file truncated by an interrupted run, must be redone."""
    from src.speechify_run import is_converted

    converted = tmp_path / "converted.json"
    converted.write_text(json.dumps({"meta": {"style": "spoken"}, "config": {}}))
    dump = tmp_path / "dump.json"
    dump.write_text(json.dumps({"meta": {"split": "train"}}))
    truncated = tmp_path / "truncated.json"
    truncated.write_text('{"example_id": "work_0000", "hist')

    assert is_converted(converted)
    assert not is_converted(dump)
    assert not is_converted(truncated)
    assert not is_converted(tmp_path / "absent.json")


def test_input_path_overrides_data_root_for_socraticlm():
    """socraticlm is documented as accepting --input_path instead of --data-root."""
    from src.speechify_run import corpus_path

    args = argparse.Namespace(input_path="/tmp/socra.json", data_root=None)
    assert corpus_path("socraticlm", "all", args) == "/tmp/socra.json"

    args = argparse.Namespace(input_path=None, data_root="/data")
    assert corpus_path("socraticlm", "all", args).startswith("/data/SocraticLM")

    # A single path cannot stand in for both files of a split-per-file corpus.
    args = argparse.Namespace(input_path="/tmp/x.json", data_root=None)
    with pytest.raises(ValueError, match="--data-root is required"):
        corpus_path("multiwoz", "train", args)


def test_no_stage_hardcodes_the_vllm_only_extra_body():
    """`chat_template_kwargs` is a vLLM extension; api.openai.com answers

        400 Unrecognized request argument supplied: chat_template_kwargs

    so every call site must route it through no_thinking_extra_body(), which
    returns None for an OpenAI model name. src/synthesis/core.py sent it
    unconditionally, which made Stage 4's own documented defaults (gpt-4.1
    writer, gpt-4.1-mini boundary detector) die on the first turn."""
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        if path.name == "llm_client.py" or path.name.startswith("test_"):
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if "chat_template_kwargs" in line and not line.lstrip().startswith("#"):
                offenders.append(f"{path.relative_to(ROOT)}:{i}")
    assert not offenders, f"literal chat_template_kwargs outside llm_client.py: {offenders}"


def test_no_thinking_extra_body_is_none_for_openai_models():
    from src.llm_client import no_thinking_extra_body

    for name in ("gpt-4.1", "gpt-4.1-mini", "gpt-5", "o3-mini"):
        assert no_thinking_extra_body(name) is None, name
    for name in ("Qwen/Qwen3-32B", "Qwen/Qwen3-14B", "meta-llama/Llama-3.1-8B"):
        assert no_thinking_extra_body(name) == {"chat_template_kwargs": {"enable_thinking": False}}
