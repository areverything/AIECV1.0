"""
Activity 1: RAGAS eval of Fireworks AI RAG vs OpenAI RAG + tracking of token usage and cost per query via LangSmith

Run:  uv run python activity1_ragas_eval.py

The two pipelines are identical except for the provider config, so any score
difference comes from the models. A single fixed judge (gpt-4.1-mini) scores
both, so the numbers are comparable. LangSmith traces every call for cost.

Targets ragas 0.4.x. The metric imports emit deprecation warnings (they move to
ragas.metrics.collections in v1.0) but still work here.
"""
from __future__ import annotations

import os
import sys
import types
import warnings

# The deprecation notices here come from ragas 0.4.x (metric imports, the Langchain
# wrappers) and langchain-community's sunset path. None are actionable in this
# script, so silence them for a clean run. Set before the imports that emit them.
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv

load_dotenv()

# ragas 0.4.3 hard-imports Vertex AI from langchain-community at load time, but
# langchain-community 0.4.x (required here for langchain-core 1.x / langgraph 1.0)
# removed those names. Register harmless stubs so `import ragas` succeeds; we never
# use Vertex (the judge is OpenAI), so the stubs are never called.
class _MissingVertex:  # noqa: D401 - placeholder for a removed integration
    pass


_vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_vertexai_stub.ChatVertexAI = _MissingVertex
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vertexai_stub)

import langchain_community.llms as _lc_llms

_lc_llms.VertexAI = _MissingVertex

import tiktoken
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

FIREWORKS_BASE = "https://api.fireworks.ai/inference/v1"
DATA_DIR = os.environ.get("RAG_DATA_DIR", "data")

# USD per 1M tokens, as (input, output). Chat-generation rates only; embeddings are
# a negligible fraction of per-query cost. Verify against the current pricing pages:
# Fireworks https://fireworks.ai/pricing , OpenAI https://openai.com/api/pricing
RATES = {
    "accounts/fireworks/models/gpt-oss-20b": (0.07, 0.30),
    "gpt-4.1-mini": (0.40, 1.60),
}


def _rate_for(model_name: str):
    """Resolve (input, output) $/1M for a model name, tolerating provider prefixes."""
    if model_name in RATES:
        return RATES[model_name]
    for key, rate in RATES.items():
        if key in model_name or key.split("/")[-1] in model_name:
            return rate
    return None


def _tiktoken_len(text: str) -> int:
    return len(tiktoken.encoding_for_model("gpt-4o").encode(text))


# ---- provider configs: the ONLY thing that differs between the two pipelines ----
def fireworks_models():
    key = os.environ["FIREWORKS_API_KEY"]
    chat = ChatOpenAI(
        model=os.environ.get("FIREWORKS_CHAT_MODEL", "accounts/fireworks/models/gpt-oss-20b"),
        openai_api_base=FIREWORKS_BASE, openai_api_key=key, temperature=0,
    )
    emb = OpenAIEmbeddings(
        model=os.environ.get("FIREWORKS_EMBEDDING_MODEL", "accounts/fireworks/models/qwen3-embedding-8b"),
        openai_api_base=FIREWORKS_BASE, openai_api_key=key,
        check_embedding_ctx_length=False, dimensions=4096,
    )
    return chat, emb


def openai_models():
    key = os.environ["OPENAI_API_KEY"]
    chat = ChatOpenAI(model="gpt-4.1-mini", openai_api_key=key, temperature=0)
    emb = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=key)
    return chat, emb


# ---- shared pipeline (same chunks, prompt, retriever for both providers) ----
PROMPT = ChatPromptTemplate.from_messages([(
    "human",
    "\n#CONTEXT:\n{context}\n\nQUERY:\n{query}\n\n"
    "Use the provided context to answer the query. Only use the provided context. "
    'If the answer is not in the context, respond with "I don\'t know".',
)])


def load_documents():
    return DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyMuPDFLoader).load()


def chunk(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750, chunk_overlap=0, length_function=_tiktoken_len
    )
    return splitter.split_documents(documents)


def build_answer_fn(chat, emb, chunks, collection):
    store = QdrantVectorStore.from_documents(
        documents=chunks, embedding=emb, location=":memory:", collection_name=collection
    )
    retriever = store.as_retriever(search_kwargs={"k": 4})

    def answer(question: str):
        ctx = retriever.invoke(question)
        text = (PROMPT | chat | StrOutputParser()).invoke(
            {"query": question, "context": ctx}
        )
        return text, [c.page_content for c in ctx]

    return answer


# ---- curated test set (question, reference answer), grounded in the PDF ----
# The PDF is the 2021 AAHA/AAFP Feline Life Stage Guidelines. RAGAS's automatic
# knowledge-graph generator is fragile on this document (its HeadlineSplitter
# errors when headline extraction returns nothing), so we use a fixed set whose
# answers are all present in the source. Both pipelines are graded against these
# same references, which is what keeps the A/B comparison fair.
TESTSET = [
    (
        "Into how many life stages does the 2021 AAHA/AAFP framework divide a cat's "
        "lifespan, and what are they?",
        "Five life stages: kitten, young adult, mature adult, senior, and end-of-life.",
    ),
    (
        "How often does the Task Force recommend examinations for all cats?",
        "A minimum of annual examinations for all cats, with increasing frequency as "
        "appropriate for the individual cat's needs.",
    ),
    (
        "How frequently should senior cats be examined?",
        "Senior cats should be seen at least every 6 months, and more frequently if they "
        "have chronic conditions.",
    ),
    (
        "Why should each examination visit include a life stage assessment?",
        "Because a cat can transition from one life stage to another in a short period, and "
        "most of the healthcare plan is guided by the cat's life stage, so each visit should "
        "reassess it.",
    ),
    (
        "What is the most fundamental presentation factor the practitioner encounters in a "
        "regular examination visit?",
        "The feline patient's life stage.",
    ),
    (
        "Does the Task Force consider end of life to be a separate feline life stage?",
        "Yes, the Task Force considers end of life and its precursor events to be a separate "
        "feline life stage.",
    ),
    (
        "Which discussion topics may only need to be covered once, during an initial "
        "consultation?",
        "Topics such as sterilization, claw care, the importance of identification and "
        "microchipping, and disaster preparedness may be covered once in an initial "
        "consultation.",
    ),
    (
        "What kind of photographs does the guideline recommend to help monitor a cat's body "
        "and muscle condition as it ages?",
        "Dorsal and lateral photographs of the patient, to help monitor body condition score "
        "and muscle condition score over time.",
    ),
    (
        "According to the guidelines, what is one of the most significant and underdiagnosed "
        "diseases in cats?",
        "Osteoarthritis, or degenerative joint disease, which is described as one of the most "
        "significant and underdiagnosed diseases in cats.",
    ),
    (
        "What proportion of cats may present with clinical signs associated with degenerative "
        "joint disease?",
        "Published estimates suggest that 40 to 92% of all cats may present with clinical "
        "signs associated with degenerative joint disease.",
    ),
    (
        "When does a kitten's sensitive socialization period begin and end?",
        "It begins as early as 2 to 3 weeks of age and may be closing by 9 to 10 weeks.",
    ),
    (
        "How much pleasant interaction with people should kittens ideally have each day?",
        "Ideally, kittens should have pleasant interactions with people for 30 to 60 minutes "
        "per day.",
    ),
    (
        "What important behaviors does a kitten learn from the queen (mother cat)?",
        "Acceptance of foods, toileting habits, substrate preferences, and a fear response to "
        "other species, including people and dogs.",
    ),
    (
        "Which scratching substrate did cats use most often when offered, and which did older "
        "cats aged 10 to 14 prefer?",
        "Rope was the most frequently used substrate when offered, while cats between 10 and "
        "14 years of age preferred carpet.",
    ),
    (
        "During the physical examination of a mature adult or senior cat, what areas receive "
        "particular focus?",
        "Pain assessment, abdominal and thyroid palpation, a detailed musculoskeletal "
        "examination for signs of osteoarthritis, and a fundic examination to detect "
        "ophthalmic disease or hypertension.",
    ),
]


# ---- run each pipeline over the test set and score with RAGAS ----
def score_pipeline(name, answer_fn, qa_pairs, judge_llm, judge_emb):
    from langchain_core.callbacks import get_usage_metadata_callback
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        answer_correctness, context_precision, context_recall, faithfulness,
    )

    # Capture generation tokens for THIS pipeline's answers only. The callback scope
    # ends before evaluate() runs, so the judge's tokens are excluded from the cost.
    samples = []
    with get_usage_metadata_callback() as usage_cb:
        for question, reference in qa_pairs:
            response, contexts = answer_fn(question)
            samples.append({
                "user_input": question,
                "retrieved_contexts": contexts,
                "response": response,
                "reference": reference,
            })
    usage = {model: dict(u) for model, u in usage_cb.usage_metadata.items()}

    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(
        dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_correctness],
        llm=judge_llm,
        embeddings=judge_emb,
    )

    print(f"\n=== {name} ===")
    print("Averages:", result)

    df = result.to_pandas()
    metric_cols = [
        c
        for c in ("context_precision", "context_recall", "faithfulness", "answer_correctness")
        if c in df.columns
    ]
    print("Per-question:")
    for i, row in df.iterrows():
        scores = {c: round(float(row[c]), 3) for c in metric_cols}
        print(f"  Q{i + 1}. {str(row.get('user_input', ''))[:75]}")
        print(f"      {scores}")

    return {
        "quality": {c: float(df[c].mean()) for c in metric_cols},
        "usage": usage,
        "n_queries": len(qa_pairs),
    }


def print_comparison(results: dict[str, dict]) -> None:
    """Side-by-side average scores for both providers, with a per-metric winner."""
    names = list(results)  # ["fireworks_oss", "openai"]
    metrics = ["context_precision", "context_recall", "faithfulness", "answer_correctness"]

    print("\n=== Comparison (averages) ===")
    print(f"{'metric':<20}{names[0]:>16}{names[1]:>12}{'winner':>14}")
    for m in metrics:
        a, b = results[names[0]]["quality"].get(m), results[names[1]]["quality"].get(m)
        if a is None or b is None:
            continue
        winner = names[0] if a > b else names[1] if b > a else "tie"
        print(f"{m:<20}{a:>16.4f}{b:>12.4f}{winner:>14}")


def print_cost_breakdown(results: dict[str, dict]) -> None:
    """Generation cost per provider, from measured tokens x published rates."""
    print("\n=== Cost breakdown (answer generation; embeddings excluded, negligible) ===")
    print(f"{'provider':<16}{'in tok':>10}{'out tok':>10}{'$/query':>12}{'$/100k q':>12}")
    for name, r in results.items():
        in_tok = out_tok = 0
        cost = 0.0
        for model, u in r["usage"].items():
            it, ot = u.get("input_tokens", 0), u.get("output_tokens", 0)
            in_tok += it
            out_tok += ot
            rate = _rate_for(model)
            if rate:
                cost += it / 1e6 * rate[0] + ot / 1e6 * rate[1]
        per_q = cost / r["n_queries"] if r["n_queries"] else 0.0
        print(f"{name:<16}{in_tok:>10}{out_tok:>10}{per_q:>12.5f}{per_q * 100_000:>12.2f}")
    print("Rates ($/1M in, out):", {k: v for k, v in RATES.items()})


def langsmith_project_url(project_name: str) -> str:
    """Best-effort web URL for the LangSmith project. Never fatal; falls back to root."""
    try:
        from langsmith import Client

        client = Client()
        project = client.read_project(project_name=project_name)
        web = client._web_url or "https://smith.langchain.com"
        return f"{web}/o/{client._get_tenant_id()}/projects/p/{project.id}"
    except Exception as exc:  # noqa: BLE001 - the URL is a convenience, not required
        return (
            f"https://smith.langchain.com  (open project '{project_name}'; "
            f"url lookup failed: {exc})"
        )


def main():
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    # One fixed judge for BOTH pipelines, so scores are comparable.
    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4.1-mini", temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"])
    )
    judge_emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.environ["OPENAI_API_KEY"])
    )

    documents = load_documents()
    qa_pairs = TESTSET
    chunks = chunk(documents)

    fw_chat, fw_emb = fireworks_models()
    oa_chat, oa_emb = openai_models()

    results = {}
    results["fireworks_oss"] = score_pipeline(
        "fireworks_oss",
        build_answer_fn(fw_chat, fw_emb, chunks, "rag_fireworks"),
        qa_pairs, judge_llm, judge_emb,
    )
    results["openai"] = score_pipeline(
        "openai",
        build_answer_fn(oa_chat, oa_emb, chunks, "rag_openai"),
        qa_pairs, judge_llm, judge_emb,
    )

    print_comparison(results)
    print_cost_breakdown(results)

    project = os.environ.get("LANGSMITH_PROJECT", "default")
    print("\nCost: open the LangSmith project and read tokens/cost per run:")
    print(f"  {langsmith_project_url(project)}")
    print("Fireworks pricing isn't known to LangSmith, so multiply its traced")
    print("token counts by the Fireworks per-token rate to fill the cost column.")


if __name__ == "__main__":
    main()