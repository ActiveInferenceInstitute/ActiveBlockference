"""Optional research orchestration with a deterministic local provider."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ResearchProvider(Protocol):
    """Provider interface used by the local orchestration workflow."""

    def complete(self, prompt: str) -> str:
        """Return a response for a research prompt."""


@dataclass(frozen=True)
class ResearchWorkspace:
    """Explicit file locations for a research exchange."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @property
    def project_description(self) -> Path:
        return self.root / "O_Project_description.txt"

    @property
    def requests(self) -> Path:
        return self.root / "requests.txt"

    @property
    def research_data(self) -> Path:
        return self.root / "research_data.txt"

    @property
    def report(self) -> Path:
        return self.root / "report.txt"

    def prepare(self) -> None:
        """Create the workspace and required input files."""

        self.root.mkdir(parents=True, exist_ok=True)
        if not self.requests.exists():
            self.requests.write_text("Summarize the supplied research corpus.\n", encoding="utf-8")
        if not self.research_data.exists():
            self.research_data.write_text("", encoding="utf-8")


class OfflineCorpusProvider:
    """Deterministic provider that answers from a local text corpus."""

    def __init__(self, corpus: str | Path | None = None) -> None:
        if corpus is None:
            self.corpus = (
                "Active inference links observations, beliefs, actions, and expected free energy."
            )
        else:
            corpus_path = Path(corpus)
            self.corpus = (
                corpus_path.read_text(encoding="utf-8") if corpus_path.is_file() else str(corpus)
            )

    def complete(self, prompt: str) -> str:
        """Return stable corpus sentences selected by prompt terms."""

        terms = {word.lower().strip(".,:;!?()[]{}") for word in prompt.split() if len(word) > 2}
        sentences = [
            sentence.strip()
            for sentence in self.corpus.replace("\n", " ").split(".")
            if sentence.strip()
        ]
        ranked = sorted(
            enumerate(sentences),
            key=lambda item: (-sum(term in item[1].lower() for term in terms), item[0]),
        )
        selected = [sentence for _, sentence in ranked[:3]]
        return (
            "Offline corpus synthesis:\n"
            + ". ".join(selected)
            + ("." if selected else " No matching corpus text.")
        )


class OpenAIProvider:
    """Explicit OpenAI provider requiring the optional SDK and API key."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OpenAI provider requires OPENAI_API_KEY")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI provider requires the optional openai package") from exc
        self.model = model
        self.timeout = 60.0 if timeout is None else timeout
        self.client = OpenAI(api_key=key, timeout=self.timeout)

    def complete(self, prompt: str) -> str:
        """Send one prompt to the configured OpenAI model."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("OpenAI provider returned an empty response")
        return content


def process_csv(
    csv_file_path: str | Path, instructions: str = "", output_path: str | Path | None = None
) -> str:
    """Produce deterministic structural analysis for a CSV file."""

    path = Path(csv_file_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError("CSV file is empty")
    header = rows[0]
    result = {
        "file": str(path),
        "rows": max(len(rows) - 1, 0),
        "columns": header,
        "instructions": instructions,
    }
    text = json.dumps(result, indent=2)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return text


@dataclass
class ResearchOrchestrator:
    """Coordinate request, corpus, and report files through one provider."""

    provider: ResearchProvider
    workspace: ResearchWorkspace

    def run(self) -> Path:
        """Process the current request and write a report."""

        self.workspace.prepare()
        request = self.workspace.requests.read_text(encoding="utf-8").strip()
        corpus = ""
        if self.workspace.project_description.exists():
            corpus = self.workspace.project_description.read_text(encoding="utf-8")
        prompt = f"Request:\n{request}\nCorpus:\n{corpus}"
        answer = self.provider.complete(prompt)
        self.workspace.research_data.write_text(answer + "\n", encoding="utf-8")
        report = f"Research report\n\nRequest\n{request}\n\nFindings\n{answer}\n"
        self.workspace.report.write_text(report, encoding="utf-8")
        return self.workspace.report


def build_provider(
    name: str,
    *,
    corpus: str | Path | None = None,
    model: str = "gpt-4o-mini",
    timeout: float | None = None,
) -> ResearchProvider:
    """Build an explicitly selected provider."""

    if name == "offline":
        return OfflineCorpusProvider(corpus)
    if name == "openai":
        return OpenAIProvider(model=model, timeout=timeout)
    raise ValueError("provider must be 'offline' or 'openai'")


__all__ = [
    "OfflineCorpusProvider",
    "OpenAIProvider",
    "ResearchOrchestrator",
    "ResearchProvider",
    "ResearchWorkspace",
    "build_provider",
    "process_csv",
]
