"""Ollama provider: open vision models on your own machine or a free cloud GPU."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from observatory.detect import vision
from observatory.detect.models import Frame

pytest.importorskip("openai")


@pytest.fixture
def fake_ollama(monkeypatch):
    """A local server speaking the slice of Ollama's OpenAI-compatible API we use."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            seen.append(("GET", self.path, None))
            self._send({"object": "list", "data": [{"id": "qwen2.5vl:7b", "object": "model", "owned_by": "library"},
                                                   {"id": "llama3.2:3b", "object": "model", "owned_by": "library"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(("POST", self.path, body))
            answer = {"ai_likelihood": 0.8, "summary": "Waxy skin.",
                      "findings": [{"frame_index": 0, "observation": "hand merges", "direction": "ai"}]}
            self._send({"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": json.dumps(answer)}}],
                        "usage": {"prompt_tokens": 900, "completion_tokens": 40, "total_tokens": 940}})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OLLAMA_HOST", f"127.0.0.1:{server.server_port}")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)
    yield seen
    server.shutdown()


def _frames(tmp_path, n):
    for i in range(n):
        (tmp_path / f"f{i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    return [Frame(index=i, timestamp=float(i), file=f"f{i}.jpg") for i in range(n)]


def test_assess_with_ollama_batches_frames_and_reads_json(tmp_path, fake_ollama):
    result, usage = vision.assess_with("ollama", "qwen2.5vl:7b", _frames(tmp_path, 6), tmp_path)
    posts = [b for m, path, b in fake_ollama if m == "POST"]
    assert [p for m, p, _ in fake_ollama] == ["/v1/chat/completions"] * 2  # 6 frames in batches of 4
    images = [part for part in posts[0]["messages"][0]["content"] if part["type"] == "image_url"]
    assert len(images) == 4 and images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert posts[0]["model"] == "qwen2.5vl:7b" and posts[0]["response_format"] == {"type": "json_object"}
    assert result.ai_likelihood == pytest.approx(0.8, abs=0.01) and len(result.findings) == 2
    assert (usage.input_tokens, usage.calls) == (1800, 2)


def test_list_models_from_ollama(fake_ollama):
    assert vision.list_vision_models("ollama", "") == ["llama3.2:3b", "qwen2.5vl:7b"]


@pytest.mark.parametrize("host, base", [
    ("", "http://localhost:11434/v1"),
    ("127.0.0.1:11434", "http://127.0.0.1:11434/v1"),
    ("https://gpu.example.com/", "https://gpu.example.com/v1"),
])
def test_ollama_base_url(monkeypatch, host, base):
    monkeypatch.setenv("OLLAMA_HOST", host)
    assert vision._ollama_base() == base


def test_live_analyzer_uses_ollama_only_with_host_and_model(monkeypatch):
    for k in ("VISION_PROVIDER", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OLLAMA_VISION_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "localhost:11434")
    assert vision.provider() is None  # no model chosen
    monkeypatch.setenv("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")
    assert vision.provider() == "ollama"


def test_visitors_cannot_point_the_site_at_ollama():
    from fastapi.testclient import TestClient

    from observatory.detect import web

    assert "ollama" not in vision.BYOK_PROVIDERS
    client = TestClient(web.app)
    assert 'value="ollama"' not in client.get("/?lang=en").text
    resp = client.post("/api/keys/check", data={"provider": "ollama", "api_key": "http://169.254.169.254/x"})
    assert resp.status_code == 400
    resp = client.post("/analyze", data={"url": "https://www.youtube.com/watch?v=x", "key_provider": "ollama",
                                         "api_key": "http://169.254.169.254/x"}, headers={"HX-Request": "true"})
    assert 'class="error"' in resp.text


def test_benchmark_notebook_is_current_valid_and_runs_ollama():
    import ast
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("make_bench_notebook", root / "scripts" / "make_bench_notebook.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    on_disk = json.loads((root / "notebooks" / "benchmark_on_kaggle.ipynb").read_text(encoding="utf-8"))
    assert on_disk == gen.build(), "notebook is stale: run python scripts/make_bench_notebook.py"
    code = [("".join(c["source"])) for c in on_disk["cells"] if c["cell_type"] == "code"]
    for cell in code:
        ast.parse(cell)  # plain Python, no shell magics
    src = "\n".join(code)
    assert '"--provider", "ollama"' in src and "OLLAMA_HOST" in src and "--variant" in src
    assert "CACHE_DATASET" in src and "benchmark_results" in src
