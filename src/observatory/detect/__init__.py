"""Single-video analysis: paste a link, get a probability that it is
AI-generated, a verdict and the evidence behind it."""

from dotenv import load_dotenv

# Keys live in .env locally and in host secrets when deployed; real env wins.
load_dotenv(override=False)

from observatory.detect.pipeline import analyze  # noqa: E402

__all__ = ["analyze"]
