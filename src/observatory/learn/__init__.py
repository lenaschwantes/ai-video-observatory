"""Learning from use, with a human in the loop.

1. Visitors say whether a video is AI (feedback.py). A vote is a hint, never
   ground truth: anyone can lie, so votes only rank the curation queue.
2. A curator confirms labels with a source (feedback.py, `observatory learn label`).
3. Signal weights are refitted on confirmed labels (fit.py) and a new version
   only replaces the current one if it does not accuse more real videos
   (champion/challenger). Every attempt, adopted or not, goes to the history.
"""
