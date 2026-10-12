"""Render a narrated architecture GIF; Pillow is only needed to rebuild it."""

import argparse
from dataclasses import dataclass
from math import hypot
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SIZE = (1440, 930)
BACKGROUND = "#f3f6fb"
INK = "#192c46"
MUTED = "#5a6b80"
BLUE, PURPLE, GREEN = "#3269bd", "#8651b5", "#128473"


def font(size, bold=False):
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


FONTS = {size: font(size) for size in (14, 16, 17, 19, 21)}
TITLE, HEADING, SMALL_BOLD = font(32, True), font(21, True), font(15, True)


@dataclass(frozen=True)
class Node:
    key: str
    column: int
    row: int
    title: str
    lines: tuple
    status: str

    @property
    def bounds(self):
        x, y = 40 + 280 * self.column, 270 + 200 * self.row
        return x, y, x + 240, y + 110

    def anchor(self, side):
        x1, y1, x2, y2 = self.bounds
        return {"left": (x1, (y1 + y2) / 2), "right": (x2, (y1 + y2) / 2),
                "top": ((x1 + x2) / 2, y1), "bottom": ((x1 + x2) / 2, y2)}[side]


NODES = [
    Node("learner", 0, 0, "Learner session", ("Request a question", "Submit an answer"), "SIMULATED"),
    Node("select", 1, 0, "Adaptive selector", ("Ability + item eligibility", "Fisher or information gain"), "RUNNABLE"),
    Node("bank", 2, 0, "Eligible item bank", ("Item + parameter versions", "Synthetic fixtures today"), "RUNNABLE"),
    Node("grade", 3, 0, "Grade the answer", ("Use the pinned answer key", "Record correctness"), "RUNNABLE"),
    Node("ability", 4, 0, "Update ability", ("IRT posterior + uncertainty", "Use for the next selection"), "RUNNABLE"),
    Node("events", 0, 1, "Local event log", ("Answers + decisions", "Versions + probabilities"), "RUNNABLE"),
    Node("data", 1, 1, "Dataset snapshot", ("Clean and version records", "Separate train / held-out"), "PROPOSED"),
    Node("train", 2, 1, "Offline training", ("Fit learner or policy model", "Save a candidate artifact"), "PROPOSED"),
    Node("evaluate", 3, 1, "Evaluate candidate", ("Baselines + accuracy", "Coverage + subgroup checks"), "PROPOSED"),
    Node("registry", 4, 1, "Approved release", ("Versioned model registry", "Rollback to prior version"), "PROPOSED"),
    Node("report", 0, 2, "Feedback routing", ("Content issue -> review", "Challenge -> preference"), "RUNNABLE"),
    Node("codes", 1, 2, "Confirm an issue", ("Reviewer checks evidence", "Approve an issue code"), "PROPOSED"),
    Node("generate", 2, 2, "Generate a draft", ("Sources + approved codes", "LLM adapter is a stub"), "PROPOSED"),
    Node("review", 3, 2, "Review and pilot", ("Verify content and key", "Collect unscored responses"), "PROPOSED"),
    Node("calibrate", 4, 2, "Calibrate and gate", ("Fit item parameters", "Check fit before release"), "PROPOSED"),
]
BY_KEY = {node.key: node for node in NODES}


@dataclass(frozen=True)
class Step:
    source: str
    target: str
    title: str
    explanation: tuple
    payload: str
    color: str
    proposed: bool = False
    route: str = "direct"


STEPS = [
    Step("learner", "select", "01  Start with the current learner state", ("A session pins its skill, item bank, policy version, and stopping rules.", "The selector reads the current ability estimate and its uncertainty."), "session + state", BLUE),
    Step("select", "bank", "02  Select from eligible items", ("Score eligible candidates by Fisher information or expected information gain.", "Optional family limits exclude related variants; exploration logs exact probabilities."), "item request", BLUE),
    Step("bank", "learner", "03  Deliver the selected question", ("The learner sees the question and choices. A future API must hide its answer key.", "The demo passes complete item objects locally; there is no network service yet."), "question", BLUE, route="online_return"),
    Step("learner", "grade", "04  Grade the learner's answer", ("Check that the answer matches the pending item, then use its pinned key.", "Reject duplicate submissions so the same response cannot count twice."), "answer", BLUE, route="online_return"),
    Step("grade", "ability", "05  Update the ability distribution", ("Correctness updates the Bayesian IRT posterior and its uncertainty.", "This is inference with fixed item parameters, not model training."), "correctness", BLUE),
    Step("ability", "select", "06  Adapt the next selection", ("The updated posterior changes which question would be useful next.", "Continue only while the session's precision and operational stopping rules allow it."), "new posterior", BLUE, route="online_return"),
    Step("grade", "events", "07  Record evidence for audit and future work", ("The local log records responses, item versions, decisions, and selection probabilities.", "Future production storage must add durable, transactional records and governed identifiers."), "audit event", BLUE, route="to_events"),
    Step("learner", "report", "08  Route explicit feedback separately", ("Too easy or too hard changes a practice preference; content complaints request review.", "Neither route rewrites the assessment score or a calibrated item difficulty."), "typed feedback", GREEN, route="feedback"),
    Step("report", "codes", "09  Confirm defects before using them", ("Proposed: a reviewer checks the report and records a confirmed issue code.", "Raw user text never becomes trusted instructions to the generator."), "review evidence", GREEN, True),
    Step("codes", "generate", "10  Use confirmed issues to guide generation", ("Proposed: approved sources, skill targets, and issue codes shape a versioned prompt.", "A model produces a draft; prompt feedback alone does not train its weights."), "approved codes", GREEN, True),
    Step("generate", "review", "11  Verify the draft before an unscored pilot", ("Proposed: check the key, explanation, ambiguity, duplicates, and required visuals.", "Reviewed items collect pilot responses without contributing to assessment scores."), "draft + sources", GREEN, True),
    Step("review", "calibrate", "12  Learn item parameters from pilot responses", ("Proposed: fit difficulty and discrimination on an anchored ability scale.", "Content approval alone is not calibration; parameter fit and uncertainty need evidence."), "pilot responses", GREEN, True),
    Step("calibrate", "bank", "13  Release an approved item-bank version", ("Proposed: publish immutable item and calibration versions only after their gates pass.", "New sessions can use the release; failed candidates stay out of the scored bank."), "approved items", GREEN, True, "bank_release"),
    Step("events", "data", "14  Build governed training data", ("Proposed: create a reproducible interaction snapshot with policy and item versions.", "Split by learner, time, and item family to reduce leakage into evaluation."), "data snapshot", PURPLE, True),
    Step("data", "train", "15  Train an offline candidate model", ("Proposed: fit a learner model or practice policy from the training split.", "Training code is an interface today; the running demo has no learned model weights."), "training split", PURPLE, True),
    Step("train", "evaluate", "16  Test whether the candidate earns a release", ("Proposed: compare held-out predictions with simple baselines and inspect uncertainty.", "A policy meant to improve learning also needs independent learning-outcome evidence."), "model + metrics", PURPLE, True),
    Step("evaluate", "train", "17  Failed gates send the candidate back", ("Proposed: reject a candidate with weak results, leakage, or unacceptable errors.", "Revise on training and validation data; repeated test-set tuning needs a fresh holdout."), "rejection reasons", PURPLE, True, "offline_return"),
    Step("evaluate", "registry", "18  Promote only a candidate that passes", ("Proposed: an approved release records data, code, model, and evaluation versions.", "Keep the previous approved artifact available for rollback."), "approved model", PURPLE, True),
    Step("registry", "select", "19  Close the loop with a versioned release", ("Proposed: new sessions pin an approved policy/model version and collect fresh evidence.", "The improvement loop succeeds only if later evaluation demonstrates a benefit."), "new policy version", PURPLE, True, "model_release"),
]


def route(step):
    source, target = BY_KEY[step.source], BY_KEY[step.target]
    if step.route == "online_return":
        a, b = source.anchor("bottom"), target.anchor("bottom")
        return [a, (a[0], 410), (b[0], 410), b]
    if step.route == "offline_return":
        a, b = source.anchor("bottom"), target.anchor("bottom")
        return [a, (a[0], 610), (b[0], 610), b]
    if step.route == "to_events":
        a, b = source.anchor("bottom"), target.anchor("top")
        return [a, (a[0], 410), (b[0], 410), b]
    if step.route == "feedback":
        a, b = source.anchor("left"), target.anchor("left")
        return [a, (18, a[1]), (18, b[1]), b]
    if step.route in {"bank_release", "model_release"}:
        a, b = source.anchor("right"), target.anchor("bottom")
        return [a, (1420, a[1]), (1420, 410), (b[0], 410), b]
    return [source.anchor("right"), target.anchor("left")]


def arrow(draw, points, color, width=3):
    draw.line(points, fill=color, width=width, joint="curve")
    a, b = points[-2:]
    length = hypot(b[0] - a[0], b[1] - a[1])
    dx, dy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
    draw.polygon([b, (b[0] - 10 * dx + 5 * dy, b[1] - 10 * dy - 5 * dx),
                  (b[0] - 10 * dx - 5 * dy, b[1] - 10 * dy + 5 * dx)], fill=color)


def point_on_path(points, progress):
    lengths = [hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])]
    distance = progress * sum(lengths)
    for a, b, length in zip(points, points[1:], lengths):
        if distance <= length:
            fraction = distance / length
            return a[0] + fraction * (b[0] - a[0]), a[1] + fraction * (b[1] - a[1])
        distance -= length
    return points[-1]


def render(step_index, progress):
    step = STEPS[step_index]
    image = Image.new("RGB", SIZE, BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((40, 28), "How information moves through the system", font=TITLE, fill=INK)
    draw.text((40, 73), "Online adaptation, reviewed content, and the proposed offline learning loop", font=FONTS[19], fill=MUTED)
    draw.rounded_rectangle((40, 112, 1400, 219), radius=12, fill="white", outline="#dbe3ef", width=2)
    draw.rectangle((40, 125, 45, 205), fill=step.color)
    draw.text((60, 128), step.title, font=HEADING, fill=INK)
    for index, line in enumerate(step.explanation):
        draw.text((60, 165 + index * 25), line, font=FONTS[17], fill=MUTED)
    label = "PROPOSED PATH" if step.proposed else "DEMO PATH"
    draw.text((1210, 131), label, font=SMALL_BOLD, fill=step.color)

    for row, label in enumerate(("ONLINE ASSESSMENT", "OFFLINE MODEL DEVELOPMENT", "FEEDBACK AND ITEM DEVELOPMENT")):
        draw.text((40, 243 + 200 * row), label, font=SMALL_BOLD, fill=MUTED)
        for column in range(4):
            arrow(draw, [(280 + 280 * column, 325 + 200 * row), (320 + 280 * column, 325 + 200 * row)], "#c7d1df", 2)

    points = route(step)
    arrow(draw, points, step.color, 4)
    for node in NODES:
        x1, y1, x2, y2 = node.bounds
        active = node.key in {step.source, step.target}
        draw.rounded_rectangle(node.bounds, radius=12, fill="#ffffff", outline=step.color if active else "#ccd6e4", width=3 if active else 2)
        draw.text((x1 + 14, y1 + 11), node.status, font=SMALL_BOLD, fill=step.color if active else MUTED)
        draw.text((x1 + 14, y1 + 35), node.title, font=HEADING, fill=INK)
        for index, line in enumerate(node.lines):
            draw.text((x1 + 14, y1 + 65 + 20 * index), line, font=FONTS[16], fill=MUTED)

    x, y = point_on_path(points, progress)
    draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=step.color, outline="white", width=2)
    draw.rounded_rectangle((40, 810, 1400, 863), radius=10, fill="white")
    draw.text((57, 827), "MOVING DATA", font=SMALL_BOLD, fill=MUTED)
    draw.text((200, 825), step.payload, font=FONTS[19], fill=step.color)
    draw.text((660, 827), "RUNNABLE = synthetic code today    |    PROPOSED = future implementation", font=FONTS[17], fill=MUTED)
    draw.text((40, 887), "An ability update is not training. New models and item parameters need evidence before release.", font=FONTS[17], fill=MUTED)
    draw.text((1285, 887), f"{step_index + 1:02d} / {len(STEPS):02d}", font=FONTS[17], fill=MUTED)
    return image


def main(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    samples = Image.new("RGB", (SIZE[0] * 3, SIZE[1]))
    for column, step in enumerate((0, 8, 14)):
        samples.paste(render(step, 0.5), (column * SIZE[0], 0))
    palette = samples.convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
    frames, durations = [], []
    for index in range(len(STEPS)):
        for frame in range(10):
            frames.append(render(index, frame / 9).quantize(palette=palette, dither=Image.Dither.NONE))
            durations.append(120 if frame < 9 else 1420)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True, disposal=1)
    render(5, 0.5).save(output.with_suffix(".png"))
    print(f"{output}: {len(frames)} frames, {sum(durations) / 1000:.1f}s, {output.stat().st_size / 1024 / 1024:.2f} MiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "architecture_flow.gif")
    main(parser.parse_args().output)
