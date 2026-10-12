"""Draw the architecture as a portable SVG using only Python's standard library."""

from html import escape
from pathlib import Path


def draw():
    parts = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="1080" viewBox="0 0 1440 1080" role="img" aria-labelledby="title desc">
<title id="title">Adaptive assessment architecture</title>
<desc id="desc">Three connected flows: generate and review questions, select and score assessment items, and use feedback for practice and offline model development. A shared audit layer records decisions.</desc>
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="none" stroke="#64748b" stroke-width="1.4"/></marker></defs>
<rect width="1440" height="1080" rx="20" fill="#f4f7fb"/>
<style>text{font-family:Arial,Helvetica,sans-serif;fill:#14263d}.title{font-size:32px;font-weight:700}.sub{font-size:17px;fill:#526276}.lane{font-size:17px;font-weight:700;letter-spacing:1px}.head{font-size:20px;font-weight:700}.body{font-size:16px}.note{font-size:14px;fill:#526276}</style>''']

    def text(x, y, content, css="body"):
        parts.append(f'<text x="{x}" y="{y}" class="{css}">{escape(content)}</text>')

    def box(x, y, title, lines, color):
        parts.append(f'<rect x="{x}" y="{y}" width="380" height="112" rx="12" fill="white" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<rect x="{x}" y="{y + 14}" width="5" height="82" rx="2" fill="{color}"/>')
        text(x + 20, y + 31, title, "head")
        for index, line in enumerate(lines):
            text(x + 20, y + 59 + index * 22, line)

    def arrow(x1, y1, x2, y2, dashed=False):
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"{dash}/>')

    text(50, 61, "Adaptive assessment architecture", "title")
    text(50, 94, "Python blueprint · synthetic executable core · proposed production extensions", "sub")
    columns = [50, 530, 1010]
    colors = ["#168879", "#3766bf", "#9460b8"]
    labels = ["01  BUILD THE ITEM BANK", "02  MEASURE ABILITY", "03  PERSONALIZE PRACTICE"]
    for x, color, label in zip(columns, colors, labels):
        parts.append(f'<rect x="{x - 12}" y="124" width="404" height="690" rx="16" fill="{color}" opacity="0.05"/>')
        text(x, 156, label, "lane")

    box(50, 180, "Generate a draft", ["Skill + sources + intended difficulty", "Template now; LLM adapter later"], colors[0])
    box(50, 346, "Validate and review", ["Format, key, ambiguity, duplicates", "Reviewer approval before pilot use"], colors[0])
    box(50, 512, "Pilot and calibrate", ["Unscored human response collection", "Fit parameters and validate the scale"], colors[0])
    box(50, 678, "Release a versioned bank", ["Approved item + calibration version", "Drafts remain outside scored delivery"], colors[0])

    box(530, 180, "Open an assessment", ["Pin skill, bank, policy, and stop rules", "Start from an ability prior"], colors[1])
    box(530, 346, "Choose the next item", ["Fisher baseline or information gain", "Optional family cap and exploration"], colors[1])
    box(530, 512, "Grade and update ability", ["Stored answer key → 2PL posterior", "Ability + uncertainty + interval"], colors[1])
    box(530, 678, "Stop or select again", ["Precision, length, bank, or family limit", "Report why the session stopped"], colors[1])

    box(1010, 180, "Receive learner feedback", ["Correctness, challenge preferences", "Content reports use a review path"], colors[2])
    box(1010, 346, "Update practice state", ["Weak skills + requested challenge", "Feedback does not rewrite exam scores"], colors[2])
    box(1010, 512, "Evaluate measurement", ["Paired simulation and policy ablations", "Correlation and calibration stress tests"], colors[2])
    box(1010, 678, "Evaluate and release models", ["Baseline, quality, and subgroup checks", "Approved registry version + rollback"], colors[2])

    for x in columns:
        for y in (292, 458, 624):
            arrow(x + 190, y + 4, x + 190, y + 48)

    parts.append('<path d="M430,734 L477,734 L477,402 L524,402" fill="none" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>')
    parts.append('<path d="M910,734 L952,734 L952,402 L916,402" fill="none" stroke="#64748b" stroke-width="2" stroke-dasharray="7 5" marker-end="url(#arrow)"/>')
    text(443, 775, "bank", "note")
    text(929, 775, "loop", "note")
    for x in (240, 720, 1200):
        arrow(x, 794, x, 853, dashed=True)

    parts.append('<rect x="50" y="860" width="1340" height="98" rx="12" fill="#e2eaf5"/>')
    text(74, 893, "Shared audit and provenance", "head")
    text(74, 924, "Record versions, candidate utilities, selection probabilities, outcomes, and review actions")
    text(50, 999, "Feedback loop: confirmed content issues → versioned generation guidance; approved model releases → serving policies", "sub")
    text(50, 1030, "Implemented: synthetic scoring, policy experiments, feedback routing, template draft, and local events. Training remains a future boundary.", "note")
    parts.append("</svg>")
    target = Path(__file__).resolve().parents[1] / "docs" / "architecture.svg"
    target.write_text("\n".join(parts), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    draw()
