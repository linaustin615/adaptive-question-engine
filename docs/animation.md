# Animated information flow

[Open the architecture GIF](architecture_flow.gif) to follow data through the system. It loops every 47.5 seconds, with a moving dot showing the active data path and a caption explaining what happens. Pause-free reading is available in the outline below. The [still image](architecture_flow.png) shows the online adaptation loop.

This is a narrated architecture illustration, not a recording of live training. The top row and the local event/feedback routing illustrate runnable synthetic code. Dataset construction, model fitting, release approval, LLM generation, content review, and item calibration are proposed workflows. Their boxes and paths are labeled accordingly.

## Follow the three loops

| Steps | Data passed | How it is used |
| --- | --- | --- |
| 1–3 | Session state, candidate information, selected question | Select an eligible question and deliver it to the learner |
| 4–6 | Answer, correctness, ability posterior | Grade against the stored key, update ability uncertainty, then choose again if stopping rules permit |
| 7 | Responses, decisions, versions, selection probabilities | Record structured evidence in the local audit log |
| 8 | Typed feedback | Change a practice preference or request content review; assessment scores remain separate |
| 9–10 | Confirmed issue codes and approved source context | Proposed reviewers identify defects; a future generator uses that evidence in its prompt |
| 11–13 | Reviewed draft, unscored pilot responses, calibrated parameters | Proposed content checks and calibration gates precede a new scored item-bank version |
| 14–16 | Versioned interaction snapshot, trained artifact, held-out metrics | Proposed offline jobs train and evaluate learner or practice-policy models |
| 17 | Failed evaluation and reason codes | Reject a candidate and revise it; this is an alternative outcome to approval |
| 18–19 | Approved model artifact and release version | Proposed registry publishes an approved policy/model for future sessions, which collect fresh evidence |

An **ability update** estimates the current learner's ability with fixed item parameters. **Item calibration** learns those item parameters from response data. **Model training** fits a learner model or policy from a dataset. **Prompt feedback** changes generation context without necessarily changing model weights. The animation keeps these mechanisms distinct.

The release path represents a policy/model that meets its intended use: a future practice policy belongs in practice, and a future assessment policy needs measurement-specific validation. It does not mean that any trained model can replace the IRT scorer. Rejected candidates and approved candidates are alternative branches, not an automatic failed-then-approved sequence.

## Rebuild the GIF

From the project folder, with Pillow installed:

```bash
python3 scripts/animate_architecture.py
```

This produces `docs/architecture_flow.gif` and its PNG still. Edit `NODES` and `STEPS` in the script to change the diagram and narration. The renderer uses Python drawing primitives, portable font fallbacks, and no image-generation service. Pillow is needed only to regenerate these visuals; the assessment demo and benchmark still use the standard library.

Motion runs at 25 frames per second with eased starts and stops, rounded turns, and a short arrival pause. Each narrated step still lasts 2.5 seconds. The renderer streams changed rectangles with a shared palette to keep memory use and download size small.
