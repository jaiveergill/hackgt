# Next features, judged against what actually won recently (2026-09-25)

## What won, and the pattern

| Event | Winner | Why it won (per organizers / winners) |
|---|---|---|
| TreeHacks 2026 (Feb, $500k) | **Shepherd**: motorized smart cane for the visually impaired, computer vision steers the user | Judging: "creativity, technological complexity and social impact". Physical, assistive, obviously real. |
| Cal Hacks 12 (Nov 2025, 695 projects) | **FaceTimeOS**: voice-control your Mac through a FaceTime call | Winner's own post-mortem: pick an angle cooler than last year's winners, sit between niche and general, and "a really cool demo with great UI" beats backend depth; they even removed a feature to make the demo look cleaner. |
| HackGT 12 (Sep 2025, our venue) | Health and accessibility swept: Dose (IoT medication tracker), Caladrius (privacy-first AI triage), VisionNav (audio navigation for blind users), Vertex A11y, plus CV projects | GT judges reward health, accessibility, privacy framing, and things that work on stage. |
| HackMIT 2026 (Sep 19-20, 299 projects) | Named standouts: APO-TOPE (antibody binding model + UI), Abstract (research workspace) | Deep-tech plus a clear UI; three general prizes, four track prizes, "Most Technically Impressive" and "Most Creative" team awards. |
| Hack the North 2026 (Sep 18-20) | No ranked winners (finalists model); sponsor prizes dominated by agents and OpenAI API usage | Agentic framing and sponsor API usage are table stakes for sponsor prizes. |

Patterns across all of them: (1) assistive and health projects win the big prizes when they visibly work; (2) a body or a physical world in the loop (cane, FaceTime, IoT bottle) beats a pure web app; (3) judges score what they see in 3 minutes, so the demo is the product; (4) privacy and "runs locally" is a differentiator in health; (5) a scientific-looking evaluation number gets "most technically impressive".

## Where Silent Running already stands

Silent speech in, cloned expressive voice out, with the pipeline shown transparently. That is a rarer core than most winners had. The gaps are: proof it works on more than one person, a real-user story, a physical or world-facing loop, and one number judges can repeat.

## Feature ideas, ranked by (demo impact x feasibility before Sunday)

### Tier 1: do these

1. **Two-person live proof, with numbers on screen.** Record the phrase set on two or three teammates tonight (`record_session.py`), run the eval, and put a live "accuracy on N speakers, M phrases: top-1 X%, top-3 Y%" badge in the header. Judges at every event above rewarded a measured claim. Adaptation (Plan 2) then becomes the "it learned my mouth in ten minutes" moment with a before/after number. Half a day, mostly recording.

2. **Nurse-side conversation loop.** A "Nurse" button that listens to the laptop mic, transcribes what the caregiver said (OpenAI transcribe), and drops it into context, so the demo is a real two-way exchange: nurse asks "Are you cold?", patient mouths "Yes", laptop answers in the patient's voice, warm or annoyed depending on their face. Context-adaptive decoding stops being a text box and becomes the story. 2 hours, planned as Plan 1 D already.

3. **Emergency phrase escalation.** If the recognized phrase is in a critical set ("I can't breathe", "My chest hurts", "I need help") with confidence above threshold: full-screen red state, the laptop speaks it louder and twice, and it pushes a text or Slack message to a "nurse station" phone with the phrase, confidence and a 2-second clip of the raw transcript. Health judges at HackGT 12 rewarded exactly this kind of care-loop closure (Dose, Caladrius). 2 hours with Twilio or a Slack webhook.

4. **Phone as a second screen for the family.** A QR code on the bedside screen opens a mobile page that shows the conversation log in real time and can send prompts into context ("Mom, do you want us to call the doctor?"). The patient answers by mouthing. Judges can hold the phone. 2 hours, it is the same websocket.

### Tier 2: strong if time allows

5. **Personal phrase learning.** Every confirmed utterance adds to a per-speaker frequency prior; after a few uses the UI shows "learned: you say 'I need my glasses' a lot at night". Also lets the nurse add a custom phrase by typing it; it is scored by the same visual model immediately, no training. 1 hour, mostly wiring what exists.

6. **Embodiment: a small speaker or robot that turns to face the nurse.** A cheap USB speaker with an LED ring, or an Arduino servo that turns a face toward whoever the patient looks at (head pose from the same landmarks). Shepherd and FaceTimeOS both won partly because there was a physical thing on stage. Only if someone has hardware in a bag; 3 hours.

7. **Whisper-back privacy mode.** Everything except TTS runs offline; show a "no audio ever leaves the laptop, video never leaves the laptop" indicator, and an offline TTS fallback (macOS `say` or Chatterbox) with a visible switch. Caladrius won HackGT 12 on privacy framing. 1 hour for the indicator and switch.

8. **Multilingual output.** Patient mouths English, laptop speaks in the family's language in the patient's own voice (ElevenLabs v3 does cross-lingual with a clone). A one-line addition with big audience effect at a Georgia hospital demo (Spanish, Korean, Hindi). 1 hour.

### Tier 3: skip unless a track demands it

- Wearable or glasses camera (Chaplin-wearable exists as a fork): cool, but the mouth ROI quality from a chin-mounted camera is untested and the model was trained on frontal video. Risky.
- Training a new model, or replacing the VSR backbone: no.
- Bilateral video call integration (patient joining Zoom with synthesized voice): tempting given FaceTimeOS, but a browser virtual-camera or audio device on macOS is a rabbit hole in a weekend.

## Suggested Sunday demo script (3 minutes)

1. Nurse asks aloud "Are you cold?" (captured as context). Patient mouths "Yes" with a warm face: laptop answers "Yes" in the patient's voice, warm. 20 s.
2. Patient mouths "My chest hurts" with a strained face: red escalation, spoken twice, nurse-station phone buzzes. 30 s.
3. Switch to Open Mode: patient mouths a free sentence; show the n-best, the LLM proposal, the visual verification gap, and the word-timing strip. 40 s.
4. Show the accuracy badge and the adapted-vs-generic numbers for two speakers. 20 s.
5. Family phone: sibling types a question, patient answers. 20 s.
6. Close on the privacy indicator and the license slide. 10 s.

## Sources

Stanford Daily on TreeHacks 2026 (Shepherd, judging criteria), Dylan Lu's Cal Hacks 12 grand prize write-up (FaceTimeOS, judging insights), HackGT 12 Devpost project gallery (winners), HackMIT 2026 Plume gallery and prizes page (project standouts, prize structure), Hack the North 2026 Devpost (finalist model, sponsor prizes).
