# Winning the Impiricus track at HackGT 13 (2026-09-25)

## The challenge, as posted

- HackGT 13 Impiricus challenge: **"Build the next HCP Engagement Tool"**, described as an HCP engagement tool they do not offer today. Prizes $3,000 / $2,000 / $1,000 cash. Judged on four criteria: **impact to the HCP, originality, technical execution, commercial fit**, with a working demo expected.
- HackGT 12 (last year) Impiricus challenge text: technology-enabled healthcare-provider engagement that delivers relevant medical information and resources. Their own examples: identifying relevant literature, notifying providers of new treatments, simplifying insurance interactions, creating patient questionnaires based on chief complaints. Prizes were $3,000 + final-round interview, $1,500 + first-round interview, $500.

## Who Impiricus is (what the judges care about)

- "The first and only AI-powered HCP Engagement Engine", founded by a practicing physician and a pharma executive. Product **Ascend** is an agentic commercialization platform: real-time physician engagement over SMS, chatbots and media, NPI-level targeting, a council of 2,000+ HCP advisors, integrations for compliant sampling (QPharma) and patient support (Impiricus Wallet). Deloitte Fast 500 fastest-growing company in North America 2025; hiring about 60 people.
- Their stated ethos: "the future of healthcare commercialization should be built with physicians, not just for them"; every interaction must be "clinically meaningful, ethically grounded", compliant, and lead to better patient care.
- Translation: the judge in the room is a physician-minded pharma-tech operator. They score whether a **clinician** gets something they want, whether it could be a product line, and whether it was really built.

## What won last year

The HCP-engagement winners in the HackGT 12 gallery (health projects tagged as winners whose pitch is provider-facing):

- **Vital**: agentic workflow that links new guidelines, drug approvals and policy changes to actionable insights on the provider's *current* patients, "empowering providers to act faster with confidence".
- **Medicus**: connects medtech companies with healthcare providers, "the right information to the right provider at the right time".
- **Doc McQuery**: "smarter searches, better care", medical information retrieval for clinicians.

Common thread: every winner put the **provider** at the center, delivered *information* to them at the moment of a decision, and used an agentic pipeline. None was patient-facing. (Caladrius, the privacy-first triage assistant, won a general track, not Impiricus.)

## Where Silent Running stands against the four criteria

| Criterion | Today | Gap |
|---|---|---|
| Impact to the HCP | Strong story, but the product faces the patient. The nurse's benefit is implicit. | Make the clinician the user of a screen. Quantify time saved per interaction and per shift. |
| Originality | Very high. Nobody else is doing silent speech + cloned voice + context in an ICU. | Keep it; make sure the demo shows it working, not slides. |
| Technical execution | Deep: pretrained VSR, verified LLM correction, expressive TTS, live measurements. | Show the Dev tab for 20 seconds, then get back to the clinician. |
| Commercial fit | Weakest for *Impiricus*. Their business is pharma-to-HCP. An ICU communicator is hospital-facing assistive tech. | Add the layer where the patient's words trigger provider-facing information and a documented action, and articulate who pays. |

## Recommended moves, ranked by points-per-hour

### 1. Nurse Station view (the HCP is the user)  — biggest lift on "impact to the HCP"
A third tab, or a second browser window, that a nurse would keep open: a queue of patient needs across beds (one bed today), each with the phrase, confidence, time, whether it was acknowledged, and the critical alerts on top. Add a per-shift counter: interactions, median time from mouthing to spoken phrase, critical alerts, and "guesses avoided" (candidates the nurse would otherwise have cycled through on a letter board). This is the slide judges write down. 2 hours; the log already exists.

### 2. Clinical information at the moment of need  — this is the Impiricus shape
When a recognized phrase maps to a symptom or medication concern ("The medication is not working", "I feel nauseous", "It hurts when I breathe"), the nurse view shows a card: the relevant label facts for the patient's listed medications (openFDA drug label API is free and needs no key), the matching guideline snippet, and two suggested follow-up questions the nurse can ask with the nurse button. Cite the source on the card. That is literally last year's brief ("identify relevant literature, notify providers of new treatments") and Vital's winning pattern, attached to a live patient signal nobody else has. 3 hours with a small mapping table from phrases to symptom terms, an OpenAI summary constrained to the fetched label text, and a fake med list per demo patient.

### 3. Documentation export  — commercial fit and "action"
One button that turns the session log into a structured note (timestamps, patient statements, confidence, nurse responses, escalations) as both a readable nursing note and a FHIR `Communication` resource JSON. Nurses spend a large share of the shift documenting; a communicator that writes its own note is a product, not a feature. 1.5 hours.

### 4. The commercial story, stated plainly on a slide and in the README
- Buyer: hospital ICU / respiratory units; roughly a million patients are mechanically ventilated in US ICUs each year and communication failure is among their most reported stressors.
- Channel that Impiricus would recognize: patient-support and adherence programs funded by pharma (their Wallet product), where a patient's own reported symptoms and medication concerns become compliant, consented signals to the treating HCP.
- Price shape: per-bed software license; zero hardware beyond the laptop or a tablet.
- Compliance: video and audio never leave the machine except the text sent to TTS; no PHI in the LLM prompt beyond the phrase and typed context; consent captured before voice cloning.

### 5. Demo choreography for the Impiricus judge (3 minutes)
1. Open on the Nurse Station: two open needs, one critical. "This is what the nurse sees." 15 s.
2. Switch to the bed: nurse asks "Are you cold?" on the mic, patient mouths "Yes", laptop answers in the patient's voice. 25 s.
3. Patient mouths "The medication is not working": the nurse view shows the medication card with the label facts and two follow-up questions. Nurse taps one, patient answers. 45 s.
4. Patient mouths "I can't breathe": red escalation, spoken twice, appears at the top of the station queue. 20 s.
5. Export the note; show the FHIR JSON for three seconds. 15 s.
6. Dev tab for 20 seconds: raw transcript, verified LLM proposal, timing strips, latency line. Then the numbers: N speakers, top-1 accuracy, median latency.
7. Close with commercial fit in two sentences.

### What not to do for this track
- Do not lead with the cloned emotional voice. It is the most impressive part for a general audience and the least relevant to "HCP engagement". Show it once, in step 2, and move on.
- Do not present it as an AAC device. Present it as a provider tool that happens to have a patient interface.
- Do not claim clinical validation or nurse interviews that did not happen. Say "designed against the ICU communication literature and the phrase sets used on bedside boards"; judges from a physician-founded company will probe.

## Sources
HackGT 13 Devpost (Impiricus: "Build the next HCP Engagement Tool", prizes); HackGT 13 sponsor summary (four judging criteria, working demo); HackGT 12 Devpost sponsor challenges (last year's Impiricus brief and prizes); HackGT 12 project gallery (Vital, Medicus, Doc McQuery, Caladrius); Impiricus company pages, LinkedIn, Yahoo Finance on Ascend, PR Newswire on Deloitte Fast 500 and Impiricus Wallet.
