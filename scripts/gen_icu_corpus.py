#!/usr/bin/env python3
"""Generate the ICU patient language data: an utterance BANK (candidates the decoder can pick) and a CORPUS (text the
ICU language model is trained on). Deterministic slot grammar for guaranteed coverage + LLM-generated variety.

  python scripts/gen_icu_corpus.py                 # writes data/icu/bank.txt and data/icu/corpus.txt
  python scripts/gen_icu_corpus.py --no-llm        # grammar only (offline)

bank.txt uses the phrases.txt format ("## category" headers, "!" marks critical), so load_phrase_table() reads it.
"""
import os, re, sys, json, random, argparse, itertools, concurrent.futures as cf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "data", "icu")

# ----------------------------------------------------------------------------- slot grammar (coverage we can reason about)
G = {
 "urgent": [
   ("I can't breathe", 1), ("I am choking", 1), ("My chest hurts", 1), ("I think I am having a heart attack", 1), ("I need help right now", 1),
   ("Call the doctor now", 1), ("Something is wrong", 1), ("Something is very wrong", 1), ("I feel like I am going to pass out", 1), ("I am bleeding", 1),
   ("I can't feel my legs", 1), ("I can't move my arm", 1), ("The tube is hurting me", 1), ("I need suction now", 1), ("I am in a lot of pain", 1),
   ("Emergency", 1), ("Get the doctor", 1), ("Please hurry", 1), ("I can't swallow", 1), ("My throat is closing", 1), ("I am going to be sick", 1),
 ],
 "pain": [
   "I am in pain", "I am in {a lot of|some|terrible|constant} pain", "My {head|throat|back|stomach|neck|leg|arm|chest|shoulder|hip|knee|foot|hand|jaw|ear|eye|side|belly} hurts",
   "My {head|back|stomach|neck|legs|arms|feet|hands} {hurt|ache|are sore}", "It hurts {here|there|when I breathe|when I move|when I swallow|when I cough|all over|a lot}",
   "The pain is {getting worse|better now|the same|sharp|dull|burning|coming and going|unbearable}", "My pain is {a|an} {two|three|four|five|six|seven|eight|nine|ten} out of ten",
   "Rate my pain as {a|an} {two|three|four|five|six|seven|eight|nine|ten}", "I need {pain medication|something for the pain|pain relief|more pain medicine|my pain pills}",
   "The {medication|medicine|painkiller} is not working", "The {medication|medicine} {helped|is helping|made me sleepy|made me sick}", "I feel {numb|tingling|pins and needles|cramping}",
   "I have a {headache|stomach ache|sore throat|cramp|migraine}", "It {stings|burns|throbs|aches}", "Please don't touch {my arm|my leg|there|it}", "That hurts",
 ],
 "breathing": [
   "I am short of breath", "I need {oxygen|air|more oxygen}", "I need to cough", "Please suction {my mouth|my throat|me}", "The {mask|tube} is too {tight|loose}",
   "I need the {mask|tube|oxygen} {adjusted|checked|moved}", "I can breathe {better|easier} now", "My nose is blocked", "I feel like I am drowning", "Can you sit me up so I can breathe",
   "I can't get enough air", "My breathing is {worse|better|hard|fast}", "There is {mucus|phlegm|something} in my throat", "I need my inhaler", "The oxygen is {too high|too low|off}",
   "I can't clear my throat", "Something is stuck in my throat",
 ],
 "comfort": [
   "I am {cold|hot|freezing|too warm|sweating|itchy|uncomfortable|stiff|dizzy}", "I need {a blanket|another blanket|a pillow|another pillow|a fan|ice chips|lip balm|a tissue|a warm blanket}",
   "Take the {blanket|pillow|mask|socks} off", "Turn me {over|on my left side|on my right side|on my back}", "I want to {sit up|lie down|lie flat|turn over|stretch|move}",
   "{Raise|Lower} the {bed|head of the bed|foot of the bed|light|volume|temperature}", "Adjust my {pillow|bed|blanket|position|mask|tube|arm|leg|head}",
   "My {legs|arms|back|neck|feet} {are|is} uncomfortable", "My {arm|leg|hand|foot} is {stuck|asleep|numb|caught}", "Please {move|scratch|rub|lift|straighten} my {arm|leg|nose|back|head|foot|hand}",
   "My {eyes|lips|mouth|skin|throat} {are|is} dry", "Please {wet my lips|wipe my face|brush my teeth|wash my hands|clean my mouth|fix my hair|wipe my eyes}",
   "The {light|TV|room} is too {bright|loud|dark|hot|cold}", "Please turn {off|on|down|up} the {light|TV|music|fan|heat|air}", "It is too {loud|bright|hot|cold|noisy}",
   "Please {close|open} the {door|window|curtain|blinds}", "I want {music|the TV|quiet|silence|the news}", "I need my {glasses|hearing aid|phone|charger|dentures|watch|book}",
   "Where is my {phone|family|bag|glasses|wallet}", "I want my {phone|glasses|blanket|pillow}", "Can you {dim|turn off|turn on} the lights", "I am comfortable",
 ],
 "needs": [
   "I need {water|a drink|ice|juice|something to drink|my medication|my medicine|medication|a nurse|a doctor|help|to rest|to sleep|a break|the bedpan|a bath|to be changed|the toilet|a urinal|to pee|to go to the bathroom|suction|a tissue|my inhaler|to sit up|a minute}",
   "I am {thirsty|hungry|starving|full|nauseous|tired|sleepy|exhausted|constipated|sick}", "I {do not|don't} want to {eat|drink|sleep|move|be turned|be alone}",
   "I want {something to eat|food|to eat|to drink|breakfast|dinner|a snack}", "I feel {nauseous|sick|like throwing up|bloated|dizzy}", "I am going to {throw up|be sick|vomit}",
   "I need to {use the bathroom|pee|poop|go|be changed|throw up|cough|spit}", "I need my {medication|medicine|pills|insulin|inhaler|eye drops} {now|please|soon}",
   "Can I have {a tissue|some ice|my inhaler|water|a drink|something for the pain|a straw|my glasses|the remote|a blanket|a pillow}", "I wet the bed", "I had an accident",
   "When is my {medication|medicine|next dose|breakfast|lunch|dinner}", "I {did not|didn't} get my {medication|medicine|pills}", "I need to {rest|sleep|lie down|sit up}",
 ],
 "people": [
   "Call my {family|wife|husband|mother|father|mom|dad|daughter|son|sister|brother|partner|friend|doctor|nurse}", "Where is my {family|wife|husband|mother|father|daughter|son|nurse|doctor}",
   "Is my {family|wife|husband|daughter|son} {here|coming|okay}", "I want to see my {family|wife|husband|mother|father|kids|children|dog}", "I want to be alone",
   "Please {stay with me|hold my hand|don't leave|come back|sit with me}", "Who are you", "What is your name", "Is the doctor coming", "When is the doctor coming",
   "Please get the {nurse|doctor|chaplain|social worker|interpreter}", "I want to talk to {the chaplain|a doctor|my family|someone|the nurse}", "I want a translator",
   "Tell my {family|wife|husband|kids} {I love them|I am okay|to come|not to worry}", "Thank you for {helping me|being here|everything|your help}",
 ],
 "questions": [
   "Where am I", "What {day|time|month|year} is it", "What happened to me", "Why am I here", "How long have I been here", "When can I {go home|eat|drink|get up|leave|see my family}",
   "When will the tube come out", "What is this {medication|medicine|tube|machine|for}", "What are you doing", "Is it serious", "Am I getting {better|worse}", "Am I going to be okay",
   "What did the {test|scan|doctor} {show|say}", "Can I have {visitors|a shower|water|my phone}", "Can I {go outside|sit up|walk|eat|drink|shower}", "Did I have surgery", "What happened",
   "Do I have to stay", "Is my {heart|lung|leg|brain} okay", "Why can't I {talk|move|feel my legs|eat}", "How am I doing",
 ],
 "feelings": [
   "I am {scared|worried|anxious|tired|exhausted|sad|frustrated|confused|bored|fine|okay|better|worse|lonely|angry|upset|embarrassed|calm|happy|grateful}",
   "I feel {dizzy|sick|weak|better|much better|worse|awful|terrible|strange|faint|shaky|hot|cold|okay|fine|safe|scared|lost|foggy}", "I {cannot|can't} sleep", "I had a bad dream",
   "I am {not|so} {okay|fine|comfortable}", "I don't understand", "I am {afraid|scared} of {dying|the surgery|the tube|being alone}", "I want to go home", "I miss my {family|wife|husband|kids|dog}",
   "I feel {alone|helpless|trapped|stuck}", "It is hard", "I am trying",
 ],
 "answers": [
   "Yes", "No", "Maybe", "I don't know", "Please", "Thank you", "Thank you very much", "Sorry", "Okay", "Not now", "Later", "Wait", "Stop", "More", "Less", "Again", "Slow down",
   "That is {right|wrong|it|not it|correct|better|worse|enough}", "Good {morning|night|afternoon|evening}", "Hello", "Goodbye", "I love you", "Please wait a moment", "Say that again",
   "I did not understand", "Speak slower", "Yes please", "No thank you", "Not yet", "A little", "A lot", "Never mind", "Forget it", "That's fine", "That's enough", "Go ahead", "Leave it",
 ],
 "care": [
   "I need {a bath|my hair washed|to be shaved|my dressing changed|my sheets changed|my gown changed|a shave|lotion|my teeth brushed}", "The {bandage|dressing|tape|cuff|strap|brace} is too tight",
   "The {IV|line|needle|catheter|drain|tube|cuff|monitor} {hurts|is leaking|is bothering me|is pulling|fell out|is loose|is beeping|is uncomfortable|is stuck}",
   "This {needle|tube|line} hurts", "I want this {tube|line|catheter|mask} out", "Can you check the {machine|monitor|line|pump|IV|alarm|oxygen}", "Please {stop|turn off|check} the alarm",
   "The {alarm|machine|monitor} {keeps beeping|is beeping|went off}", "My {bandage|dressing|gown|sheet} is {wet|dirty|loose|bloody}", "I think the IV {came out|is leaking|is infiltrated}",
 ],
}
CRITICAL = {t for t, c in G["urgent"] if c}


def expand(tmpl):
    parts = re.split(r"(\{[^}]*\})", tmpl)
    opts = [[p] if not p.startswith("{") else p[1:-1].split("|") for p in parts]
    for combo in itertools.product(*opts):
        s = "".join(combo)
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"\ba an\b", "an", s)
        yield s


def grammar_bank():
    bank = {}
    for cat, items in G.items():
        for it in items:
            tmpl = it[0] if isinstance(it, tuple) else it
            for s in expand(tmpl):
                bank.setdefault(_key(s), (s, cat))
    return bank


def _key(s):
    return re.sub(r"[^a-z' ]", "", s.lower()).strip()


# ----------------------------------------------------------------------------- LLM variety
CATS = {"urgent": "life-threatening problems (breathing, chest, choking, bleeding, stroke signs)", "pain": "pain and discomfort, where it hurts, how bad",
        "breathing": "breathing, oxygen, suction, the tube or mask", "comfort": "position, temperature, blankets, light, noise, personal items",
        "needs": "water, food, toilet, medication, sleep, rest", "people": "family, staff, wanting company or privacy", "questions": "questions about their condition, time, place, plans",
        "feelings": "emotions and how they feel", "answers": "short replies and social phrases", "care": "IV lines, dressings, catheters, machines and alarms"}


def llm_batch(cat, n, seed, model):
    from dotenv import load_dotenv; load_dotenv(os.path.join(ROOT, ".env"))
    from openai import OpenAI
    c = OpenAI()
    sys_p = ("You write realistic things a voiceless ICU patient (tracheostomy, ventilator) would MOUTH to a nurse. First person, plain spoken English, "
             "2 to 9 words, no punctuation, no medical jargon beyond what patients say, no numbers as digits (write 'eight' not '8'). Each line a distinct "
             "utterance; vary wording (my meds / my medicine / my pills), include both requests and statements. Output a JSON object {\"lines\": [...]}.")
    user = f"Topic: {CATS[cat]}. Give {n} distinct utterances. Variation seed {seed}."
    r = c.chat.completions.create(model=model, temperature=1.0, messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                                  response_format={"type": "json_object"})
    try:
        lines = json.loads(r.choices[0].message.content).get("lines", [])
    except Exception:
        lines = []
    out = []
    for s in lines:
        if not isinstance(s, str): continue
        s = re.sub(r"[^A-Za-z' ]", "", s).strip()
        if 1 <= len(s.split()) <= 10 and not re.search(r"\d", s):
            out.append(s[0].upper() + s[1:])
    return out


def llm_conversation(n, seed, model):
    """Longer, multi-clause sentences for the language model only (open mode says these; the bank stays short)."""
    from dotenv import load_dotenv; load_dotenv(os.path.join(ROOT, ".env"))
    from openai import OpenAI
    c = OpenAI()
    sys_p = ("Write realistic sentences a hospital ICU patient says to a nurse or family member: needs, pain, feelings, questions, small talk about "
             "family, home, food, sleep, the room. First person, natural spoken English, 4 to 14 words, no punctuation, numbers as words. "
             "Output a JSON object {\"lines\": [...]} with distinct sentences.")
    r = c.chat.completions.create(model=model, temperature=1.0, messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": f"Give {n} sentences. Seed {seed}."}],
                                  response_format={"type": "json_object"})
    try:
        lines = json.loads(r.choices[0].message.content).get("lines", [])
    except Exception:
        lines = []
    return [re.sub(r"[^A-Za-z' ]", "", s).strip() for s in lines if isinstance(s, str) and 3 <= len(s.split()) <= 16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--per-cat", type=int, default=120, help="LLM utterances requested per category per round")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--conv", type=int, default=1500, help="longer LM-only sentences")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    bank = grammar_bank()
    print(f"grammar: {len(bank)} utterances")
    conv = []
    if not args.no_llm:
        jobs = [(cat, args.per_cat, r) for cat in CATS for r in range(args.rounds)]
        with cf.ThreadPoolExecutor(8) as ex:
            for (cat, _, r), lines in zip(jobs, ex.map(lambda j: llm_batch(j[0], j[1], j[2], args.model), jobs)):
                new = 0
                for s in lines:
                    k = _key(s)
                    if k and k not in bank:
                        bank[k] = (s, cat); new += 1
                print(f"  llm {cat} round {r}: {len(lines)} lines, {new} new")
            convjobs = list(range(max(1, args.conv // 100)))
            for lines in ex.map(lambda s: llm_conversation(100, s, args.model), convjobs):
                conv.extend(lines)
        print(f"llm conversation sentences: {len(conv)}")
    # curated phrases first (they keep their categories), then the rest
    cur = {}
    from silent_running.vsr import load_phrase_table
    for r in load_phrase_table():
        cur[_key(r["phrase"])] = (r["phrase"], r["category"], r["critical"])
    by_cat = {}
    for k, (s, cat) in bank.items():
        if k in cur:
            continue
        by_cat.setdefault(cat, []).append(s)
    with open(os.path.join(OUT, "bank.txt"), "w") as f:
        f.write("# ICU utterance bank: generated by scripts/gen_icu_corpus.py (slot grammar + LLM variety). Same format as phrases.txt.\n")
        f.write("# The curated phrases.txt entries are NOT repeated here; the server loads both.\n")
        for cat in CATS:
            f.write(f"\n## {cat}\n")
            for s in sorted(set(by_cat.get(cat, []))):
                f.write(s + (" !" if s in CRITICAL else "") + "\n")
    n_bank = sum(len(v) for v in by_cat.values())
    with open(os.path.join(OUT, "corpus.txt"), "w") as f:
        for k, (s, cat, crit) in cur.items():      # curated phrases weigh 3x
            for _ in range(3): f.write(s + "\n")
        for k, (s, cat) in bank.items():
            if k not in cur: f.write(s + "\n")
        for s in conv: f.write(s + "\n")
    print(f"wrote data/icu/bank.txt ({n_bank} extra utterances) and data/icu/corpus.txt ({len(cur)*3 + n_bank + len(conv)} lines)")


if __name__ == "__main__":
    main()
