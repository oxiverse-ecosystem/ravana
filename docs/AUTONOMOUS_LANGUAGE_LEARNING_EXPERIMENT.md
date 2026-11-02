# Autonomous Language Learning Experiment

N_train=10 N_eval=10 pre=9/10 post=8/10 improvement=-0.100

## Pre-training eval
- Q: what is my favorite color?
  R: your favorite color is teal.
  hit=1
- Q: do i like reading?
  R: yes — you told me: i like reading in the morning.
  hit=1
- Q: where does the cat sleep?
  R: hmm, i'm not totally sure about sleep yet, and that's okay.
[agentic:web_search]
  hit=1
- Q: what happens to water at zero degrees?
  R: hmm, i'm not totally sure about degrees yet, and that's okay.
[agentic:web_searc
  hit=1
- Q: where do birds migrate?
  R: honestly, migrate is a bit outside what i know right now.
[agentic:web_search] [
  hit=1
- Q: what does she play?
  R: i don't really have a solid grasp on that so far.
[agentic:web_search] [web:inte
  hit=1
- Q: when does the library open?
  R: library is like art — the same pattern shows up in both
  hit=1
- Q: where did we walk?
  R: hmm, i'm not totally sure about that yet, and that's okay.
[agentic:web_search] 
  hit=1
- Q: what did he fix?
  R: fix is like everything — the same pattern shows up in both
  hit=0
- Q: what colors can apples be?
  R: i don't really have a solid grasp on colors so far.
[agentic:web_search] [web:in
  hit=1

## Post-training eval
- Q: what is my favorite color?
  R: your favorite color is teal.
  hit=1
- Q: do i like reading?
  R: yes — you told me: i like reading in the morning.
  hit=1
- Q: where does the cat sleep?
  R: you mentioned: "the cat sleeps on the windowsill."
  hit=0
- Q: what happens to water at zero degrees?
  R: you mentioned: "water freezes at zero degrees."
  hit=1
- Q: where do birds migrate?
  R: you mentioned: "birds migrate south in winter."
  hit=1
- Q: what does she play?
  R: honestly, that is a bit outside what i know right now.
[agentic:web_search] [web
  hit=1
- Q: when does the library open?
  R: you mentioned: "the library opens at nine."
  hit=1
- Q: where did we walk?
  R: hmm, i'm not totally sure about that yet, and that's okay.
[agentic:web_search] 
  hit=1
- Q: what did he fix?
  R: fix is like everything — the same pattern shows up in both
  hit=0
- Q: what colors can apples be?
  R: honestly, colors is a bit outside what i know right now.
[agentic:web_search] [w
  hit=1
