# Autonomous Language Learning Experiment

N_train=10 N_eval=10 pre=8/10 post=8/10 improvement=0.000

## Pre-training eval
- Q: what is my favorite color?
  R: i don't know your favorite color yet! what is it?
  hit=1
- Q: do i like reading?
  R: i'm curious about ing. it sits close to tracking, which i am against — so that p
  hit=0
- Q: where does the cat sleep?
  R: honestly, sleep is a bit outside what i know right now. i'd love to hear your ta
  hit=1
- Q: what happens to water at zero degrees?
  R: hmm, i'm not totally sure about degrees yet, and that's okay. what's your sense 
  hit=1
- Q: where do birds migrate?
  R: honestly, migrate is a bit outside what i know right now. what made you wonder a
  hit=1
- Q: what does she play?
  R: hmm, i'm not totally sure about that yet, and that's okay. what do you think abo
  hit=1
- Q: when does the library open?
  R: i don't have a clean line on library yet, but art keeps coming up nearby. the tw
  hit=1
- Q: where did we walk?
  R: i don't really have a solid grasp on that so far. i'd love to hear your take fir
  hit=1
- Q: what did he fix?
  R: fix is like everything — the same pattern shows up in both
  hit=0
- Q: what colors can apples be?
  R: i don't really have a solid grasp on colors so far. let's dig into it — what's y
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
  R: i don't really have a solid grasp on that so far.
[agentic:web_search] [web:inte
  hit=1
- Q: what did he fix?
  R: fix is like everything — the same pattern shows up in both
  hit=0
- Q: what colors can apples be?
  R: hmm, i'm not totally sure about colors yet, and that's okay.
[agentic:web_search
  hit=1
