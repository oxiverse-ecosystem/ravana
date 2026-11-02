# Generalisation and Abstraction Experiment

N_train=5 N_eval=4 pre=4/4 post=4/4 improvement=0.000

## Pre-training eval
- Q: do i like the ocean?
  R: i'm drawn to ocean. it sits close to people, which i care about — so that pulls 
  hit=1
- Q: am i fond of the ocean?
  R: hmm, i'm not totally sure about ocean yet, and that's okay.
[agentic:web_search]
  hit=1
- Q: does the ocean please me?
  R: the closest i get to ocean is earth and water — and earth seems to be earth is t
  hit=1
- Q: is the ocean dear to me?
  R: honestly, ocean is a bit outside what i know right now.
[agentic:web_search] [we
  hit=1

## Post-training eval
- Q: do i like the ocean?
  R: i'm drawn to ocean. it sits close to people, which i care about — so that pulls 
  hit=1
- Q: am i fond of the ocean?
  R: i don't really have a solid grasp on ocean so far.
[agentic:web_search] [web] se
  hit=1
- Q: does the ocean please me?
  R: honestly, ocean is a bit outside what i know right now. i'd love to hear your ta
  hit=1
- Q: is the ocean dear to me?
  R: i don't really have a solid grasp on ocean so far.
[agentic:web_search] [web] se
  hit=1
