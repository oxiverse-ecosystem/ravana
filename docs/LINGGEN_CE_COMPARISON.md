# LingGen CE comparison

N = 10

Method: per-path surface forms scored by the same decoder (read-only sampled-softmax NLL); coherence is a path-independent check

decoder-CE = 3.9278

template-CE = 3.9239

decoder-CE <= template-CE: False

decoder coherence = 0.678

template coherence = 0.7159

decoder not worse on coherence: False

decoder training examples = 0

uniform baseline CE (ln 51) = 3.9318

replies produced: decoder 10/10, template 10/10

Decision: use_linggen = False. Flag stays False. The decoder has never been trained (0 examples), so there is no distribution to fit -- the measurement is valid and the answer is genuinely no. Run scripts/train.py first.
