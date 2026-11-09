# LingGen CE comparison

N = 10

Method: per-path surface forms scored by the same decoder (read-only sampled-softmax NLL); coherence is a path-independent check

decoder-CE = 3.8204

template-CE = 3.7527

decoder-CE <= template-CE: False

decoder coherence = 0.5727

template coherence = 0.5727

decoder not worse on coherence: True

decoder training examples = 1300160

uniform baseline CE (ln 51) = 3.9318

replies produced: decoder 6/10, template 6/10

Decision: use_linggen = False. Flag stays False: the decoder does not beat the template path on both signals.
