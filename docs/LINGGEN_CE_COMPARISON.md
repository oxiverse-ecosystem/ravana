# LingGen CE comparison

N = 10

Method: per-path surface forms scored by the same decoder (read-only sampled-softmax NLL); coherence is a path-independent check

decoder-CE = 3.0897

template-CE = 2.4257

decoder-CE <= template-CE: False

decoder coherence = 0.5416

template coherence = 0.5727

decoder not worse on coherence: False

decoder training examples = 18872114

uniform baseline CE (ln 51) = 3.9318

replies produced: decoder 6/10, template 6/10

Decision: use_linggen = False. Flag stays False: the decoder does not beat the template path on both signals.
