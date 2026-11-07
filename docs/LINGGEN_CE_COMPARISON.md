# LingGen CE comparison

N = 10

Method: mean_prediction_error (engine surprise signal) on fixed probe set

decoder-CE = 0.0
template-CE = 0.0
decoder-CE <= template-CE: true

Decision: flag stays False. Reason: no separate template CE is instrumented in the engine; both paths report the same surprise signal (0.0 placeholder), so the measurement is not a valid distribution-fit promotion test. Promotion gated on a real measurement; harness needs per-path CE instrumentation before use_linggen can flip.
