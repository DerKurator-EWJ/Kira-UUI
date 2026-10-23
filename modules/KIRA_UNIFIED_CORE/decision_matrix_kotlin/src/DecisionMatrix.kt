package kira.core.evaluation

import kotlin.math.abs

enum class Reaction(val utterance: String) {
    OK("OK."), MMH("Mmh."), GEIL("Geil!"), HAEH("Häh?!"), WAS("Was?!"),
    NOTGEIL("Notgeil!"), FUCK("Fuck!"), UUH("Uuh!"), BORING("Boring."),
    SURPRISE("Surprise, Motherfucker!"), GUILTY("Guilty."), EGAL("Egal."), CANTOR("Cantor.")
}

/** Values are normalized to [0,1], except valence [-1,1].
 * No object-specific rules, no automatic memory writes, no external actions.
 * surprise is measured prediction error, not a synonym for curiosity.
 */
data class EvaluationInput(
    val expectationConfidence: Double,
    val surprise: Double,
    val success: Double,
    val novelty: Double,
    val relevance: Double,
    val informationGain: Double,
    val valence: Double,
    val repetition: Double = 0.0,
    val contradictionScope: Double = 0.0,
    val suspiciousRecurrence: Double = 0.0,
    val uncertainty: Double = 0.0
) {
    init {
        listOf(expectationConfidence, surprise, success, novelty, relevance,
            informationGain, repetition, contradictionScope, suspiciousRecurrence, uncertainty)
            .forEach { require(it.isFinite() && it in 0.0..1.0) }
        require(valence.isFinite() && valence in -1.0..1.0)
    }
}

data class Evaluation(
    val primary: Reaction,
    val signals: Set<Reaction>,
    val investigatePriority: Double,
    val repeatPriority: Double,
    val evidenceReviewRequired: Boolean
)

/** Deterministic heuristic baseline; thresholds must be calibrated on actual Kira traces. */
class DecisionMatrix {
    fun evaluate(x: EvaluationInput): Evaluation {
        val surprise = x.surprise >= 0.65
        val highConfidenceContradiction = surprise && x.expectationConfidence >= 0.75
        val good = x.valence >= 0.55 && x.success >= 0.55
        val bad = x.valence <= -0.55 || (x.success <= 0.25 && x.relevance >= 0.6)
        val signals = linkedSetOf<Reaction>()
        if (x.relevance <= 0.15 && x.informationGain <= 0.15) signals += Reaction.EGAL
        if (x.repetition >= 0.8 && x.informationGain <= 0.15) signals += Reaction.BORING
        if (x.informationGain >= 0.7 && x.uncertainty <= 0.35) signals += Reaction.UUH
        if (x.novelty >= 0.75 && x.informationGain >= 0.3 && surprise) signals += Reaction.SURPRISE
        if (x.suspiciousRecurrence >= 0.8 && highConfidenceContradiction) signals += Reaction.GUILTY
        if (surprise && x.contradictionScope >= 0.9 && x.expectationConfidence >= 0.9) signals += Reaction.CANTOR
        if (surprise && x.contradictionScope >= 0.7) signals += Reaction.WAS
        if (surprise) signals += Reaction.HAEH
        if (good) signals += Reaction.GEIL
        if (bad) signals += Reaction.FUCK
        if (good && surprise) signals += Reaction.NOTGEIL
        if (bad && surprise) signals += Reaction.FUCK
        if (x.uncertainty >= 0.55 || (x.novelty >= 0.6 && !good && !bad)) signals += Reaction.MMH
        if (signals.isEmpty()) signals += Reaction.OK

        // Priority is a presentation policy; preserve all signals for memory and diagnostics.
        val precedence = listOf(Reaction.CANTOR, Reaction.NOTGEIL, Reaction.FUCK,
            Reaction.WAS, Reaction.GUILTY, Reaction.SURPRISE, Reaction.HAEH,
            Reaction.UUH, Reaction.GEIL, Reaction.MMH, Reaction.BORING,
            Reaction.EGAL, Reaction.OK)
        val primary = precedence.first { it in signals }
        val investigate = (0.4*x.surprise*x.expectationConfidence +
            0.3*x.informationGain + 0.2*x.novelty + 0.1*x.relevance).coerceIn(0.0, 1.0)
        val repeat = (0.55*x.success + 0.25*(x.valence.coerceAtLeast(0.0)) +
            0.2*x.informationGain - 0.4*x.repetition).coerceIn(0.0, 1.0)
        return Evaluation(primary, signals, investigate, repeat, highConfidenceContradiction)
    }
}
