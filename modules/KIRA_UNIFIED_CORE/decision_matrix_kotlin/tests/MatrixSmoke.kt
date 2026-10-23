import kira.core.evaluation.*
fun main() {
    val m = DecisionMatrix()
    fun input(surprise: Double, valence: Double, success: Double, scope: Double = 0.0) = EvaluationInput(
        expectationConfidence=0.95, surprise=surprise, success=success, novelty=0.8,
        relevance=0.8, informationGain=0.5, valence=valence, contradictionScope=scope)
    check(m.evaluate(input(0.1,0.7,0.9)).primary == Reaction.GEIL)
    check(m.evaluate(input(0.9,0.8,0.9)).primary == Reaction.NOTGEIL)
    check(m.evaluate(input(0.9,-0.8,0.0)).primary == Reaction.FUCK)
    check(m.evaluate(input(0.95,0.0,0.5,1.0)).primary == Reaction.CANTOR)
    check(m.evaluate(input(0.9,0.8,0.9)).evidenceReviewRequired)
    println("4 smoke tests passed")
}
