package nlp.assignments;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import nlp.langmodel.LanguageModel;
import nlp.util.Counter;
import nlp.util.CounterMap;

/**
 * Vanilla trigram language model.
 */
class EmpiricalTrigramLanguageModel implements LanguageModel {

	static final double			lambda1			= 0.5;
	static final double			lambda2			= 0.3;
	static final String			START			= "<S>";
	static final String			STOP			= "</S>";
	static final String			UNKNOWN			= "*UNKNOWN*";

	CounterMap<String, String>	bigramCounter	= new CounterMap<String, String>();
	CounterMap<String, String>	trigramCounter	= new CounterMap<String, String>();
	Counter<String>				wordCounter		= new Counter<String>();
	CounterMap<String, String>	bigramCounterUnNorm	= new CounterMap<String, String>();
	CounterMap<String, String>	trigramCounterUnNorm	= new CounterMap<String, String>();
	Counter<String>				wordCounterUnNorm		= new Counter<String>();

	public EmpiricalTrigramLanguageModel(
			Collection<List<String>> sentenceCollection) {
		for (final List<String> sentence : sentenceCollection) {
			final List<String> stoppedSentence = new ArrayList<String>(
					sentence);
			stoppedSentence.add(0, START);
			stoppedSentence.add(0, START);
			stoppedSentence.add(STOP);
			String prePreviousWord = stoppedSentence.get(0);
			String previousWord = stoppedSentence.get(1);
			for (int i = 2; i < stoppedSentence.size(); i++) {
				final String word = stoppedSentence.get(i);
				wordCounter.incrementCount(word, 1.0);
				bigramCounter.incrementCount(previousWord, word, 1.0);
				trigramCounter.incrementCount(prePreviousWord + previousWord,
						word, 1.0);
				wordCounterUnNorm.incrementCount(word, 1.0);
				bigramCounterUnNorm.incrementCount(previousWord, word, 1.0);
				trigramCounterUnNorm.incrementCount(prePreviousWord + previousWord,
						word, 1.0);
				prePreviousWord = previousWord;
				previousWord = word;
			}
		}
		wordCounter.incrementCount(UNKNOWN, 1.0);
		wordCounterUnNorm.incrementCount(UNKNOWN, 1.0);
		normalizeDistributions();
	}

	@Override
	public List<String> generateSentence() {
		System.out.println("WARNING -- DUMMY PLACEHOLDER IMPLEMENTATION");
		final List<String> sentence = new ArrayList<String>();
		String word = generateWord();
		while (!word.equals(STOP)) {
			sentence.add(word);
			word = generateWord();
		}
		return sentence;
	}

	@Override
	public double getSentenceProbability(List<String> sentence) {
		final List<String> stoppedSentence = new ArrayList<String>(sentence);
		stoppedSentence.add(0, START);
		stoppedSentence.add(0, START);
		stoppedSentence.add(STOP);
		double probability = 1.0;
		String prePreviousWord = stoppedSentence.get(0);
		String previousWord = stoppedSentence.get(1);
		for (int i = 2; i < stoppedSentence.size(); i++) {
			final String word = stoppedSentence.get(i);
			probability *= getTrigramProbability(prePreviousWord, previousWord,
					word);
			prePreviousWord = previousWord;
			previousWord = word;
		}
		return probability;
	}

	public double getTrigramProbability(String prePreviousWord,
			String previousWord, String word) {
		final double trigramCount = trigramCounter
				.getCount(prePreviousWord + previousWord, word);
		final double bigramCount = bigramCounter.getCount(previousWord, word);
		double unigramCount = wordCounter.getCount(word);
		if (unigramCount == 0) {
			System.out.println("UNKNOWN Word: " + word);
			unigramCount = wordCounter.getCount(UNKNOWN);
		}

		//Adding Witten-Bell Smoothing
		Counter<String> triCounter = trigramCounterUnNorm.getCounter(prePreviousWord + previousWord);
		double estLambda1 = triCounter.size() != 0? 1 - (triCounter.size()/(triCounter.size() + triCounter.totalCount())): 0;

		Counter<String> biCounter = bigramCounterUnNorm.getCounter(previousWord);
		double estLambda2 = 1 - (biCounter.size()/(biCounter.size() + biCounter.totalCount()));
		estLambda2 *= (1-estLambda1);

		return estLambda1 * trigramCount + estLambda2 * bigramCount
				+ (1.0 - estLambda1 - estLambda2) * unigramCount;

		//return lambda1 * trigramCount + lambda2 * bigramCount
		//		+ (1.0 - lambda1 - lambda2) * unigramCount;
	}

	private void normalizeDistributions() {
		for (final String previousBigram : trigramCounter.keySet()) {
			trigramCounter.getCounter(previousBigram).normalize();
		}
		for (final String previousWord : bigramCounter.keySet()) {
			bigramCounter.getCounter(previousWord).normalize();
		}
		wordCounter.normalize();
	}

	String generateWord() {
		final double sample = Math.random();
		double sum = 0.0;
		for (final String word : wordCounter.keySet()) {
			sum += wordCounter.getCount(word);
			if (sum > sample) {
				return word;
			}
		}
		return UNKNOWN;
	}
}
