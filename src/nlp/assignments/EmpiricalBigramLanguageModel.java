package nlp.assignments;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;

import nlp.langmodel.LanguageModel;
import nlp.util.Counter;
import nlp.util.CounterMap;

/**
 * Vanilla bi-gram language model.
 */
class EmpiricalBigramLanguageModel implements LanguageModel {

	static final double			lambda			= 0.6;
	static final String			START			= "<S>";
	static final String			STOP			= "</S>";
	static final String			UNKNOWN			= "*UNKNOWN*";

	CounterMap<String, String>	bigramCounter	= new CounterMap<String, String>();
	Counter<String>				wordCounter		= new Counter<String>();
	CounterMap<String, String>	bigramCounterUnNorm = new CounterMap<String, String>();
	Counter<String>				wordCounterUnNorm = new Counter<String>();

	public EmpiricalBigramLanguageModel(
			Collection<List<String>> sentenceCollection) {
		for (final List<String> sentence : sentenceCollection) {
			final List<String> stoppedSentence = new ArrayList<String>(
					sentence);
			stoppedSentence.add(0, START);
			stoppedSentence.add(STOP);
			String previousWord = stoppedSentence.get(0);
			for (int i = 1; i < stoppedSentence.size(); i++) {
				final String word = stoppedSentence.get(i);
				wordCounter.incrementCount(word, 1.0);
				bigramCounter.incrementCount(previousWord, word, 1.0);
				wordCounterUnNorm.incrementCount(word, 1.0);
				bigramCounterUnNorm.incrementCount(previousWord, word, 1.0);
				previousWord = word;
			}
		}
		wordCounter.incrementCount(UNKNOWN, 1.0);
		wordCounterUnNorm.incrementCount(UNKNOWN, 1.0);
		normalizeDistributions();
	}

	/*@Override
	public List<String> generateSentence() {
		final List<String> sentence = new ArrayList<String>();
		Object[] keys = bigramCounter.keySet().toArray();
		int rdmIndex = (int) (Math.random() * keys.length);
		String word = generateBigramWord((String) keys[rdmIndex]);
		while (!word.equals(STOP)) {
			sentence.add(word);
			word = generateBigramWord(word);
		}
		return sentence;
	}*/

	@Override
	public List<String> generateSentence() {
		final List<String> sentence = new ArrayList<String>();
		String word = generateWord(START);
		while (!word.equals(STOP)) {
			sentence.add(word);
			word = generateWord(word);
		}
		return sentence;
	}

	public double getBigramProbability(String previousWord, String word, Boolean withSmoothing) {
		final double bigramCount = bigramCounter.getCount(previousWord, word);
		double unigramCount = wordCounter.getCount(word);
		if (unigramCount == 0) {
			System.out.println("UNKNOWN Word: " + word);
			unigramCount = wordCounter.getCount(UNKNOWN);
		}

		//Add Witten-Bell Smoothing
		Counter<String> counter = bigramCounterUnNorm.getCounter(previousWord);
		double estLambda = counter.size() != 0? (1 - counter.size()/(counter.size() + counter.totalCount())) : 0;
		if(withSmoothing) {
			return estLambda * bigramCount + (1.0 - estLambda) * unigramCount;
		} else
			return lambda * bigramCount + (1.0 - lambda) * unigramCount;
	}

	@Override
	public double getSentenceProbability(List<String> sentence) {
		final List<String> stoppedSentence = new ArrayList<String>(sentence);
		stoppedSentence.add(0, START);
		stoppedSentence.add(STOP);
		double probability = 1.0;
		String previousWord = stoppedSentence.get(0);
		for (int i = 1; i < stoppedSentence.size(); i++) {
			final String word = stoppedSentence.get(i);
			probability *= getBigramProbability(previousWord, word, true);
			previousWord = word;
		}
		return probability;
	}

	private void normalizeDistributions() {
		for (final String previousWord : bigramCounter.keySet()) {
			bigramCounter.getCounter(previousWord).normalize();
		}
		wordCounter.normalize();
	}

	String generateWord(String previousWord) {
		final double sample = Math.random();
		double sum = 0.0;
		for (final String word : wordCounter.keySet()) {
			sum += this.getBigramProbability(previousWord, word, true);
			if (sum > sample) {
				return word;
			}
		}
		return UNKNOWN;
	}

	/*String generateWord() {
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

	*//**among list of words that follow the given word randomly choose next word**//*
	String generateBigramWord(String key) {

		if(bigramCounter.containsKey(key)) {
			Counter<String> counter = bigramCounter.getCounter(key);
			if(counter.isEmpty()) {
				return generateWord();
			}
			//randomly chooses instead of taking the argmax, otherwise generated sentences will be relatively the same
			Object[] keys = counter.keySet().toArray();
			int rdmIndex = (int) (Math.random() * keys.length);
			return (String) keys[rdmIndex];

		} else {
			return generateWord();
		}
	}*/
}
