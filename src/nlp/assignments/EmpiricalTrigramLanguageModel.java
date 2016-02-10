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
				trigramCounter.incrementCount(prePreviousWord + " " + previousWord,
						word, 1.0);
				wordCounterUnNorm.incrementCount(word, 1.0);
				bigramCounterUnNorm.incrementCount(previousWord, word, 1.0);
				trigramCounterUnNorm.incrementCount(prePreviousWord + " " + previousWord,
						word, 1.0);
				prePreviousWord = previousWord;
				previousWord = word;
			}
		}
		wordCounter.incrementCount(UNKNOWN, 1.0);
		wordCounterUnNorm.incrementCount(UNKNOWN, 1.0);
		normalizeDistributions();
	}

	//@Override
	public List<String> generateSentence1() {
		final List<String> sentence = new ArrayList<String>();
		Object[] keys = trigramCounter.keySet().toArray();
		int rdmIndex = (int) (Math.random() * keys.length);
		String[] words = ((String)keys[rdmIndex]).split(" ");
		String prePreviousWord = words.length > 0? words[0] : "";
		String previousWord = words.length > 1? words[1] : "";
		String word = generateTrigramWord(prePreviousWord, previousWord);
		sentence.add(prePreviousWord);
		sentence.add(previousWord);
		while (!word.equals(STOP)) {

			sentence.add(word);
			word = generateTrigramWord(previousWord, word);
		}
		return sentence;
	}

	@Override
	public List<String> generateSentence() {
		final List<String> sentence = new ArrayList<String>();
		String prePreviousWord = START;
		String previousWord = START;
		String word = generateWord(prePreviousWord, previousWord);

		while (!word.equals(STOP)) {
			sentence.add(word);
			prePreviousWord = previousWord;
			previousWord = word;
			word = generateWord(prePreviousWord, previousWord);
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
					word, true);
			prePreviousWord = previousWord;
			previousWord = word;
		}
		return probability;
	}

	public double getTrigramProbability(String prePreviousWord,
			String previousWord, String word, Boolean withSmoothing) {
		final double trigramCount = trigramCounter
				.getCount(prePreviousWord + " " + previousWord, word);
		final double bigramCount = bigramCounter.getCount(previousWord, word);
		double unigramCount = wordCounter.getCount(word);
		if (unigramCount == 0) {
			System.out.println("UNKNOWN Word: " + word);
			unigramCount = wordCounter.getCount(UNKNOWN);
		}

		//Adding Witten-Bell Smoothing
		Counter<String> triCounter = trigramCounterUnNorm.getCounter(prePreviousWord + " " + previousWord);
		double estLambda1 = triCounter.size() != 0? 1 - (triCounter.size()/(triCounter.size() + triCounter.totalCount())): 0;

		Counter<String> biCounter = bigramCounterUnNorm.getCounter(previousWord);
		double estLambda2 = 1 - (biCounter.size()/(biCounter.size() + biCounter.totalCount()));
		estLambda2 *= (1-estLambda1);

		if(withSmoothing)
			return estLambda1 * trigramCount + estLambda2 * bigramCount + (1.0 - estLambda1 - estLambda2) * unigramCount;
		else
			return lambda1 * trigramCount + lambda2 * bigramCount + (1.0 - lambda1 - lambda2) * unigramCount;
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

	String generateWord(String prePreviousWord, String previousWord) {
		final double sample = Math.random();
		double sum = 0.0;
		for (final String word : wordCounter.keySet()) {
			sum += this.getTrigramProbability(prePreviousWord, previousWord, word, true);
			if (sum > sample) {
				return word;
			}
		}
		return UNKNOWN;
	}

	String generateWord1() {
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

	/**among list of words that follow the given word randomly choose next word**/
	String generateBigramWord(String key) {

		if(bigramCounter.containsKey(key)) {
			Counter<String> counter = bigramCounter.getCounter(key);
			if(counter.isEmpty()) {
				return generateWord1();
			}
			//randomly chooses instead of taking the argmax, otherwise generated sentences will be relatively the same
			Object[] keys = counter.keySet().toArray();
			int rdmIndex = (int) (Math.random() * keys.length);
			return (String) keys[rdmIndex];

		} else {
			return generateWord1();
		}
	}

	String generateTrigramWord(String prePreviousWord, String previousWord) {
		if(trigramCounter.containsKey(prePreviousWord + " " + previousWord)) {
			Counter<String> counter = trigramCounter.getCounter(prePreviousWord + " " + previousWord);
			if(counter.isEmpty()) {
				return generateBigramWord(previousWord);
			}
			//randomly chooses instead of taking the argmax, otherwise generated sentences will be relatively the same
			Object[] keys = counter.keySet().toArray();
			int rdmIndex = (int) (Math.random() * keys.length);
			return (String) keys[rdmIndex];

		} else { //fallback to bigram generation
			return generateBigramWord(previousWord);
		}
	}
}
