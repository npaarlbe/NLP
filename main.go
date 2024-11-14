package main

import (
	"fmt"
	"github.com/cdipaolo/sentiment"
	"github.com/kljensen/snowball"
	"strings"
	"unicode"
)

func main() {
	// Initialize the sentiment model
	model, err := sentiment.Restore()
	if err != nil {
		panic(fmt.Sprintf("Could not restore sentiment model: %v", err))
	}

	// Sample text for processing
	text := "The text is very pretty and awful. NLP in Go is amazing and perfect!"

	// Tokenization
	tokens := tokenize(text)
	fmt.Println("Tokens:")
	for _, token := range tokens {
		fmt.Printf("- %s\n", token)
	}

	// Stopword removal
	filteredTokens := removeStopwords(tokens)
	fmt.Println("\nTokens after stopword removal:")
	for _, token := range filteredTokens {
		fmt.Printf("- %s\n", token)
	}

	// Stemming
	stems := stemTokens(filteredTokens)
	fmt.Println("\nStems:")
	for i, stem := range stems {
		fmt.Printf("- %s: %s\n", filteredTokens[i], stem)
	}

	// Sentence Segmentation
	sentences := segmentSentences(text)
	fmt.Println("\nSentences:")
	for _, sentence := range sentences {
		fmt.Printf("- %s\n", sentence)
	}

	// Sentiment Analysis
	for _, sentence := range sentences {
		sentiment, confidence := analyzeSentiment(sentence, model)
		fmt.Printf("\nSentiment Analysis for: %s\n", sentence)
		fmt.Printf("Sentiment: %s\n", sentiment)
		fmt.Printf("Confidence: %.2f\n", confidence)
	}

	// Basic Named Entity Recognition
	entities := extractEntities(tokens)
	fmt.Println("\nNamed Entities:")
	for _, entity := range entities {
		fmt.Printf("- %s\n", entity)
	}

	// Word Frequency
	freq := wordFrequency(filteredTokens)
	fmt.Println("\nWord Frequency:")
	for word, count := range freq {
		fmt.Printf("- %s: %d\n", word, count)
	}

	// Word Associations
	associations := buildWordAssociations(tokens)
	fmt.Println("\nWord Associations:")
	for word, associated := range associations {
		fmt.Printf("- %s: %v\n", word, associated)
	}
}

func tokenize(text string) []string {
	return strings.FieldsFunc(text, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
}

func removeStopwords(tokens []string) []string {
	stopwords := map[string]bool{
		"a": true, "an": true, "and": true, "are": true, "as": true, "at": true, "be": true,
		"by": true, "for": true, "from": true, "has": true, "he": true, "in": true,
		"is": true, "it": true, "its": true, "of": true, "on": true, "that": true,
		"the": true, "to": true, "was": true, "were": true, "will": true, "with": true,
	}

	filtered := make([]string, 0)
	for _, token := range tokens {
		if !stopwords[strings.ToLower(token)] {
			filtered = append(filtered, token)
		}
	}
	return filtered
}

func stemTokens(tokens []string) []string {
	stems := make([]string, len(tokens))
	for i, token := range tokens {
		stem, err := snowball.Stem(strings.ToLower(token), "english", true)
		if err != nil {
			stems[i] = token
		} else {
			stems[i] = stem
		}
	}
	return stems
}

func segmentSentences(text string) []string {
	return strings.FieldsFunc(text, func(r rune) bool {
		return r == '.' || r == '!' || r == '?'
	})
}

func extractEntities(tokens []string) []string {
	entities := make([]string, 0)
	for _, token := range tokens {
		if unicode.IsUpper(rune(token[0])) && len(token) > 1 {
			entities = append(entities, token)
		}
	}
	return entities
}

func wordFrequency(tokens []string) map[string]int {
	freq := make(map[string]int)
	for _, token := range tokens {
		freq[strings.ToLower(token)]++
	}
	return freq
}

func buildWordAssociations(tokens []string) map[string][]string {
	associations := make(map[string][]string)
	for i, token := range tokens {
		if _, exists := associations[token]; !exists {
			associations[token] = []string{}
		}
		// Add a word associated with the token (the next token)
		if len(associations[token]) == 0 && i+1 < len(tokens) {
			associations[token] = append(associations[token], tokens[i+1])
		}
	}
	return associations
}
func analyzeSentiment(sentence string, model sentiment.Models) (string, float64) {
	analysis := model.SentimentAnalysis(sentence, sentiment.English)

	if analysis.Score == 1 {
		return "Positive", 1.0
	} else {
		return "Negative", 0.0
	}
}
