package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"time"

	aai "github.com/AssemblyAI/assemblyai-go-sdk"
	"github.com/cdipaolo/sentiment"
	"github.com/gordonklaus/portaudio"
	"strings"
	"unicode"
)

const (
	sampleRate = 44100
	channels   = 1
	sampleSize = 2  // 16-bit samples
	recordTime = 10 // Record for 10 seconds
	apiKey     = "22d890b9b686402b8f7b893dadaac295"
)

func main() {
	fmt.Println("Hi I am a NLP program designed to analyze a pick it date conversation and let you know how likely she will go with you on the pick it date is")

	// Initialize PortAudio
	err := portaudio.Initialize()
	if err != nil {
		panic(fmt.Sprintf("Error initializing PortAudio: %v", err))
	}
	defer portaudio.Terminate()

	// Record audio
	audioData := recordAudio()

	// Check if buffer contains data
	if allZero(audioData) {
		fmt.Println("Buffer is empty, no audio data recorded.")
		return
	}

	// Save audio to file
	saveAudioToFile(audioData, "recorded_audio.wav")

	// Transcribe audio
	text, err := transcribeAudio(apiKey, "recorded_audio.wav")
	if err != nil {
		panic(fmt.Sprintf("Error transcribing audio: %v", err))
	}

	if text == "" {
		fmt.Println("No speech detected in the recording or transcription failed.")
		return
	}

	fmt.Println("Transcribed Text:")
	fmt.Println(text)

	// Initialize the sentiment model
	model, err := sentiment.Restore()
	if err != nil {
		panic(fmt.Sprintf("Could not restore sentiment model: %v", err))
	}

	// Sentence Segmentation
	sentences := segmentSentences(text)
	fmt.Println("\nSentences:")
	for _, sentence := range sentences {
		fmt.Printf("- %s\n", sentence)
	}

	// Sentiment Analysis
	var total float64 = 0
	var countTot int = 0
	for _, sentence := range sentences {
		countTot++
		sentiment, confidence := analyzeSentiment(sentence, model)
		fmt.Printf("\nSentiment Analysis for: %s\n", sentence)
		fmt.Printf("Sentiment: %s\n", sentiment)
		fmt.Printf("Confidence: %.2f\n", confidence)
		total += confidence
	}
	if countTot == 0 {
		fmt.Println("No sentences to analyze.")
		return
	}
	averageConfidence := total / float64(countTot)

	// Print the average confidence
	fmt.Printf("\nAverage Confidence: %.2f\n", averageConfidence)

	// Determine confidence level based on averageConfidence
	var confidenceLevel string
	switch {
	case averageConfidence >= 0.8:
		confidenceLevel = "High Confidence"
	case averageConfidence >= 0.5:
		confidenceLevel = "Medium Confidence"
	default:
		confidenceLevel = "Low Confidence"
	}

	fmt.Printf("Confidence Level: %s\n", confidenceLevel)
}

func recordAudio() []int16 {
	fmt.Println("Recording... Speak into the microphone.")

	buffer := make([]int16, sampleRate*channels*50)
	devices, err := portaudio.Devices()
	if err != nil {
		panic(fmt.Sprintf("Error getting devices: %v", err))
	}

	var inputDevice *portaudio.DeviceInfo
	for _, d := range devices {
		if d.MaxInputChannels > 0 {
			inputDevice = d
			break
		}
	}

	if inputDevice == nil {
		panic("No input device found")
	}

	// Use low latency parameters with a smaller buffer size
	parameters := portaudio.LowLatencyParameters(inputDevice, nil)
	parameters.FramesPerBuffer = 128 // Adjust this value to a lower number
	stream, err := portaudio.OpenStream(parameters, buffer)
	if err != nil {
		panic(fmt.Sprintf("Error opening stream: %v", err))
	}
	defer stream.Close()

	err = stream.Start()
	if err != nil {
		panic(fmt.Sprintf("Error starting stream: %v", err))
	}

	// Actively read data into the buffer without sleeping
	go func() {
		for i := 0; i < 50; i++ {
			fmt.Printf("\rRecording... %d seconds remaining", 50-i-1)
			time.Sleep(time.Second)
		}
		fmt.Println("\nRecording finished.")
	}()

	stream.Read()

	//for i := 0; i < recordTime; i++ {
	//	err := stream.Read()
	//if err != nil {
	//	if err.Error() == "Input overflowed" {
	//		fmt.Println("Input overflow detected. Continuing...")
	//		continue
	//	}
	//	panic(fmt.Sprintf("Error reading from stream: %v", err))
	//}
	//}

	err = stream.Stop()
	if err != nil {
		panic(fmt.Sprintf("Error stopping stream: %v", err))
	}

	// Check if buffer contains data
	if allZero(buffer) {
		fmt.Println("Buffer is empty, no audio data recorded.")
	}

	return buffer
}

func allZero(data []int16) bool {
	for _, sample := range data {
		if sample != 0 {
			return false
		}
	}
	return true
}

func saveAudioToFile(data []int16, filename string) {
	file, err := os.Create(filename)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	// Write WAV header
	writeWAVHeader(file, len(data))

	// Write audio data
	for _, sample := range data {
		err = binary.Write(file, binary.LittleEndian, sample)
		if err != nil {
			panic(err)
		}
	}

	fmt.Printf("Audio saved to %s\n", filename)
}

func writeWAVHeader(file io.Writer, samples int) {
	// RIFF header
	file.Write([]byte("RIFF"))
	binary.Write(file, binary.LittleEndian, int32(36+samples*sampleSize))
	file.Write([]byte("WAVE"))

	// fmt chunk
	file.Write([]byte("fmt "))
	binary.Write(file, binary.LittleEndian, int32(16))
	binary.Write(file, binary.LittleEndian, int16(1))
	binary.Write(file, binary.LittleEndian, int16(channels))
	binary.Write(file, binary.LittleEndian, int32(sampleRate))
	binary.Write(file, binary.LittleEndian, int32(sampleRate*channels*sampleSize))
	binary.Write(file, binary.LittleEndian, int16(channels*sampleSize))
	binary.Write(file, binary.LittleEndian, int16(16))

	// data chunk
	file.Write([]byte("data"))
	binary.Write(file, binary.LittleEndian, int32(samples*sampleSize))
}

func transcribeAudio(apiKey, audioFile string) (string, error) {
	client := aai.NewClient(apiKey)
	ctx := context.Background()

	file, err := os.Open(audioFile)
	if err != nil {
		return "", err
	}
	defer file.Close()

	transcript, err := client.Transcripts.TranscribeFromReader(ctx, file, &aai.TranscriptOptionalParams{})
	if err != nil {
		fmt.Printf("Error transcribing audio: %v\n", err)
		return "", err
	}

	if transcript.Text == nil || *transcript.Text == "" {
		fmt.Println("No speech detected in the recording or transcription failed.")
		return "", nil
	}

	return *transcript.Text, nil
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

func segmentSentences(text string) []string {
	return strings.FieldsFunc(text, func(r rune) bool {
		return r == '.' || r == '!' || r == '?'
	})
}

func analyzeSentiment(sentence string, model sentiment.Models) (string, float64) {
	analysis := model.SentimentAnalysis(sentence, sentiment.English)

	if analysis.Score == 1 {
		return "Positive", 1.0
	} else {
		return "Negative", 0.0
	}
}
