import sys
import requests
import sounddevice
import pyaudio
import numpy
import torch
from TTS.api import TTS
from faster_whisper import WhisperModel

SPEAKER_MODEL="tts_models/en/ljspeech/tacotron2-DDC"
LLM_GENERATE_API="http://localhost:11434/api/generate"
LLM_MODEL="llama3"
DEVICE="cuda" # Cuda or cpu, cpu is painfully slow

print("Loading Voice Activity Detection Model...")
vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True, trust_repo=True)
print("Loading Speetch to Text Model")
stt_model = WhisperModel("base.en", device="cuda", compute_type="float16")

print("Loading Text to Speech Model...")
tts_model = TTS(SPEAKER_MODEL).to(DEVICE)

print("Retrieving audio device...")
audio = pyaudio.PyAudio()
# This block of code is responsible for playing the synthesized speech.
# It uses the PyAudio library, which is a Python binding for PortAudio,
# a free cross-platform audio I/O library.

def int2float(sound):
    # Convert the input sound data from integers to floating-point numbers.
    abs_max = numpy.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        # Scale the sound data to a range that's more suitable for audio processing.
        sound *= 1/32768
    sound = sound.squeeze()
    return sound

def query_ollama(query):
    # Send an HTTP request to the LLM API and retrieve its response.
    url = LLM_GENERATE_API
    headers = {}
    data = {"model": LLM_MODEL, "stream": False, "prompt": query}

    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        # If the response is successful, return its JSON payload.
        return response.json().get("response", "No response from LLM")
    else:
        # If there's an error, return a descriptive message instead.
        return "Error: " + response.text

def speak_text(text):
    # Synthesize text into audio using the TTS model and play it back.
    wav = tts_model.tts(text=text)
    sounddevice.play(wav, samplerate=22050)
    sounddevice.wait()

def listen_to_voice():
    # Initialize the PyAudio library for capturing audio input.
    num_samples = 1536

    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=num_samples)

    print("Started Recording")
    while True:
        data = []
        low_confidence_count = 0
        high_confidence_count = 0
        while True:
            # Capture audio chunks from the microphone.
            audio_chunk = stream.read(num_samples)
            audio_int16 = numpy.frombuffer(audio_chunk, numpy.int16)
            audio_float32 = int2float(audio_int16)
            confidence = vad_model(torch.from_numpy(audio_float32), 16000).item()

            if(confidence < 0.05):
                # If the audio is deemed low-confidence (i.e., background noise),
                # increment a counter to track the number of low-confidence samples.
                low_confidence_count += 1
            else:
                # If the audio is deemed high-confidence (i.e., speech), 
                # append it to our data buffer and reset the low-confidence count.
                high_confidence_count += 1
                data.append(audio_chunk)
                low_confidence_count = 0

            if low_confidence_count > 15:
                # If we've seen too many consecutive frames of low confidence,
                # break out of this loop to stop recording.
                break
        if high_confidence_count > 5: # Five catches to reduce false positives
            # Convert our audio buffer into a single numpy array.
            audio_data = numpy.frombuffer(b''.join(data), dtype=numpy.int16)
            # Transcribe the captured speech using the WhisperModel.
            result = stt_model.transcribe(audio_data, beam_size=5, language="en")
            segments, info = result
            transcribed_text = " ".join([segment.text for segment in segments])
            if(transcribed_text != ""):
                break
        print("Still listening...")
    
    print(f"Stopped the recording: {transcribed_text} | {high_confidence_count}")
    return transcribed_text

def main():
    # Main loop that listens to audio input, transcribes speech, 
    # sends queries to LLM, and plays back synthesized responses.
    while True:
        print("Waiting for instructions")
        query = listen_to_voice()
        if query.lower() == "quit":
            print("Exiting...")
            sys.exit(0)
        response = query_ollama(query)
        print("Response: " + response)

        speak_text(response)

if __name__ == "__main__":
    main()
