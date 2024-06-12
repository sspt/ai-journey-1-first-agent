# AI Journey - The First Agent
This is my first agent, I'm going to start with a simple one. This will be the foundation for more complex agents in future projects. My aim is always to develop a usable agent that runs locally and can perform tasks like speech recognition, text to speech conversion, etc completly offline.

## Requirements:
### Hardware
1. A GPU with 6GB or more of RAM for faster processing
 - Ollama with llama3:latest (7B) - 5066 of VRAM
 - Python with FasterWhisper+CoquiTTS+VAD=688MB of RAM (you can squeeze this even more by using smaller models but the accuracy will be affected)
### Software
1. CUDA - Drivers and Toolkit
2. CUDNN - Neural Network library (**not included** in the CUDA Toolkit)
3. Python3 Enviornment - Preferably Anaconda Distribution
4. Numpy Library - For Data Manipulation
5. Torch Library - For Deep Learning Models (preferably PyTorch with CUDA support for GPU acceleration)
6. Faster Whisper - For local Speech-To-Text recognition
7. Coqui-TTS - For Text-to-Speech, the maintained pyPI project is 'coqui-tts' (please use this instead of the old, unmaintained 'TTS')

## On 'The First Agent'
The agent will rely on an pre-trained ollama model of your choice (I've used llama3-7B), it will:
1. Listen to your voice and 'assume' that you're not talking/stopped talking
2. Print what it 'understood' using a Speech-To-Text library (faster-whisper)
3. Query the ollama model for the full response (no streaming)
4. Print the response from Ollama
5. Read the response using coqui-tts

### How to Use
1. Run voice_assistant.py
```bash
python3 voice_assistant.py
```
2. Speak
3. Wait for the response
4. Go back to step 2 or if you want to quit, go to step 5
5. To quit press Ctrl+C or say 'quit'