"""Part 13 demo: talk to the chatbot. Speech in (Whisper), speech out (VITS), Part 12's chatbot in the middle.

    python voice_demo.py                       # spoken questions are made with the voice, then answered
    python voice_demo.py my_question.wav       # or answer your own recording (any audio format)
Answers are saved as WAV files in artifacts/ (open them to listen).
"""
import sys
from pathlib import Path

from transformers.utils import logging as transformers_logging

from reviewsense.chat.dialog import Chatbot
from reviewsense.config import get_settings
from reviewsense.data import load_reviews
from reviewsense.safety.safe_assistant import SafeAssistant
from reviewsense.speech.evaluate import score_domain, score_real_speech
from reviewsense.speech.stt import Transcriber
from reviewsense.speech.tts import Speaker
from reviewsense.speech.voice import VoiceAssistant

transformers_logging.disable_progress_bar()          # hide "Loading weights" bars to keep the output readable
transformers_logging.set_verbosity_error()          # and generation-settings notices

out = get_settings().artifacts_dir
speaker = Speaker()
voice = VoiceAssistant(Chatbot(SafeAssistant(load_reviews())), Transcriber(), speaker)

# 1. Questions as audio: your own recording, or spoken by the voice (saved as artifacts/question_N.wav).
if len(sys.argv) > 1:
    questions = [sys.argv[1]]
else:
    questions = []
    for number, text in enumerate(["Is Taqueria El Sol clean?", "Book a table at Bella Napoli for two on Friday at 8 pm",
                                   "yes"], start=1):
        questions.append(speaker.save(text, str(out / f"question_{number}.wav")))

# 2. Each one: transcribed, answered by the chatbot, answer spoken (artifacts/answer_N.wav).
for number, audio in enumerate(questions, start=1):
    result = voice.respond(audio, "voice-demo", out / f"answer_{number}.wav")
    print(f"I heard: {result['transcript']}")
    print(f"Answer:  {result['answer']}")
    print(f"Spoken:  {result['audio_file']}\n")

# 3. Measured: real human speech (LibriSpeech) and our restaurant vocabulary.
print("Real speech:", score_real_speech(voice.transcriber))
domain = score_domain(voice.transcriber, speaker)
print(f"Restaurant questions: WER {domain['wer']:.0%}, restaurant name recognised {domain['restaurant_found']:.0%}")
print(f"(Listen to the files in {Path(out).resolve()})")
