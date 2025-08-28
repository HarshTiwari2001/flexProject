import speech_recognition as sr
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

RECORD_SECONDS = 1000
FILE_NAME = "output.txt"
LANGUAGE = "en-IN"
PHRASE_TIME_LIMIT = 10
TIMEOUT = 3

recognizer = sr.Recognizer()


def recognize_and_write(audio: sr.AudioData):
    """Non-blocking: recognize text and append to output.txt"""
    try:
        text = sr.Recognizer().recognize_google(audio, language=LANGUAGE)
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        line = f"{timestamp} : {text}"
        print(line, flush=True)
        with open(FILE_NAME, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except sr.UnknownValueError:
        pass
    except sr.RequestError as e:
        print(f"Speech service error: {e}", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)


def main():
    print("Program starting...")
    Path(FILE_NAME).touch(exist_ok=True)
    print(f"Writing to: {Path(FILE_NAME).resolve()}")

    start_time = time.time()

    with sr.Microphone(
        sample_rate=16000, chunk_size=1024
    ) as source, ThreadPoolExecutor(max_workers=2) as pool:
        print("Adjusting for ambient noise (0.5s)...")
        recognizer.dynamic_energy_threshold = False
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        recognizer.energy_threshold = int(recognizer.energy_threshold * 1.5)

        recognizer.pause_threshold = 0.35
        recognizer.non_speaking_duration = 0.2
        recognizer.phrase_threshold = 0.1

        print("Listening...")
        pending = set()

        while True:
            elapsed = time.time() - start_time
            remaining = RECORD_SECONDS - int(elapsed)
            if remaining <= 0:
                break

            print(f"Time left: {remaining} seconds", end="\r")

            try:
                audio_data = recognizer.listen(
                    source, timeout=TIMEOUT, phrase_time_limit=PHRASE_TIME_LIMIT
                )
                fut = pool.submit(recognize_and_write, audio_data)
                pending.add(fut)

                if len(pending) > 4:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"Error: {e}", flush=True)
                time.sleep(0.05)

            time.sleep(0.05)

        if pending:
            wait(pending)

    print(f"\nFinished recording. Transcription saved to {FILE_NAME}")


if __name__ == "__main__":
    main()
