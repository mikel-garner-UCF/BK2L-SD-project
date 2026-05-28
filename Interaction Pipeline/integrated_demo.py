"""Knightro Integrated Demo — Face Recognition + Voice Interaction + TTS

This script ties together all subsystems into a single demo:
  1. Camera opens and runs face detection/tracking/recognition
  2. Wake word ("Hey Knightro") activates the system
  3. When a known person is detected, Knightro asks for confirmation
  4. Commands are routed through the interaction pipeline
  5. Knightro speaks the response (via Piper TTS)

Usage:
    cd "Interaction Pipeline"
    python3 integrated_demo.py

    Press 'q' in the camera window to quit.
    Press 'w' to simulate wake word.

Requirements:
    pip install piper-tts sounddevice soundfile opencv-python mediapipe onnxruntime
"""

import os
import sys
import time
import threading
import queue

# Add paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_CV_SRC = os.path.join(_PROJECT_ROOT, "Computer Vision", "src")
_CV_ROOT = os.path.join(_PROJECT_ROOT, "Computer Vision")
sys.path.insert(0, _CV_SRC)
sys.path.insert(0, _SCRIPT_DIR)

import cv2

from face_detection import FaceDetector, draw_boxes
from face_tracker import FaceTracker
from face_recognizer import FaceRecognizer, RecognitionResult

import intent_detection
import interaction_router
import safety_filters
import system_state
import tts


# ---- SETTINGS ----
CAMERA_INDEX = 0
RECOGNITION_COOLDOWN = 10.0
GREETING_DELAY_FRAMES = 3       # 3 seconds at 30 FPS
STT_MODE = "whisper"                # "typed" or "whisper"
WAKE_WORD_MODE = True


class KnightroDemo:
    def __init__(self):
        print("=" * 55)
        print("  Knightro Integrated Demo")
        print("  Face Recognition + Voice + TTS")
        print("=" * 55)
        print()

        print("[demo] Initializing face detection...")
        self.detector = FaceDetector(min_confidence=0.5)

        print("[demo] Initializing face tracker...")
        self.tracker = FaceTracker(iou_threshold=0.3, max_missed_frames=15)

        # Load face database from the correct path
        from pathlib import Path
        from face_database import FaceDatabase

        cv_data_dir = Path(_CV_ROOT) / "data"
        db_path = cv_data_dir / "dlib_face_embeddings.enc"
        key_path = cv_data_dir / ".dlib_encryption_key"

        print(f"[demo] Looking for face database at: {db_path}")
        print(f"[demo] Database exists: {db_path.exists()}")

        db = FaceDatabase(db_path=db_path, key_path=key_path)

        print("[demo] Initializing face recognizer...")
        self.recognizer = FaceRecognizer(
            similarity_threshold=0.970,
            db=db,
        )

        print(f"[demo] System state: {'ONLINE' if system_state.is_online() else 'OFFLINE'}")
        print(f"[demo] Enrolled faculty: {self.recognizer.enrolled_count}")
        print()

        self._greeted: dict = {}
        self._recognized_tracks: set = set()
        self._activated = not WAKE_WORD_MODE
        self._busy = False  # True when interaction is in progress

        # Input queue — background threads put user text here
        self._input_queue: queue.Queue = queue.Queue()
        self._waiting_for_input = False

        # Status message shown on camera overlay
        self._status_msg = 'IDLE — Say "Hey Knightro!" or press W' if WAKE_WORD_MODE else "ACTIVE"

    def run(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print("[demo] ERROR: Could not open camera!")
            return

        print("[demo] Camera opened. Controls:")
        print("  q = quit")
        print("  w = wake word (activate Knightro)")
        print()

        # Store camera reference so recognition retries can grab fresh frames
        self._cap = cap

        # Start the input reader thread (reads typed input without blocking)
        if STT_MODE == "typed":
            threading.Thread(target=self._input_reader, daemon=True).start()

        # Start wake word listener immediately
        if WAKE_WORD_MODE and not self._activated:
            threading.Thread(target=self._wait_for_wake_word, daemon=True).start()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Mirror the frame — matches how enrollment captures were taken
                frame = cv2.flip(frame, 1)

                boxes = self.detector.detect(frame)
                active_tracks = self.tracker.update(boxes)

                # Only run recognition when activated and not busy with interaction
                if self._activated and not self._busy:
                    for track in active_tracks:
                        if track.track_id not in self._recognized_tracks:
                            if track.frames_seen >= GREETING_DELAY_FRAMES:
                                self._recognized_tracks.add(track.track_id)
                                self._busy = True
                                threading.Thread(
                                    target=self._recognize_and_greet,
                                    args=(frame.copy(), track),
                                    daemon=True,
                                ).start()

                display = self._draw_overlay(frame, active_tracks)
                cv2.imshow("Knightro Demo", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('w') and not self._activated:
                    self._activate()

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()
            print("[demo] Demo ended. Go Knights!")

    # ------------------------------------------------------------------
    # Input handling — runs in its own thread, never blocks the camera
    # ------------------------------------------------------------------

    def _input_reader(self):
        """Continuously read from stdin in a background thread."""
        while True:
            try:
                line = input().strip().lower()
                self._input_queue.put(line)
            except (EOFError, KeyboardInterrupt):
                break

    def _get_user_input(self, prompt: str, timeout: float = 30.0) -> str | None:
        """Get input from user without blocking the camera.

        In typed mode, prints a prompt and waits for the input reader thread.
        In whisper mode, uses Whisper STT.
        Returns None on timeout.
        """
        # Clear any stale input
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                break

        if STT_MODE == "typed":
            print(f"{prompt}", end="", flush=True)
            self._waiting_for_input = True
            try:
                text = self._input_queue.get(timeout=timeout)
                self._waiting_for_input = False
                return text
            except queue.Empty:
                self._waiting_for_input = False
                print("\n[demo] Timed out.")
                return None
        else:
            import speech_to_text
            result = speech_to_text.speech_to_text()
            if result["error"]:
                print(f"[demo] STT: {result['reason']}")
                return None
            return result["text"]

    # ------------------------------------------------------------------
    # Wake word
    # ------------------------------------------------------------------

    def _activate(self):
        self._status_msg = "ACTIVE — Scanning..."
        print("[demo] ACTIVATED!")
        # Speak BEFORE setting _activated so the camera loop doesn't
        # start recognition while Knightro is still talking
        tts.speak("Hey! I'm here! Let me see who I'm talking to!")
        time.sleep(1.0)
        # NOW enable recognition
        self._activated = True
        self._recognized_tracks.clear()
        self._greeted.clear()

    def _deactivate(self):
        self._activated = False
        self._busy = False
        self._recognized_tracks.clear()
        self._greeted.clear()
        self._status_msg = 'IDLE — Say "Hey Knightro!" or press W'
        print("[demo] Back to idle.")

        # If wake word mode, start listening for it again
        if WAKE_WORD_MODE:
            threading.Thread(target=self._wait_for_wake_word, daemon=True).start()

    def _wait_for_wake_word(self):
        WAKE_PHRASES = ["hey knightro", "knightro", "hey nitro", "nitro",
                        "hey knight", "hey night row"]

        # Keep listening until we hear the wake word
        while not self._activated:
            self._status_msg = 'IDLE — Say "Hey Knightro!" or press W'
            print("[demo] Waiting for wake word...")

            text = self._get_user_input("", timeout=8.0)

            if self._activated:
                # Got activated by 'W' key while we were listening
                return

            if text is None:
                # Timeout — just loop and listen again
                continue

            if any(phrase in text for phrase in WAKE_PHRASES):
                self._activate()
                return
            else:
                print(f"[demo] Heard '{text}', not the wake word. Still listening...")

    # ------------------------------------------------------------------
    # Face recognition + greeting
    # ------------------------------------------------------------------

    def _recognize_and_greet(self, frame, track):
        """Run face recognition with retries. Runs in a thread."""
        MAX_RETRIES = 3
        best_result = None

        for attempt in range(MAX_RETRIES):
            # Use the stored camera reference to grab a FRESH frame each retry
            if attempt > 0 and hasattr(self, '_cap') and self._cap is not None:
                ret, frame = self._cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                # Re-detect faces in the fresh frame
                boxes = self.detector.detect(frame)
                if not boxes:
                    print(f"[demo] Retry {attempt+1}: no face detected, skipping")
                    time.sleep(0.3)
                    continue
                # Use the largest face
                largest = max(boxes, key=lambda b: b.width * b.height)
                box_to_use = largest
            else:
                box_to_use = track.bbox

            margin = 20
            y1 = max(0, box_to_use.y - margin)
            y2 = min(frame.shape[0], box_to_use.y2 + margin)
            x1 = max(0, box_to_use.x - margin)
            x2 = min(frame.shape[1], box_to_use.x2 + margin)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            result = self.recognizer.recognize(face_crop)
            print(f"[demo] Recognition attempt {attempt+1}/{MAX_RETRIES}: "
                  f"name={result.name}, similarity={result.distance:.3f}, "
                  f"is_known={result.is_known}")

            if result.is_known:
                best_result = result
                break
            else:
                # Keep the best unknown result in case all retries fail
                if best_result is None or result.distance > best_result.distance:
                    best_result = result
                time.sleep(0.3)  # Brief pause before retry

        if best_result is None:
            self._busy = False
            return

        if best_result.is_known:
            self._status_msg = f"Recognized: {best_result.name}?"
            self._confirm_and_greet(best_result.name)
        else:
            self._status_msg = "Unknown visitor"
            self._greet_unknown_and_listen()

    def _confirm_and_greet(self, name: str, guess_count: int = 1, wrong_guesses: set = None):
        """Ask for confirmation, then greet or try next match."""
        if wrong_guesses is None:
            wrong_guesses = set()

        MAX_GUESSES = 2

        tts.speak(f"Hello! Are you {name}?")
        self._status_msg = f"Asking: Are you {name}?"
        time.sleep(0.5)  # Let audio finish before opening mic

        text = self._get_user_input("You (yes/no): ", timeout=15.0)
        if text is None:
            tts.speak("I didn't catch that. Nice to meet you anyway! Go Knights!")
            self._finish_interaction()
            return

        yes_words = ["yes", "yeah", "yep", "yup", "that's me", "correct", "si", "ye"]
        no_words = ["no", "nah", "nope", "wrong", "not me", "negative"]

        if any(w in text for w in yes_words):
            print(f"[demo] Confirmed: {name}")
            self._status_msg = f"Greeting {name}"
            greeting = self._greet_known(name)
            tts.speak(greeting)
            self._greeted[name] = time.time()
            self._listen_for_commands()

        elif any(w in text for w in no_words):
            wrong_guesses.add(name)

            if guess_count >= MAX_GUESSES:
                tts.speak("Sorry about that! You can say 'try again' if you'd like me to take another look. Otherwise, how can I help?")
                self._listen_for_commands()
            else:
                tts.speak("My apologies! Let me try again.")
                next_name = self._get_next_match(wrong_guesses)
                if next_name:
                    self._confirm_and_greet(next_name, guess_count + 1, wrong_guesses)
                else:
                    tts.speak("I don't think we've met! Welcome to UCF!")
                    self._listen_for_commands()
        else:
            tts.speak(f"I'll take that as a yes! Welcome, {name}! Go Knights!")
            self._greeted[name] = time.time()
            self._listen_for_commands()

    def _greet_unknown_and_listen(self):
        """Greet an unknown visitor and listen for commands."""
        import random
        greetings = [
            "Hey there! Welcome to UCF! I'm Knightro!",
            "What's up! Welcome to the Engineering building! Go Knights!",
            "Hey! I don't think we've met. I'm Knightro, UCF's mascot!",
            "Welcome, fellow Knight! How can I help you today?",
        ]
        tts.speak(random.choice(greetings))
        tts.speak("You can say 'try again' if you think I should know you!")
        self._listen_for_commands()

    def _get_next_match(self, exclude: set) -> str | None:
        for name in self.recognizer._enrolled:
            if name not in exclude:
                return name
        return None

    def _greet_known(self, name: str) -> str:
        import random
        return random.choice([
            f"Hey, {name}! Great to see you! Welcome back! Go Knights!",
            f"Well well well, if it isn't {name}! Good to see you around campus!",
            f"Look who it is! Welcome, {name}! Charge On!",
            f"Hey there, {name}! What's up! Always good to see a familiar face!",
            f"{name}! My favorite Knight! How's it going?",
        ])

    # ------------------------------------------------------------------
    # Voice command handling
    # ------------------------------------------------------------------

    def _listen_for_commands(self):
        """Listen for voice commands. Can be called multiple times for conversation."""
        self._status_msg = "Listening for command..."
        print("\n[demo] Listening... (say a command, 'try again', or 'goodbye')")

        text = self._get_user_input("You: ", timeout=20.0)
        if text is None:
            tts.speak("Looks like you're done. See you around! Go Knights!")
            self._finish_interaction()
            return

        # Try again
        retry_phrases = ["try again", "scan again", "look again", "retry",
                         "re-scan", "rescan", "check again"]
        if any(phrase in text for phrase in retry_phrases):
            tts.speak("Let me take another look!")
            self._recognized_tracks.clear()
            self._greeted.clear()
            self.tracker.reset()
            self._busy = False
            return

        # Safety check
        is_safe, _ = safety_filters.check_input_safety(text)
        if not is_safe:
            tts.speak("I can't help with that, but ask me about UCF!")
            self._listen_for_commands()
            return

        # Route intent
        intent = intent_detection.detect_intent(text)
        response = interaction_router.route_intent(intent, text, input_passed_safety=True)

        is_safe, _ = safety_filters.check_output_safety(response)
        if not is_safe:
            response = "Let me think of something else. Ask me about UCF!"

        print(f"[demo] Knightro: {response}")
        tts.speak(response)

        # Farewell -> back to idle
        if intent == "farewell":
            self._finish_interaction()
            return

        # Keep listening for more commands
        self._listen_for_commands()

    def _finish_interaction(self):
        """End the current interaction and return to idle."""
        self._busy = False
        if WAKE_WORD_MODE:
            self._deactivate()
        else:
            self._status_msg = "ACTIVE — Scanning..."
            self._recognized_tracks.clear()
            self._greeted.clear()

    def _should_greet(self, name: str) -> bool:
        if name not in self._greeted:
            return True
        return (time.time() - self._greeted[name]) > RECOGNITION_COOLDOWN

    # ------------------------------------------------------------------
    # Handle 't' key for quick typed commands
    # ------------------------------------------------------------------

    def _handle_typed_command(self):
        print("\n[demo] Type a command:")
        text = self._get_user_input("You: ", timeout=15.0)
        if not text:
            return

        is_safe, _ = safety_filters.check_input_safety(text)
        if not is_safe:
            tts.speak("I can't help with that, but ask me about UCF!")
            return

        intent = intent_detection.detect_intent(text)
        response = interaction_router.route_intent(intent, text, input_passed_safety=True)

        is_safe, _ = safety_filters.check_output_safety(response)
        if not is_safe:
            response = "Let me think of something else. Ask me about UCF!"

        print(f"[demo] Knightro: {response}")
        tts.speak(response)

    # ------------------------------------------------------------------
    # Camera overlay
    # ------------------------------------------------------------------

    def _draw_overlay(self, frame, tracks):
        display = frame.copy()

        for track in tracks:
            box = track.bbox
            if self._activated:
                if hasattr(track, 'identity') and track.identity:
                    color = (0, 199, 255)  # gold
                    label = track.identity
                elif track.is_recognized:
                    color = (255, 255, 255)
                    label = "Visitor"
                else:
                    color = (128, 128, 128)
                    label = "Detecting..."
            else:
                color = (128, 128, 128)
                label = ""

            cv2.rectangle(display, (box.x, box.y), (box.x2, box.y2), color, 2)
            if label:
                cv2.putText(display, label, (box.x, max(0, box.y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Status
        is_active = self._activated
        color = (0, 255, 0) if is_active else (0, 165, 255)
        cv2.putText(display, self._status_msg, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        info = f"Enrolled: {self.recognizer.enrolled_count} | q=quit  w=wake"
        cv2.putText(display, info, (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return display


if __name__ == "__main__":
    demo = KnightroDemo()
    demo.run()