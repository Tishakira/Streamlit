import streamlit as st
import speech_recognition as sr
import time

# Function to transcribe speech with enhanced functionality
def transcribe_speech(api, language):
    # Initialize recognizer class
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Speak now... (Pause or Resume as needed)")
        try:
            # Adjust microphone for ambient noise
            r.adjust_for_ambient_noise(source)
            audio_text = r.listen(source, timeout=10)
            st.info("Transcribing...")
            
            # Use selected API for transcription
            if api == "Google":
                text = r.recognize_google(audio_text, language=language)
            elif api == "Sphinx":
                text = r.recognize_sphinx(audio_text, language=language)
            else:
                return "Unsupported API selected."

            return text

        except sr.UnknownValueError:
            return "Could not understand the audio. Please try again."
        except sr.RequestError as e:
            return f"API request failed: {e}"
        except sr.WaitTimeoutError:
            return "No speech detected within the time limit. Please try again."
        except Exception as e:
            return f"An unexpected error occurred: {e}"

# Main function for the Streamlit app
def main():
    st.title("Enhanced Speech Recognition App")
    st.write("Select options below and click the button to start speaking:")

    # API selection
    api = st.selectbox("Select Speech Recognition API", ["Google", "Sphinx"])
    
    # Language selection
    language = st.text_input("Enter language code (e.g., 'en-US' for English, 'fr-FR' for French):", "en-US")
    
    # Add pause/resume toggle
    pause = st.checkbox("Pause after each recognition (for longer sessions)")

    # Add Start Recording button
    if st.button("Start Recording"):
        transcriptions = []
        while True:
            # Transcribe speech
            text = transcribe_speech(api, language)
            transcriptions.append(text)
            st.write(f"Transcription: {text}")

            if pause:
                if not st.button("Resume"):
                    break
            else:
                break

        # Save transcriptions to a file
        if st.button("Save Transcription"):
            with open("transcription.txt", "w") as f:
                f.write("\n".join(transcriptions))
            st.success("Transcription saved to 'transcription.txt'.")

# Run the app
if __name__ == "__main__":
    main()
