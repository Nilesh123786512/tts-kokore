import streamlit as st
from kokoro import KPipeline
import numpy as np
import time
# Cache the pipeline for faster reloads
@st.cache_resource
def load_pipeline():
    return KPipeline(lang_code='a')

def main():
    st.title("🎙️ Kokoro TTS App")
    st.markdown("Text-to-Speech using Kokoro's lightweight AI model")
    
    # Initialize pipeline
    pipeline = load_pipeline()
    
    # App layout
    with st.sidebar:
        st.header("Settings")
        voice_option = st.selectbox(
            "Voice Style",
            ("af_heart",),  # Add more voices if available
            help="Select voice style for synthesis"
        )
    
    # Main content
    col1, col2 = st.columns([3, 1])
    with col1:
        input_text = st.text_area(
            "Input Text",
            height=200,
            value="[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters..."
        )

    with col2:
        st.markdown("**Text Formatting**")
        st.markdown("Use `/phonemes/` for exact pronunciation")
        st.markdown("Example: `[Kokoro](/kˈOkəɹO/)`")

    # Generation button
    if st.button("Generate Speech", type="primary"):
        if not input_text.strip():
            st.error("Please enter some text to synthesize")
            return
            
        with st.spinner("Synthesizing..."):
            try:
                # Generate audio chunks
                generator = pipeline(input_text, voice=voice_option)
                audio_chunks = []
                i=0
                time_start=time.time()
                for _, _, audio in generator:
                    # print(f"Chunk {i+1} completed")
                    i+=1
                    audio_chunks.append(audio)
                time_end=time.time()
                if not audio_chunks:
                    st.error("No audio generated")
                    return
                
                # Combine audio chunks
                full_audio = np.concatenate(audio_chunks)
                
                
                # Display results
                st.success(f"Audio generated!\nTime taken:{time_end-time_start}s")
                st.audio(full_audio,sample_rate=24000, format="audio/flac")
                
                
            except Exception as e:
                st.error(f"Error generating audio: {str(e)}")

if __name__ == "__main__":
    main()
