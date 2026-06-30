# BatsASR
BatsASR is a fine-tuned Whisper-based automatic speech recognition model for Batsbi / Tsova-Tush. It transcribes audio into the current Georgian-based orthography used in the project and reaches approximately 0.91 character accuracy.

![alt text](https://panthernow.com/wp-content/uploads/pasted-image-0-25.png)

## Colab quick start

### 1. Clone the repository

```python
!git clone https://github.com/m4ru51x/BatsASR.git
%cd BatsASR
```

### 2. Install dependencies

```python
!pip install -r requirements.txt
```

### 3. Transcribe audio

Upload an audio file:

```python
from google.colab import files

uploaded = files.upload()
audio_path = next(iter(uploaded.keys()))
```

Run ASR:

```python
!python scripts/transcribe_file.py \
--model "models/final_batsbi_whisper" \
--audio "$audio_path"
```

The current pipeline is:

```text
audio → fine-tuned Whisper → Batsbi transcription
```

### Contributor

Maria Medvedeva, DoTaCL MSU
