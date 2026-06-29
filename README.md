# BatsASR
BatsASR is a fine-tuned Whisper-based automatic speech recognition model for Batsbi / Tsova-Tush. It transcribes audio into the current Georgian-based orthography used in the project and reaches approximately 0.91 character accuracy.
## Colab quick start

### 1. Clone the repository

```python
!git clone https://github.com/m4ru51x/BatsASR.git
%cd batsbi-whisper-asr
```

### 2. Install dependencies

```python
!pip install -r requirements.txt
```

### 3. Install/check the model

The model is already included in the repository. Run the model setup script:

```python
!python scripts/install_model.py \
--model "models/final_batsbi_whisper"
```

### 4. Transcribe audio

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
