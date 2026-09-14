curl http://127.0.0.1:8902/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "toy-model",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Please transcribe or evaluate this audio recording."
          },
          {
            "type": "audio_url",
            "audio_url": {
              "url": "file:///home/voice/data/voice/ASR/testset/16k/ultimate_testset/vietnamese_testset/vlsp2022-task1/wavs/2022_1005_00002640_00003214.wav"
            }
          }
        ]
      }
    ],
    "max_tokens": 512
  }'