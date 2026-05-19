# FFmpeg Helpers

FFmpeg is optional but useful for merging Playwright video output with separately recorded narration.

PowerShell example:

```powershell
./merge-audio-video.ps1 -VideoPath "../../recordings/raw/demo.webm" -AudioPath "../../audio/narration.wav" -OutputPath "../../recordings/final/demo.mp4"
```

Bash example:

```bash
./merge-audio-video.sh ../../recordings/raw/demo.webm ../../audio/narration.wav ../../recordings/final/demo.mp4
```

The helpers validate inputs, check that `ffmpeg` is available, and write a merged output file.

