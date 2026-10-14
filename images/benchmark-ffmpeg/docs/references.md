# References

Terms specific to this image, alphabetized. Common terms (e.g. AWS, CV, NUMA,
SMT) are defined in the [glossary](https://docs.sparecores.com/glossary) of
the Spare Cores docs, which also lists the terms each image uses.

- **AAC** (Advanced Audio Coding): A lossy audio codec, the usual audio of MP4
  files. [FFmpeg AAC encoder](https://ffmpeg.org/ffmpeg-codecs.html#aac)
- **AV1** (AOMedia Video 1): A royalty-free video codec, the successor of VP9.
  [Specification](https://aomedia.org/specifications/av1/)
- **CC BY 3.0** (Creative Commons Attribution 3.0): A license allowing reuse
  with attribution. [License](https://creativecommons.org/licenses/by/3.0/)
- **CC0 1.0** (Creative Commons Zero 1.0): A public domain dedication.
  [License](https://creativecommons.org/publicdomain/zero/1.0/)
- **CQ** (Constant Quality): The quality-based rate control of NVENC, the GPU
  counterpart of CRF.
  [NVENC rate control](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-video-encoder-api-prog-guide/index.html#rate-control)
- **CRF** (Constant Rate Factor): The quality-based rate control of x264 and
  x265: lower values mean higher quality and larger files.
  [FFmpeg wiki](https://trac.ffmpeg.org/wiki/Encode/H.264#crf)
- **FFmpeg**: The command-line tool and libraries this image benchmarks.
  [`ffmpeg` docs](https://ffmpeg.org/ffmpeg.html)
- **FLAC** (Free Lossless Audio Codec): A lossless audio codec.
  [FLAC](https://xiph.org/flac/)
- **NVDEC** (NVIDIA Video Decoder): The hardware video decoder of NVIDIA GPUs.
  [NVIDIA GPUs in FFmpeg](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/ffmpeg-with-nvidia-gpu/index.html)
- **NVENC** (NVIDIA Video Encoder): The hardware video encoder of NVIDIA GPUs.
  [NVIDIA GPUs in FFmpeg](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/ffmpeg-with-nvidia-gpu/index.html)
- **null muxer**: An FFmpeg output format that discards its input, to measure
  encoding or decoding without writing a file.
  [FFmpeg formats](https://ffmpeg.org/ffmpeg-formats.html#null)
- **PSNR** (Peak Signal-to-Noise Ratio): A video quality metric comparing an
  encode to its source, in decibels.
  [FFmpeg filter](https://ffmpeg.org/ffmpeg-filters.html#psnr)
- **SSIM** (Structural Similarity Index Measure): A video quality metric
  comparing an encode to its source, from 0 to 1.
  [FFmpeg filter](https://ffmpeg.org/ffmpeg-filters.html#ssim)
- **Tears of Steel**: An open movie by the Blender Foundation, the source of
  the test clips. [Project page](https://mango.blender.org/)
- **VMAF** (Video Multimethod Assessment Fusion): A perceptual video quality
  metric by Netflix. [GitHub](https://github.com/Netflix/vmaf)
