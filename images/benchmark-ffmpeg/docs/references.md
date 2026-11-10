# References

Terms specific to this image, alphabetized. Common terms (e.g. AWS, CV, NUMA,
SMT) are defined in the [glossary](https://docs.sparecores.com/glossary) of
the Spare Cores docs, which also lists the terms each image uses.

- **AAC** (Advanced Audio Coding): A lossy audio codec standardized in MPEG-2
  and MPEG-4 as the successor of MP3. It sounds better than MP3 at the same
  bitrate, and it is the usual audio track of MP4 files and streaming video.
  [FFmpeg AAC encoder](https://ffmpeg.org/ffmpeg-codecs.html#aac)
- **AV1** (AOMedia Video 1): A royalty-free video codec by the Alliance for Open
  Media, the successor of VP9. It compresses better than H.264 and HEVC at the
  same quality, at the cost of slower encoding. Common encoders are libaom,
  SVT-AV1 and rav1e.
  [Specification](https://aomedia.org/specifications/av1/)
- **CC BY 3.0** (Creative Commons Attribution 3.0): A Creative Commons license
  that allows anyone to copy, adapt and use a work, also commercially, as long
  as they credit its author.
  [License](https://creativecommons.org/licenses/by/3.0/)
- **CC0 1.0** (Creative Commons Zero 1.0): A Creative Commons dedication with
  which authors waive all their copyright in a work, placing it as close to the
  public domain as the law allows. Anyone may use the work for any purpose,
  without attribution.
  [License](https://creativecommons.org/publicdomain/zero/1.0/)
- **CQ** (Constant Quality): The quality-based rate control of NVIDIA's NVENC
  hardware encoder, which varies the bitrate to keep a target quality level.
  Lower values mean higher quality and larger files, as with CRF in software
  encoders.
  [NVENC rate control](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-video-encoder-api-prog-guide/index.html#rate-control)
- **CRF** (Constant Rate Factor): The quality-based rate control of the x264
  and x265 encoders, which varies the bitrate to keep the perceived quality
  constant across scenes. Lower values mean higher quality and larger files;
  x264's default is 23 on a scale of 0 to 51.
  [FFmpeg wiki](https://trac.ffmpeg.org/wiki/Encode/H.264#crf)
- **FFmpeg**: A free, open-source collection of libraries and command-line
  tools to decode, encode, transcode, filter and stream audio and video. Its
  `ffmpeg` command converts between nearly all media formats, and its
  libraries, such as libavcodec, power many media players and services.
  [`ffmpeg` docs](https://ffmpeg.org/ffmpeg.html)
- **FLAC** (Free Lossless Audio Codec): A lossless audio codec by the Xiph.Org
  Foundation. It compresses audio without discarding any information, so
  decoding restores the original samples bit for bit.
  [FLAC](https://xiph.org/flac/)
- **H.264**: A lossy video codec standardized in 2003, also called AVC. It is
  the most widely supported video format, used in Blu-ray, broadcast,
  streaming, and video calls. Common encoders are x264 and hardware encoders
  such as NVENC.
  [Specification](https://www.itu.int/rec/T-REC-H.264)
- **H.265**: Another name for HEVC, from the number of its standard.
  [Specification](https://www.itu.int/rec/T-REC-H.265)
- **HEVC** (High Efficiency Video Coding): A lossy video codec standardized in
  2013 as the successor of H.264, also called H.265. It compresses about twice
  as well as H.264 at the same quality, at the cost of slower encoding, and is
  used for 4K video and streaming. Common encoders are x265 and NVENC.
  [Specification](https://www.itu.int/rec/T-REC-H.265)
- **NVDEC** (NVIDIA Video Decoder): The hardware video decoder of NVIDIA GPUs,
  a dedicated unit next to the CUDA cores. It decodes codecs such as H.264,
  HEVC, VP9 and, on newer GPUs, AV1, leaving the CPU and the CUDA cores free.
  [NVIDIA GPUs in FFmpeg](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/ffmpeg-with-nvidia-gpu/index.html)
- **NVENC** (NVIDIA Video Encoder): The hardware video encoder of NVIDIA GPUs,
  a dedicated unit next to the CUDA cores. It encodes H.264, HEVC and, on newer
  GPUs, AV1, much faster than software encoders, though usually at a somewhat
  lower quality for the same bitrate. FFmpeg exposes it as encoders such as
  `h264_nvenc`.
  [NVIDIA GPUs in FFmpeg](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/ffmpeg-with-nvidia-gpu/index.html)
- **null muxer**: An FFmpeg output format, selected with `-f null`, that
  discards everything it receives instead of writing a file. It is used to
  measure decoding, filtering or encoding speed, or to run analysis filters,
  without the cost of writing to disk.
  [FFmpeg formats](https://ffmpeg.org/ffmpeg-formats.html#null)
- **Opus**: A royalty-free, lossy audio codec standardized by the IETF in 2012.
  It covers speech and music from low to high bitrates with low delay, which
  makes it common in voice and video calls and web streaming.
  [Opus](https://opus-codec.org/)
- **PSNR** (Peak Signal-to-Noise Ratio): A quality metric that compares a
  processed image or video to its source by the mean squared error of their
  pixels, in decibels. Higher values mean a closer match. It is simple and
  widely used, but follows perceived quality only loosely.
  [FFmpeg filter](https://ffmpeg.org/ffmpeg-filters.html#psnr)
- **SSIM** (Structural Similarity Index Measure): A quality metric that
  compares a processed image or video to its source by local patterns of
  luminance, contrast and structure. A score of 1 means identical, and the
  score follows perceived quality more closely than PSNR.
  [FFmpeg filter](https://ffmpeg.org/ffmpeg-filters.html#ssim)
- **Tears of Steel**: A 2012 science-fiction short film by the Blender
  Foundation, made with open-source tools and released under CC BY 3.0. Its
  high-resolution footage is freely available, which makes it common test
  material for video codecs. [Project page](https://mango.blender.org/)
- **VMAF** (Video Multimethod Assessment Fusion): A perceptual video quality
  metric by Netflix that combines several elementary metrics with a machine
  learning model trained on human ratings. It scores from 0 to 100, and is
  widely used to tune and compare video encoders.
  [GitHub](https://github.com/Netflix/vmaf)
- **Vorbis**: A royalty-free, lossy audio codec by the Xiph.Org Foundation,
  usually stored in Ogg files. It was designed as a free alternative to MP3 and
  AAC, and is used in games and streaming; Opus has largely replaced it for new
  uses. [Vorbis](https://xiph.org/vorbis/)
