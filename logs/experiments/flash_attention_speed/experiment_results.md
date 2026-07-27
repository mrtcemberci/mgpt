| Flash Attention | Window | Batch | Checkpointing | Net Peak VRAM (MB) | Avg Step Time (ms) | Status |
|-----------------|--------|-------|---------------|--------------------|--------------------|--------|
| On | 128 | 8 | On | 4460 | 217 | Success |
| On | 256 | 8 | On | 5060 | 465 | Success |
| On | 512 | 8 | On | 5788 | 1305 | Success |
| On | 1024 | 8 | On | 7180 | 4265 | Success |
| On | 128 | 16 | On | 5060 | 311 | Success |
| On | 256 | 16 | On | 5788 | 769 | Success |
| On | 512 | 16 | On | 7180 | 2384 | Success |
| On | 1024 | 16 | On | 10514 | 8131 | Success |
| On | 128 | 8 | Off | 4460 | 173 | Success |
| On | 256 | 8 | Off | 5060 | 384 | Success |
| On | 512 | 8 | Off | 5788 | 1088 | Success |
| On | 1024 | 8 | Off | OOM (> 6364 MB) | N/A | OOM / Crash |
| On | 128 | 16 | Off | 5060 | 255 | Success |
| On | 256 | 16 | Off | 5788 | 641 | Success |
| On | 512 | 16 | Off | OOM (> 6184 MB) | N/A | OOM / Crash |
| On | 1024 | 16 | Off | OOM (> 6606 MB) | N/A | OOM / Crash |
| Off | 128 | 8 | On | 4524 | 128 | Success |
| Off | 256 | 8 | On | 5220 | 185 | Success |
| Off | 512 | 8 | On | 6428 | 412 | Success |
| Off | 1024 | 8 | On | 9740 | 1368 | Success |
| Off | 128 | 16 | On | 5156 | 168 | Success |
| Off | 256 | 16 | On | 6108 | 291 | Success |
| Off | 512 | 16 | On | 8460 | 887 | Success |
| Off | 1024 | 16 | On | 11981 | 14694 | Success |
| Off | 128 | 8 | Off | 4524 | 103 | Success |
| Off | 256 | 8 | Off | 5220 | 146 | Success |
| Off | 512 | 8 | Off | 6428 | 310 | Success |
| Off | 1024 | 8 | Off | OOM (> 6814 MB) | N/A | OOM / Crash |
| Off | 128 | 16 | Off | 5156 | 133 | Success |
| Off | 256 | 16 | Off | 6108 | 236 | Success |
| Off | 512 | 16 | Off | OOM (> 7844 MB) | N/A | OOM / Crash |
| Off | 1024 | 16 | Off | OOM (> 7126 MB) | N/A | OOM / Crash |
