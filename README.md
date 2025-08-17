Offline AI Voice Assistant for Embedded Systems

An edge-deployed AI voice assistant designed for offline use on resource-constrained hardware (Raspberry Pi), optimized for low-latency voice interactions without cloud dependencies.

🔧 Technical Highlights

Model Architecture: DS-CNN (Depthwise Separable CNN) optimized for on-device speech tasks.

Inference Engine: TensorFlow Lite with INT8 quantized QNNs and tensor decomposition for efficient edge inference.

Audio Pipeline: µ-law PCM telephony audio processing for lightweight speech input.

Wake Word & ASR: Streamlined wake-word detection and automatic speech recognition with <50 ms inference latency.

Edge Hardware: Built and tested on Linux-based Raspberry Pi hardware for robust, context-aware command execution.

⚡ Performance

Latency: Achieved sub-50 ms response times for natural language commands.

Resiliency: Fully offline (no internet required) → privacy-preserving and reliable in low-connectivity environments.

Portability: Optimized for deployment across embedded Linux systems.
