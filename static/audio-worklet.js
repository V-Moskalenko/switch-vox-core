import { CaptureResampler, PlaybackBuffer } from "./audio-core.mjs";

class VoxCapture extends AudioWorkletProcessor {
    constructor() {
        super();
        this.active = false;
        this.converter = null;
        this.meterFrames = 0;
        this.port.onmessage = ({ data }) => {
            if (data.type === "capture.start") {
                this.converter = new CaptureResampler(sampleRate);
                this.active = true;
            } else if (data.type === "capture.stop") {
                this.active = false;
                // Последний неполный кадр дополняется до формата PCM до подтверждения capture.flushed.
                for (const buffer of this.converter?.flush() ?? []) this.port.postMessage({ type: "capture.frame", buffer }, [buffer]);
                this.port.postMessage({ type: "capture.flushed" });
            } else if (data.type === "capture.clear") {
                this.active = false;
                this.converter = null;
            }
        };
    }

    process(inputs) {
        const channel = inputs[0]?.[0];
        if (this.active && channel) {
            for (const buffer of this.converter.push(channel)) this.port.postMessage({ type: "capture.frame", buffer }, [buffer]);
            if (++this.meterFrames % 12 === 0) {
                const rms = Math.sqrt(channel.reduce((sum, value) => sum + value * value, 0) / channel.length);
                this.port.postMessage({ type: "capture.level", value: Math.min(1, rms * 4) });
            }
        }
        return true;
    }
}

class VoxPlayback extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = new PlaybackBuffer(sampleRate);
        this.port.onmessage = ({ data }) => {
            try {
                if (data.type === "playback.start") this.buffer.start(data.generation_id);
                else if (data.type === "playback.clear") this.buffer.clear();
                else if (data.type === "playback.audio") this.buffer.push(data.header, data.buffer);
            } catch (error) {
                // Ошибка уходит в основной поток; там останавливается ответ и выводится безопасный код диагностики.
                this.buffer.clear();
                this.port.postMessage({ type: "playback.error", message: error.message });
            }
        };
    }

    process(_inputs, outputs) {
        const channel = outputs[0]?.[0];
        if (channel) {
            const progress = this.buffer.render(channel);
            // Отмечаем конец обработанного кванта; основной поток дождётся его фактического вывода устройством.
            if (progress.length) this.port.postMessage({ type: "playback.progress", progress, context_time: (currentFrame + channel.length) / sampleRate });
        }
        return true;
    }
}

registerProcessor("vox-capture", VoxCapture);
registerProcessor("vox-playback", VoxPlayback);
