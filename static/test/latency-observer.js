// Диагностическое расширение AudioWorklet; запускающий код заменяет две исходные регистрации процессоров.
class MeasVoxCapture extends VoxCapture {
    constructor() {
        super();
        this.measurement = { threshold: 0.001, silenceMs: 200, probes: [], speaking: false, last: null };
        this.port.addEventListener("message", ({ data }) => {
            if (data.type === "benchmark.configure") {
                this.measurement.threshold = data.threshold;
                this.measurement.silenceMs = data.silenceMs;
            } else if (data.type === "benchmark.probe") this.measurement.probes.push(data.id);
            else if (["capture.stop", "capture.clear"].includes(data.type)) this.endMeasuredSpeech(data.type);
        });
        this.port.start();
    }

    endMeasuredSpeech(reason) {
        const state = this.measurement;
        if (!state.speaking) return;
        this.port.postMessage({ type: "benchmark.capture.end", frame: state.last + 1, sample_rate: sampleRate, reason });
        state.speaking = false;
    }

    process(inputs, outputs, parameters) {
        const state = this.measurement;
        for (const id of state.probes.splice(0)) {
            this.port.postMessage({ type: "benchmark.probe", id, context_time: currentFrame / sampleRate, quantum_frames: inputs[0]?.[0]?.length || 128, sample_rate: sampleRate });
        }
        const channel = inputs[0]?.[0];
        if (this.active && channel) {
            let first = -1;
            let last = -1;
            for (let i = 0; i < channel.length; i++) if (Math.abs(channel[i]) > state.threshold) {
                if (first < 0) first = i;
                last = i;
            }
            if (first >= 0) {
                if (!state.speaking) this.port.postMessage({ type: "benchmark.capture.start", frame: currentFrame + first, sample_rate: sampleRate });
                state.speaking = true;
                state.last = currentFrame + last;
            }
            if (state.speaking && currentFrame + channel.length - state.last - 1 >= state.silenceMs * sampleRate / 1000) this.endMeasuredSpeech("silence");
        }
        return super.process(inputs, outputs, parameters);
    }
}

class MeasVoxPlayback extends VoxPlayback {
    constructor() {
        super();
        this.measurement = { first: new Set(), last: null, clears: [], underflow: false, activityAt: -Infinity };
        const original = this.port.onmessage;
        this.port.onmessage = (event) => {
            if (event.data.type === "playback.clear") this.measurement.clears.push({
                generation_id: this.buffer.generation, queued_samples: this.buffer.bufferedSamples, last_nonzero_frame: this.measurement.last,
            });
            original(event);
        };
    }

    process(inputs, outputs, parameters) {
        const generation = this.buffer.generation;
        const queued = this.buffer.bufferedSamples;
        const result = super.process(inputs, outputs, parameters);
        const channel = outputs[0]?.[0];
        if (!channel) return result;
        const state = this.measurement;
        for (const clear of state.clears.splice(0)) this.port.postMessage({
            type: "benchmark.playback.clear", ...clear, frame: currentFrame, sample_rate: sampleRate, quantum_frames: channel.length,
        });
        let first = -1;
        let last = -1;
        for (let i = 0; i < channel.length; i++) if (channel[i] !== 0) {
            if (first < 0) first = i;
            last = i;
        }
        if (first >= 0 && generation) {
            state.last = currentFrame + last;
            if (!state.first.has(generation)) {
                state.first.add(generation);
                this.port.postMessage({ type: "benchmark.playback.first", generation_id: generation, frame: currentFrame + first, sample_rate: sampleRate });
            }
            if (currentFrame - state.activityAt >= sampleRate / 50) {
                state.activityAt = currentFrame;
                this.port.postMessage({ type: "benchmark.playback.activity", generation_id: generation, frame: state.last, sample_rate: sampleRate, queued_samples: this.buffer.bufferedSamples });
            }
        }
        const underflow = Boolean(generation && state.first.has(generation) && queued === 0);
        if (underflow !== state.underflow) this.port.postMessage({ type: "benchmark.playback.underflow", generation_id: generation, frame: currentFrame, sample_rate: sampleRate, active: underflow });
        state.underflow = underflow;
        return result;
    }
}

registerProcessor("vox-capture", MeasVoxCapture);
registerProcessor("vox-playback", MeasVoxPlayback);
