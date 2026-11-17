// Измеритель задержек для тестов: захват входа и вывод устройством используют отдельные часы.
export function firstNonzero(samples, threshold = 0) {
    for (let i = 0; i < samples.length; i++) if (Math.abs(samples[i]) > threshold) return i;
    return -1;
}

export function mapOutput(contextTime, timestamp) {
    if (!Number.isFinite(contextTime) || contextTime < 0 || !timestamp || !Number.isFinite(timestamp.contextTime)
        || timestamp.contextTime < 0 || !Number.isFinite(timestamp.performanceTime) || timestamp.performanceTime <= 0) throw new Error("Invalid output timestamp");
    return timestamp.performanceTime + (contextTime - timestamp.contextTime) * 1000;
}

export function captureBounds(contextTime, probes, sampleRate, quantumFrames = 128, maxRttMs = 10, maxGapSeconds = 0.2) {
    if (!Number.isFinite(contextTime) || !Number.isFinite(sampleRate) || sampleRate <= 0 || !Number.isFinite(quantumFrames) || quantumFrames <= 0) throw new Error("Invalid capture probe parameters");
    const valid = probes.filter((probe) => Number.isFinite(probe.context_time) && Number.isFinite(probe.sent_ms) && Number.isFinite(probe.received_ms)
        && probe.received_ms >= probe.sent_ms && probe.received_ms - probe.sent_ms <= maxRttMs && Math.abs(probe.context_time - contextTime) <= maxGapSeconds);
    valid.sort((a, b) => Math.abs(a.context_time - contextTime) - Math.abs(b.context_time - contextTime));
    const probe = valid[0];
    if (!probe) throw new Error("No nearby valid capture probe");
    const delta = (contextTime - probe.context_time) * 1000;
    const quantumMs = (probe.quantum_frames ?? quantumFrames) * 1000 / sampleRate;
    const offsets = valid.map((item) => ({ lower: item.sent_ms - item.context_time * 1000, upper: item.received_ms - item.context_time * 1000,
        middle: (item.sent_ms + item.received_ms) / 2 - item.context_time * 1000 }));
    const spread = Math.max(...offsets.map((item) => item.middle)) - Math.min(...offsets.map((item) => item.middle));
    // Асимметричный RTT сдвигает середины проб: несовместимые интервалы неопределённости отклоняются.
    if (Math.max(...offsets.map((item) => item.lower)) - quantumMs > Math.min(...offsets.map((item) => item.upper)) + quantumMs) {
        throw new Error("Capture clock discontinuity between nearby probes");
    }
    // Учитываем измеренный разброс планирования, не считая ближайшую пробу точным временем.
    return { estimate_ms: (probe.sent_ms + probe.received_ms) / 2 + delta,
        lower_ms: contextTime * 1000 + Math.min(...offsets.map((item) => item.lower)) - quantumMs,
        upper_ms: contextTime * 1000 + Math.max(...offsets.map((item) => item.upper)) + quantumMs,
        probe_id: probe.id, rtt_ms: probe.received_ms - probe.sent_ms, gap_ms: Math.abs(delta), quantum_ms: quantumMs,
        offset_spread_ms: spread, nearby_probes: valid.length };
}

export function nearestRank(observations, percentile) {
    if (!observations.length || observations.some((value) => !Number.isFinite(value)) || !Number.isFinite(percentile) || percentile < 0 || percentile > 1) throw new Error("Invalid quantile observations");
    return [...observations].sort((a, b) => a - b)[Math.max(0, Math.ceil(percentile * observations.length) - 1)];
}

export function installLatencyObserver({ probeIntervalMs = 50, threshold = 0.001, silenceMs = 200, maxEvents = 100000 } = {}) {
    if (window.__voxMetrics) throw new Error("Latency observer already installed");
    const NativeNode = window.AudioWorkletNode;
    const NativeSocket = window.WebSocket;
    const nodes = [];
    const pending = new Map();
    const events = [];
    const probes = [];
    const turns = [];
    const firstHeaders = new Set();
    let sequence = 0;
    let stopped = false;
    let overflow = false;
    const add = (list, event) => {
        if (events.length + probes.length >= maxEvents) { overflow = true; return; }
        list.push(event);
    };
    const outputStamp = (context, frame, rate) => {
        const timestamp = context.getOutputTimestamp?.();
        const observed = performance.now();
        const raw = timestamp ? { contextTime: timestamp.contextTime, performanceTime: timestamp.performanceTime } : null;
        if (context.state !== "running") return { output_error: "AudioContext is not running", output_timestamp: raw };
        try {
            const value = mapOutput(frame / rate, raw);
            if (Math.abs(observed - raw.performanceTime) > 200 || Math.abs(frame / rate - raw.contextTime) > 0.2) throw new Error("Stale output timestamp");
            return { output_ms: value, output_timestamp: raw };
        } catch (error) { return { output_error: error.message, output_timestamp: raw }; }
    };
    const onWorklet = (node, name, { data }) => {
        if (stopped || typeof data.type !== "string" || !data.type.startsWith("benchmark.")) return;
        const received = performance.now();
        if (data.type === "benchmark.probe") {
            const sent = pending.get(data.id);
            pending.delete(data.id);
            if (sent !== undefined) add(probes, { id: data.id, context_time: data.context_time, sent_ms: sent, received_ms: received,
                quantum_frames: data.quantum_frames, sample_rate: data.sample_rate });
            return;
        }
        const item = { ...data, received_ms: received, node: name };
        if (data.type.startsWith("benchmark.playback.")) Object.assign(item, outputStamp(node.context, data.frame, data.sample_rate));
        add(events, item);
    };
    window.AudioWorkletNode = class extends NativeNode {
        constructor(context, name, options) {
            super(context, name, options);
            if (!["vox-capture", "vox-playback"].includes(name)) return;
            this.port.addEventListener("message", (event) => onWorklet(this, name, event));
            this.port.start();
            nodes.push({ node: this, name });
            if (name === "vox-capture") this.port.postMessage({ type: "benchmark.configure", threshold, silenceMs });
        }
    };
    window.WebSocket = class extends NativeSocket {
        constructor(...args) {
            super(...args);
            this.addEventListener("message", ({ data }) => {
                if (stopped || typeof data !== "string") return;
                const event = JSON.parse(data);
                if (event.type === "response.audio") {
                    if (firstHeaders.has(event.generation_id)) return;
                    firstHeaders.add(event.generation_id);
                }
                if (["input.started", "response.started", "response.cancelled", "response.done", "error", "transcript.final", "response.text", "response.audio"].includes(event.type)) {
                    const item = { type: `socket.${event.type}`, generation_id: event.generation_id, segment_id: event.segment_id,
                        turn_id: event.turn_id, code: event.code, performance_ms: performance.now() };
                    add(events, item);
                    if (event.type === "input.started") turns.push({ generation_id: event.generation_id, turn_id: event.turn_id, performance_ms: item.performance_ms });
                }
            });
        }
        send(data) {
            if (!stopped && typeof data === "string") {
                const event = JSON.parse(data);
                if (["input.start", "input.commit", "response.cancel"].includes(event.type)) add(events, {
                    type: `send.${event.type}`, generation_id: event.generation_id, performance_ms: performance.now(),
                });
            }
            return super.send(data);
        }
    };
    const timer = setInterval(() => {
        if (stopped) return;
        for (const { node, name } of nodes) if (name === "vox-capture" && node.context.state === "running") {
            const id = ++sequence;
            const sent = performance.now();
            pending.set(id, sent);
            node.port.postMessage({ type: "benchmark.probe", id });
        }
        for (const [id, sent] of pending) if (performance.now() - sent > 1000) pending.delete(id);
    }, probeIntervalMs);
    window.__voxMetrics = {
        events, probes, turns,
        mark(type, fields = {}) { add(events, { ...fields, type, performance_ms: performance.now() }); },
        snapshot() { return JSON.parse(JSON.stringify({ events, probes, turns, overflow, settings: { probeIntervalMs, threshold, silenceMs, maxEvents },
            contexts: nodes.map(({ node, name }) => ({ name, state: node.context.state, sample_rate: node.context.sampleRate, base_latency: node.context.baseLatency, output_latency: node.context.outputLatency })) })); },
        stop() { stopped = true; clearInterval(timer); pending.clear(); window.AudioWorkletNode = NativeNode; window.WebSocket = NativeSocket; return this.snapshot(); },
    };
    return window.__voxMetrics;
}
