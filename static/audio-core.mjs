// Чистая обработка PCM; тот же код используется в AudioWorklet и Node-проверках.
export const INPUT_RATE = 16000;
export const OUTPUT_RATE = 24000;
export const MAX_BUFFER_SAMPLES = OUTPUT_RATE * 2;

function encodeFrame(values) {
    const buffer = new ArrayBuffer(values.length * 2);
    const view = new DataView(buffer);
    for (let i = 0; i < values.length; i++) view.setInt16(i * 2, values[i], true);
    return buffer;
}

export class CaptureResampler {
    constructor(sourceRate) {
        if (!Number.isInteger(sourceRate) || sourceRate < INPUT_RATE) throw new Error("Неподдерживаемая частота микрофона");
        this.sourceRate = sourceRate;
        this.history = new Float64Array(63);
        this.coefficients = new Float64Array(63);
        // Фильтр перед понижением частоты подавляет частоты выше полосы будущего сигнала.
        const cutoff = 0.45 * INPUT_RATE / sourceRate;
        for (let i = 0; i < 63; i++) {
            const distance = i - 31;
            const sinc = distance === 0 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * distance) / (Math.PI * distance);
            this.coefficients[i] = sinc * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / 62));
        }
        const total = this.coefficients.reduce((sum, value) => sum + value, 0);
        for (let i = 0; i < 63; i++) this.coefficients[i] /= total;
        this.historyPosition = 0;
        this.windowUnits = 0;
        this.windowValue = 0;
        this.frame = [];
    }

    push(input) {
        const frames = [];
        for (const sample of input) {
            this.history[this.historyPosition] = Number.isFinite(sample) ? sample : 0;
            let filtered = 0;
            for (let i = 0; i < 63; i++) filtered += this.coefficients[i] * this.history[(this.historyPosition - i + 63) % 63];
            this.historyPosition = (this.historyPosition + 1) % 63;
            // Целочисленная шкала сохраняет фазу между блоками и точное число отсчётов на 44,1/48 кГц.
            let remaining = INPUT_RATE;
            while (remaining > 0) {
                const part = Math.min(remaining, this.sourceRate - this.windowUnits);
                this.windowValue += filtered * part;
                this.windowUnits += part;
                remaining -= part;
                if (this.windowUnits === this.sourceRate) {
                    const value = Math.max(-1, Math.min(1, this.windowValue / this.sourceRate));
                    this.frame.push(Math.round(value * (value < 0 ? 32768 : 32767)));
                    this.windowUnits = 0;
                    this.windowValue = 0;
                    if (this.frame.length === 320) {
                        frames.push(encodeFrame(this.frame));
                        this.frame = [];
                    }
                }
            }
        }
        return frames;
    }

    flush() {
        if (!this.frame.length) return [];
        while (this.frame.length < 320) this.frame.push(0);
        const result = [encodeFrame(this.frame)];
        this.frame = [];
        return result;
    }
}

export class PlaybackBuffer {
    constructor(deviceRate) {
        if (!Number.isFinite(deviceRate) || deviceRate < OUTPUT_RATE) throw new Error("Неподдерживаемая частота воспроизведения");
        this.deviceRate = deviceRate;
        this.clear();
    }

    clear() {
        this.generation = null;
        this.queue = [];
        this.bufferedSamples = 0;
        this.phase = 0;
        this.played = new Map();
    }

    start(generation) {
        this.clear();
        this.generation = generation;
    }

    push(header, buffer) {
        if (header.generation_id !== this.generation) return false;
        if (!(buffer instanceof ArrayBuffer) || buffer.byteLength % 2 || !buffer.byteLength) throw new Error("Неверный формат выходного PCM");
        const samples = buffer.byteLength / 2;
        if (this.bufferedSamples + samples > MAX_BUFFER_SAMPLES) throw new Error("Переполнен двухсекундный буфер воспроизведения");
        const pcm = new Int16Array(samples);
        const view = new DataView(buffer);
        for (let i = 0; i < samples; i++) pcm[i] = view.getInt16(i * 2, true);
        this.queue.push({ pcm, cursor: 0, segment: header.segment_id });
        this.bufferedSamples += samples;
        return true;
    }

    render(output) {
        output.fill(0);
        const changed = new Map();
        for (let i = 0; i < output.length && this.queue.length; i++) {
            const head = this.queue[0];
            const current = head.pcm[head.cursor];
            // Интерполяция продолжается через границу соседних PCM-кадров без сброса фазы.
            const next = head.cursor + 1 < head.pcm.length ? head.pcm[head.cursor + 1] : (this.queue[1]?.pcm[0] ?? current);
            output[i] = (current + (next - current) * this.phase) / 32768;
            this.phase += OUTPUT_RATE / this.deviceRate;
            while (this.phase >= 1 - 1e-12 && this.queue.length) {
                this.phase = Math.max(0, this.phase - 1);
                const item = this.queue[0];
                item.cursor++;
                this.bufferedSamples--;
                this.played.set(item.segment, (this.played.get(item.segment) ?? 0) + 1);
                changed.set(item.segment, this.played.get(item.segment));
                if (item.cursor === item.pcm.length) this.queue.shift();
            }
        }
        return [...changed].map(([segment_id, played_samples]) => ({ generation_id: this.generation, segment_id, played_samples }));
    }
}

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export class AudioReceiver {
    constructor() {
        this.reset();
    }

    reset() {
        this.pending = null;
        this.generation = null;
        this.sequences = new Map();
        this.receivedSamples = new Map();
        this.ended = new Set();
    }

    start(generation) {
        if (!UUID.test(generation)) throw new Error("Неверный идентификатор генерации");
        this.generation = generation;
        this.sequences.clear();
        this.receivedSamples.clear();
        this.ended.clear();
    }

    cancel() {
        this.generation = null;
        this.sequences.clear();
        this.receivedSamples.clear();
        this.ended.clear();
        // Уже принятый заголовок сохраняется: следующий binary нужно потребить и отбросить вместе с ним.
    }

    header(value) {
        if (this.pending) throw new Error("Между заголовком и PCM пришёл другой аудиозаголовок");
        if (value.version !== 1 || value.type !== "response.audio" || !UUID.test(value.generation_id) || !UUID.test(value.segment_id)
            || value.sample_rate !== OUTPUT_RATE || !Number.isInteger(value.byte_length) || value.byte_length <= 0 || value.byte_length % 2
            || value.byte_length > MAX_BUFFER_SAMPLES * 2 || !Number.isInteger(value.sequence) || value.sequence < 0) throw new Error("Неверный заголовок аудио");
        if (value.generation_id === this.generation) {
            if (this.ended.has(value.segment_id)) throw new Error("Аудио получено после завершения сегмента");
            const expected = this.sequences.get(value.segment_id) ?? 0;
            if (value.sequence !== expected) throw new Error("Нарушен порядок аудиокадров");
            if (!this.sequences.has(value.segment_id) && this.sequences.size >= 128) throw new Error("Слишком много аудиосегментов");
            this.sequences.set(value.segment_id, expected + 1);
        }
        // Одно ожидающее binary-сообщение соответствует ровно одному заголовку, включая устаревшую генерацию.
        this.pending = value;
    }

    binary(buffer) {
        const header = this.pending;
        this.pending = null;
        if (!header) throw new Error("PCM получен без аудиозаголовка");
        if (!(buffer instanceof ArrayBuffer) || buffer.byteLength !== header.byte_length) throw new Error("Размер PCM не совпадает с заголовком");
        if (header.generation_id !== this.generation) return null;
        this.receivedSamples.set(header.segment_id, (this.receivedSamples.get(header.segment_id) ?? 0) + buffer.byteLength / 2);
        return { header, buffer };
    }

    endSegment(value) {
        if (value.generation_id !== this.generation) return;
        if (!Number.isInteger(value.total_samples) || value.total_samples < 0 || value.total_samples !== (this.receivedSamples.get(value.segment_id) ?? 0)) {
            throw new Error("Число отсчётов завершённого сегмента не совпадает с PCM");
        }
        this.ended.add(value.segment_id);
    }
}
