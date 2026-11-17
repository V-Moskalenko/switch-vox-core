import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

import { captureBounds, firstNonzero, mapOutput, nearestRank } from "./browser-latency.mjs";
import { CaptureResampler, PlaybackBuffer } from "../audio-core.mjs";

test("output timestamp includes device delay without adding it twice", () => {
    assert.equal(mapOutput(2, { contextTime: 1.9, performanceTime: 2100 }), 2200);
    const capture = captureBounds(2, [{ id: 1, context_time: 2, sent_ms: 1999, received_ms: 2001 }], 48000);
    assert.equal(2200 - capture.estimate_ms, 200);
    assert.ok(capture.lower_ms < 2000 && capture.upper_ms > 2000);
});

test("capture clock uses nearest probe with bounded RTT and retains quantum uncertainty", () => {
    const result = captureBounds(3.01, [
        { id: 1, context_time: 3, sent_ms: 2999, received_ms: 3003 },
        { id: 2, context_time: 2.9, sent_ms: 2900, received_ms: 2901 },
    ], 48000);
    assert.equal(result.probe_id, 1);
    assert.ok(Math.abs(result.estimate_ms - 3011) < 1e-9);
    assert.ok(Math.abs(result.lower_ms - (3009 - 128 / 48)) < 1e-9);
    assert.ok(Math.abs(result.upper_ms - (3013 + 128 / 48)) < 1e-9);
    assert.throws(() => captureBounds(3, [{ context_time: 3, sent_ms: 0, received_ms: 11 }], 48000), /probe/);
    assert.throws(() => captureBounds(4, [{ context_time: 3, sent_ms: 0, received_ms: 1 }], 48000), /probe/);
});

test("unsupported timestamps and clock reversal cannot produce passing latency", () => {
    for (const timestamp of [null, { contextTime: 0, performanceTime: 0 }, { contextTime: NaN, performanceTime: 2 }]) {
        assert.throws(() => mapOutput(1, timestamp), /timestamp/);
    }
    assert.throws(() => captureBounds(1, [{ context_time: 1, sent_ms: 3, received_ms: 2 }], 48000), /probe/);
});

test("capture calibration includes nearby jitter and rejects a clock discontinuity", () => {
    const result = captureBounds(3, [
        { id: 1, context_time: 3, sent_ms: 2999, received_ms: 3001 },
        { id: 2, context_time: 2.9, sent_ms: 2897, received_ms: 2899 },
    ], 48000);
    assert.equal(result.offset_spread_ms, 2);
    assert.ok(Math.abs(result.lower_ms - (2997 - 128 / 48)) < 1e-9);
    assert.ok(Math.abs(result.upper_ms - (3001 + 128 / 48)) < 1e-9);
    assert.throws(() => captureBounds(3, [
        { id: 1, context_time: 3, sent_ms: 2999, received_ms: 3001 },
        { id: 2, context_time: 2.9, sent_ms: 2929, received_ms: 2931 },
    ], 48000), /discontinuity/);
});

test("asymmetric probe RTT retains compatible clock offsets and full uncertainty bounds", () => {
    const result = captureBounds(3, [
        { id: 1, context_time: 3, sent_ms: 3161, received_ms: 3161.2 },
        { id: 2, context_time: 2.9, sent_ms: 3050, received_ms: 3059.8 },
    ], 48000);
    assert.equal(result.probe_id, 1);
    assert.ok(result.offset_spread_ms > 2 * result.quantum_ms);
    assert.ok(result.lower_ms <= 3161 && result.upper_ms >= 3161);
    assert.ok(Math.abs(result.lower_ms - (3150 - 128 / 48)) < 1e-9);
    assert.ok(Math.abs(result.upper_ms - (3161.2 + 128 / 48)) < 1e-9);
});

test("capture offset intervals may touch but must share a common point", () => {
    const probes = [
        { id: 1, context_time: 3, sent_ms: 3000, received_ms: 3000 },
        { id: 2, context_time: 2.9, sent_ms: 2902, received_ms: 2902 },
    ];
    const result = captureBounds(3, probes, 128000);
    assert.equal(result.quantum_ms, 1);
    assert.ok(result.lower_ms <= 3001 && result.upper_ms >= 3001);
    probes[1] = { ...probes[1], sent_ms: 2902.001, received_ms: 2902.001 };
    assert.throws(() => captureBounds(3, probes, 128000), /discontinuity/);
});

test("first signal offset ignores leading zeros and samples below declared threshold", () => {
    assert.equal(firstNonzero(new Float32Array([0, 0, 0.0001, -0.02]), 0.001), 3);
    assert.equal(firstNonzero(new Float32Array(128), 0.001), -1);
});

test("nearest-rank quantiles are explicit and reject missing observations", () => {
    const values = Array.from({ length: 30 }, (_, i) => i + 1).reverse();
    assert.equal(nearestRank(values, 0.5), 15);
    assert.equal(nearestRank(values, 0.95), 29);
    assert.throws(() => nearestRank([], 0.95), /observations/);
    assert.throws(() => nearestRank([1, NaN], 0.95), /observations/);
});

function processors() {
    const registered = new Map();
    const scope = {
        currentFrame: 0, sampleRate: 24000, CaptureResampler, PlaybackBuffer,
        AudioWorkletProcessor: class {
            constructor() {
                const listeners = [];
                this.port = { messages: [], onmessage: null, start() {},
                    postMessage(data) { this.messages.push(data); },
                    addEventListener(_type, callback) { listeners.push(callback); },
                    dispatch(data) { const event = { data }; this.onmessage?.(event); for (const callback of listeners) callback(event); },
                };
            }
        },
        registerProcessor(name, value) { registered.set(name, value); },
    };
    const original = readFileSync(new URL("../audio-worklet.js", import.meta.url), "utf8")
        .replace(/^import .*;\r?\n/, "").replace(/^registerProcessor\(.*\);\r?$/gm, "");
    const appendix = readFileSync(new URL("./latency-observer.js", import.meta.url), "utf8");
    vm.runInNewContext(`${original}\n${appendix}`, scope);
    return { scope, Capture: registered.get("vox-capture"), Playback: registered.get("vox-playback") };
}

test("observer measures actual output offset and first clear quantum without changing render", () => {
    const { scope, Playback } = processors();
    const processor = new Playback();
    const generation = "11111111-1111-4111-8111-111111111111";
    processor.port.dispatch({ type: "playback.start", generation_id: generation });
    const pcm = new Int16Array(1000).fill(16384);
    pcm[0] = pcm[1] = 0;
    processor.port.dispatch({ type: "playback.audio", header: { generation_id: generation, segment_id: "segment" }, buffer: pcm.buffer });
    const first = new Float32Array(128);
    processor.process([], [[first]]);
    assert.equal(first[2], 0.5);
    const onset = processor.port.messages.find((event) => event.type === "benchmark.playback.first");
    assert.equal(onset.frame, 2);
    scope.currentFrame = 128;
    processor.port.dispatch({ type: "playback.clear" });
    const stopped = new Float32Array(128).fill(1);
    processor.process([], [[stopped]]);
    assert.ok(stopped.every((value) => value === 0));
    const clear = processor.port.messages.find((event) => event.type === "benchmark.playback.clear");
    assert.equal(clear.frame, 128);
    assert.equal(clear.generation_id, generation);
    assert.equal(clear.queued_samples, 872);
    assert.equal(clear.last_nonzero_frame, 127);
    processor.port.dispatch({ type: "playback.audio", header: { generation_id: generation, segment_id: "segment" }, buffer: pcm.buffer });
    scope.currentFrame = 256;
    processor.process([], [[stopped]]);
    assert.ok(stopped.every((value) => value === 0));
});

test("capture observer timestamps last active sample rather than silence confirmation", () => {
    const { scope, Capture } = processors();
    const processor = new Capture();
    processor.port.dispatch({ type: "capture.start" });
    processor.port.dispatch({ type: "benchmark.probe", id: 42 });
    processor.process([[new Float32Array(128)]]);
    const speech = new Float32Array(128);
    speech.fill(0.2, 20, 111);
    scope.currentFrame = 128;
    processor.process([[speech]]);
    for (let frame = 256; frame < 5500; frame += 128) {
        scope.currentFrame = frame;
        processor.process([[new Float32Array(128)]]);
    }
    const onset = processor.port.messages.find((event) => event.type === "benchmark.capture.start");
    const end = processor.port.messages.find((event) => event.type === "benchmark.capture.end");
    const probe = processor.port.messages.find((event) => event.type === "benchmark.probe");
    assert.equal(onset.frame, 148);
    assert.equal(end.frame, 239);
    assert.equal(end.reason, "silence");
    assert.equal(probe.id, 42);
    assert.equal(probe.context_time, 0);
    assert.ok(processor.port.messages.some((event) => event.type === "capture.frame"));
});
