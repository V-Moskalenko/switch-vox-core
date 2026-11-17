import assert from "node:assert/strict";
import test from "node:test";

import { AudioReceiver, CaptureResampler, PlaybackBuffer } from "../audio-core.mjs";

const generation = "11111111-1111-4111-8111-111111111111";
const segment = "22222222-2222-4222-8222-222222222222";

test("48 kHz input becomes 16 kHz PCM without losing phase across worklet blocks", () => {
    const converter = new CaptureResampler(48000);
    const frames = [];
    const source = Float32Array.from({ length: 48000 }, (_, i) => 0.5 * Math.sin(2 * Math.PI * 1000 * i / 48000));
    for (let offset = 0; offset < source.length; offset += 128) frames.push(...converter.push(source.subarray(offset, offset + 128)));
    const pcm = Int16Array.from(frames.flatMap((frame) => [...new Int16Array(frame)]));
    assert.equal(pcm.length, 16000);
    let rising = 0;
    for (let i = 1601; i < pcm.length; i++) if (pcm[i - 1] <= 0 && pcm[i] > 0) rising++;
    assert.ok(Math.abs(rising - 900) <= 1);
    assert.ok(Math.max(...pcm.slice(1600)) > 14000);
});

test("44.1 kHz capture keeps exact long-running sample count and flush pads only final frame", () => {
    const converter = new CaptureResampler(44100);
    let samples = 0;
    for (let offset = 0; offset < 44100; offset += 128) {
        for (const frame of converter.push(new Float32Array(Math.min(128, 44100 - offset)).fill(0.2))) samples += frame.byteLength / 2;
    }
    assert.equal(samples, 16000);
    converter.push(new Float32Array(441));
    const tail = converter.flush();
    assert.equal(tail.length, 1);
    assert.equal(tail[0].byteLength, 640);
});

test("downsampling attenuates 12 kHz source instead of aliasing it into voice band", () => {
    const converter = new CaptureResampler(48000);
    const frames = converter.push(Float32Array.from({ length: 4800 }, (_, i) => Math.sin(2 * Math.PI * 12000 * i / 48000)));
    const pcm = Int16Array.from(frames.flatMap((frame) => [...new Int16Array(frame)]));
    const rms = Math.sqrt(pcm.slice(320).reduce((sum, value) => sum + value * value, 0) / (pcm.length - 320)) / 32768;
    assert.ok(rms < 0.02, `RMS=${rms}`);
});

test("24 kHz PCM playback keeps one-second duration at a 48 kHz audio device", () => {
    const playback = new PlaybackBuffer(48000);
    playback.start(generation);
    playback.push({ generation_id: generation, segment_id: segment }, new Int16Array(24000).fill(16384).buffer);
    let audible = 0;
    let finalProgress;
    for (let frame = 0; frame < 48000; frame += 128) {
        const output = new Float32Array(Math.min(128, 48000 - frame));
        const progress = playback.render(output);
        audible += output.filter((sample) => sample > 0.49).length;
        if (progress.length) finalProgress = progress.at(-1);
    }
    assert.equal(audible, 48000);
    assert.equal(finalProgress.played_samples, 24000);
    assert.equal(playback.bufferedSamples, 0);
});

test("cancel removes scheduled audio and rejects stale generation chunks", () => {
    const playback = new PlaybackBuffer(48000);
    playback.start(generation);
    playback.push({ generation_id: generation, segment_id: segment }, new Int16Array(2400).fill(10000).buffer);
    playback.clear();
    assert.equal(playback.push({ generation_id: generation, segment_id: segment }, new Int16Array(100).buffer), false);
    const output = new Float32Array(128);
    assert.deepEqual(playback.render(output), []);
    assert.ok(output.every((sample) => sample === 0));
});

test("24 kHz playback preserves a 1 kHz tone at a 44.1 kHz device", () => {
    const playback = new PlaybackBuffer(44100);
    playback.start(generation);
    const source = Int16Array.from({ length: 24000 }, (_, i) => Math.round(12000 * Math.sin(2 * Math.PI * 1000 * i / 24000)));
    playback.push({ generation_id: generation, segment_id: segment }, source.buffer);
    const output = new Float32Array(44100);
    playback.render(output);
    let rising = 0;
    for (let i = 1; i < output.length; i++) if (output[i - 1] <= 0 && output[i] > 0) rising++;
    assert.ok(Math.abs(rising - 1000) <= 1);
    assert.equal(playback.bufferedSamples, 0);
});

test("playback rejects over two seconds without silently dropping samples", () => {
    const playback = new PlaybackBuffer(48000);
    playback.start(generation);
    playback.push({ generation_id: generation, segment_id: segment }, new Int16Array(48000).buffer);
    assert.throws(() => playback.push({ generation_id: generation, segment_id: segment }, new Int16Array(1).buffer), /буфер/i);
});

test("audio receiver enforces header/binary pairing and per-segment sequence", () => {
    const receiver = new AudioReceiver();
    receiver.start(generation);
    const header = { version: 1, type: "response.audio", generation_id: generation, segment_id: segment, sequence: 0, byte_length: 4, sample_rate: 24000 };
    receiver.header(header);
    assert.throws(() => receiver.header(header), /аудио/i);
    assert.throws(() => receiver.binary(new ArrayBuffer(2)), /размер/i);
    receiver.reset();
    receiver.start(generation);
    receiver.header(header);
    assert.equal(receiver.binary(new ArrayBuffer(4)).header.sequence, 0);
    assert.throws(() => receiver.header({ ...header, sequence: 2 }), /поряд/i);
});

test("cancelled header/payload pair is consumed but never returned for playback", () => {
    const receiver = new AudioReceiver();
    receiver.start(generation);
    receiver.header({ version: 1, type: "response.audio", generation_id: generation, segment_id: segment, sequence: 0, byte_length: 2, sample_rate: 24000 });
    receiver.cancel();
    assert.equal(receiver.binary(new ArrayBuffer(2)), null);
});

test("segment completion verifies total samples and forbids late audio for that segment", () => {
    const receiver = new AudioReceiver();
    receiver.start(generation);
    const header = { version: 1, type: "response.audio", generation_id: generation, segment_id: segment, sequence: 0, byte_length: 4, sample_rate: 24000 };
    receiver.header(header);
    receiver.binary(new ArrayBuffer(4));
    assert.throws(() => receiver.endSegment({ generation_id: generation, segment_id: segment, total_samples: 3 }), /отсч/i);
    receiver.endSegment({ generation_id: generation, segment_id: segment, total_samples: 2 });
    assert.throws(() => receiver.header({ ...header, sequence: 1 }), /заверш/i);
});
