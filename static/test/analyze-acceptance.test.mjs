import assert from "node:assert/strict";
import test from "node:test";

import { analyzeBundle } from "./analyze-acceptance.mjs";

function completeSeries() {
    const bundle = { result: { expected: 33, completed: true, errors: [], protocol_errors: [] },
        snapshot: { turns: [], events: [], probes: [], overflow: false }, wire: [], acknowledgements: [] };
    for (let index = 0; index < 33; index++) {
        const generation_id = String(index);
        const segment_id = `segment-${index}`;
        const start = index * 10000;
        const latency = index < 3 ? 5000 : 1500;
        bundle.snapshot.turns.push({ generation_id, performance_ms: start });
        bundle.snapshot.probes.push({ id: index, context_time: (start + 1000) / 1000, sent_ms: start + 999, received_ms: start + 1001 });
        bundle.snapshot.events.push(
            { type: "benchmark.capture.end", frame: (start + 1000) * 48, sample_rate: 48000, received_ms: start + 1200 },
            { type: "socket.response.started", generation_id, performance_ms: start + 1700 },
            { type: "benchmark.playback.first", generation_id, output_ms: start + 1000 + latency },
            { type: "socket.response.done", generation_id },
        );
        bundle.wire.push({ type: "response.text", generation_id, segment_id, text: "Москва." },
            { type: "response.segment.end", generation_id, segment_id, total_samples: 24000 });
        bundle.acknowledgements.push({ generation_id, segment_id, played_samples: 24000 });
    }
    return bundle;
}

test("a complete series excludes warmups only from latency quantiles", () => {
    const result = analyzeBundle(completeSeries(), 3);
    assert.equal(result.series_complete, true);
    assert.equal(result.acceptance_complete, true);
    assert.equal(result.acceptance_latency_pass, true);
    assert.equal(result.all_full_playback, true);
    assert.equal(result.all_done, true);
    assert.equal(result.attempts, 30);
    assert.equal(result.valid_timings, 30);
    assert.equal(result.p50_ms, 1500);
    assert.equal(result.p95_ms, 1500);
});

for (const [index, label] of [[0, "warmup"], [3, "measured turn"]]) {
    test(`a ${label} without its final playback ACK invalidates the full series`, () => {
        const bundle = completeSeries();
        bundle.acknowledgements = bundle.acknowledgements.filter((ack) => ack.generation_id !== String(index));
        const result = analyzeBundle(bundle, 3);
        assert.equal(result.rows[index].full_playback, false);
        assert.equal(result.series_complete, false);
        assert.equal(result.acceptance_complete, false);
        assert.equal(result.acceptance_latency_pass, false);
        assert.equal(result.all_full_playback, false);
        assert.equal(result.p95_ms, 1500);
    });

    test(`a ${label} without response.done invalidates the full series`, () => {
        const bundle = completeSeries();
        bundle.snapshot.events = bundle.snapshot.events.filter((event) => event.type !== "socket.response.done" || event.generation_id !== String(index));
        const result = analyzeBundle(bundle, 3);
        assert.equal(result.rows[index].done, false);
        assert.equal(result.series_complete, false);
        assert.equal(result.acceptance_complete, false);
        assert.equal(result.acceptance_latency_pass, false);
        assert.equal(result.all_done, false);
        assert.equal(result.p95_ms, 1500);
    });
}
