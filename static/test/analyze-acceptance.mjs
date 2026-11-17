import { readFile, writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { captureBounds, nearestRank } from "./browser-latency.mjs";

export function analyzeBundle(bundle, warmups = 0) {
    const { snapshot, wire, acknowledgements = [] } = bundle;
    const events = snapshot.events;
    const rows = snapshot.turns.map((turn, index) => {
        const generation = turn.generation_id;
        const row = { index: index + 1, warmup: index < warmups, generation_id: generation,
            transcript: wire.find((event) => event.type === "transcript.final" && event.generation_id === generation)?.text,
            answer: wire.filter((event) => event.type === "response.text" && event.generation_id === generation).map((event) => event.text).join(" ") };
        const response = events.find((event) => event.type === "socket.response.started" && event.generation_id === generation);
        const first = events.find((event) => event.type === "benchmark.playback.first" && event.generation_id === generation);
        const ends = events.filter((event) => event.type === "benchmark.capture.end" && event.received_ms >= turn.performance_ms
            && event.received_ms <= (response?.performance_ms ?? -1) && event.received_ms < (snapshot.turns[index + 1]?.performance_ms ?? Infinity));
        const end = ends.at(-1);
        const segments = wire.filter((event) => event.type === "response.segment.end" && event.generation_id === generation);
        const textSegments = wire.filter((event) => event.type === "response.text" && event.generation_id === generation);
        row.full_playback = textSegments.length > 0 && textSegments.every((text) => segments.some((segment) => segment.segment_id === text.segment_id
            && segment.total_samples > 0 && acknowledgements.some((ack) => ack.generation_id === generation
                && ack.segment_id === segment.segment_id && ack.played_samples === segment.total_samples)));
        row.done = events.some((event) => event.type === "socket.response.done" && event.generation_id === generation);
        try {
            if (!end || !first || !Number.isFinite(first.output_ms)) throw new Error(first?.output_error || "Missing capture end or first played audio");
            const capture = captureBounds(end.frame / end.sample_rate, snapshot.probes, end.sample_rate);
            row.latency_ms = first.output_ms - capture.estimate_ms;
            row.calibration_lower_ms = first.output_ms - capture.upper_ms;
            row.calibration_upper_ms = first.output_ms - capture.lower_ms;
            row.capture_calibration = capture;
            row.capture_frame = end.frame;
            row.first_played_frame = first.frame;
            if (row.latency_ms < 0) throw new Error("Negative end-to-end latency");
        } catch (error) { row.timing_error = error.message; }
        return row;
    });
    const measured = rows.filter((row) => !row.warmup);
    const valid = measured.filter((row) => !row.timing_error);
    const quantile = (key, p) => valid.length ? nearestRank(valid.map((row) => row[key]), p) : null;
    const seriesComplete = bundle.result.completed && rows.length === bundle.result.expected && rows.length > 0
        && new Set(rows.map((row) => row.generation_id)).size === rows.length && !snapshot.overflow
        && new Set(valid.map((row) => row.capture_frame)).size === valid.length
        && !bundle.result.errors.length && !bundle.result.protocol_errors.length
        && valid.length === measured.length && measured.length > 0 && rows.every((row) => row.done && row.full_playback);
    const acceptanceComplete = seriesComplete && warmups >= 3 && measured.length >= 30;
    return { run: bundle.result, warmups, attempts: measured.length, valid_timings: valid.length, timing_failures: measured.length - valid.length,
        series_complete: Boolean(seriesComplete), acceptance_complete: Boolean(acceptanceComplete),
        acceptance_latency_pass: Boolean(acceptanceComplete && quantile("latency_ms", 0.95) <= 2000),
        all_full_playback: rows.every((row) => row.full_playback), all_done: rows.every((row) => row.done),
        p50_ms: quantile("latency_ms", 0.5), p95_ms: quantile("latency_ms", 0.95),
        p95_calibration_lower_ms: quantile("calibration_lower_ms", 0.95), p95_calibration_upper_ms: quantile("calibration_upper_ms", 0.95),
        scope: "Browser capture to UA-estimated device output; calibration interval excludes unknown hardware and UA timestamp uncertainty", rows };
}

export function analyzeInterruption({ snapshot, result }, pairIndex = 0) {
    const [oldTurn, nextTurn] = snapshot.turns.slice(pairIndex, pairIndex + 2);
    if (!oldTurn || !nextTurn) return { qualified: false, reason: "Two confirmed inputs required" };
    const events = snapshot.events;
    const clear = events.find((event) => event.type === "benchmark.playback.clear" && event.generation_id === oldTurn.generation_id
        && event.received_ms >= nextTurn.performance_ms - 10);
    const onset = events.filter((event) => event.type === "benchmark.capture.start" && event.received_ms <= nextTurn.performance_ms).at(-1);
    const activity = events.filter((event) => event.type === "benchmark.playback.activity" && event.generation_id === oldTurn.generation_id
        && event.frame < (clear?.frame ?? -1)).at(-1);
    try {
        if (!clear || !onset || !activity || clear.queued_samples <= 0 || !Number.isFinite(clear.output_ms)) throw new Error("No timed clear of actively queued old audio");
        if ((clear.frame - activity.frame) / clear.sample_rate > 0.1) throw new Error("Old playback was not recently active");
        const capture = captureBounds(onset.frame / onset.sample_rate, snapshot.probes, onset.sample_rate);
        const stale = events.some((event) => event.type === "benchmark.playback.activity" && event.generation_id === oldTurn.generation_id && event.frame >= clear.frame);
        const stop = clear.output_ms - capture.estimate_ms;
        const cancelled = events.find((event) => event.type === "socket.response.cancelled" && event.generation_id === oldTurn.generation_id);
        const notified = Math.min(nextTurn.performance_ms, cancelled?.performance_ms ?? Infinity);
        return { pair_index: pairIndex, qualified: !snapshot.overflow && !result.errors.length && !result.protocol_errors.length && !stale && stop >= 0,
            stop_ms: stop, calibration_lower_ms: clear.output_ms - capture.upper_ms,
            calibration_upper_ms: clear.output_ms - capture.lower_ms, queued_samples: clear.queued_samples,
            speech_onset_to_notification_ms: notified - capture.estimate_ms, notification_to_estimated_clear_ms: clear.output_ms - notified,
            observed_old_activity_after_clear: stale, activity_observation: "50 Hz; shorter bursts are not ruled out",
            whole_path_under_200_ms: !snapshot.overflow && !result.errors.length && !result.protocol_errors.length && !stale && stop >= 0 && stop <= 200, capture_calibration: capture };
    } catch (error) { return { qualified: false, reason: error.message }; }
}

export function analyzeStages(bundle, warmups = 0) {
    const baseline = analyzeBundle(bundle, warmups);
    const rows = baseline.rows.filter((row) => !row.warmup).map((row) => {
        const selected = bundle.snapshot.events.filter((event) => event.generation_id === row.generation_id);
        const find = (type) => selected.find((event) => event.type === type);
        const types = ["socket.response.started", "socket.transcript.final", "socket.response.text", "socket.response.audio"];
        const stamps = types.map((type) => find(type)?.performance_ms);
        const first = find("benchmark.playback.first");
        try {
            if (stamps.some((value) => !Number.isFinite(value)) || !Number.isFinite(first?.output_ms)) throw new Error("Missing browser-observed stage");
            if (find("socket.response.text").segment_id !== find("socket.response.audio").segment_id) throw new Error("First text/audio segment mismatch");
            const points = [...stamps, first.output_ms];
            const durations = points.slice(1).map((value, index) => value - points[index]);
            if (durations.some((value) => value < 0)) throw new Error("Non-monotonic browser-observed stages");
            const [asr, phrase, tts, playback] = durations;
            const endpoint = row.timing_error ? null : stamps[0] - row.capture_calibration.estimate_ms;
            if (endpoint !== null && endpoint < 0) throw new Error("Negative observed endpoint interval");
            return { generation_id: row.generation_id, endpoint_ms: endpoint, asr_final_ms: asr, llm_phrase_ms: phrase, tts_first_header_ms: tts,
                header_to_played_ms: playback, total_ms: row.timing_error ? null : row.latency_ms, capture_error: row.timing_error,
                sum_error_ms: row.timing_error ? null : endpoint + durations.reduce((sum, value) => sum + value, 0) - row.latency_ms };
        } catch (error) { return { generation_id: row.generation_id, error: error.message }; }
    });
    const valid = rows.filter((row) => !row.error);
    const fields = ["endpoint_ms", "asr_final_ms", "llm_phrase_ms", "tts_first_header_ms", "header_to_played_ms", "total_ms"];
    const observations = Object.fromEntries(fields.map((key) => [key, valid.map((row) => row[key]).filter(Number.isFinite)]));
    const median = Object.fromEntries(fields.map((key) => [key, observations[key].length ? nearestRank(observations[key], 0.5) : null]));
    return { scope: "Browser-observed pipeline intervals include dispatch/transport; not pure server component durations", warmups,
        attempts: rows.length, valid: valid.length, calibrated: valid.filter((row) => !row.capture_error).length,
        failures: rows.length - valid.length, capture_failures: valid.filter((row) => row.capture_error).length,
        sample_counts: Object.fromEntries(fields.map((key) => [key, observations[key].length])), median, rows };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
    const bundle = JSON.parse(await readFile(process.argv[2], "utf8"));
    const result = analyzeBundle(bundle, Number(process.argv[3] || 0));
    if (process.argv[4] === "barge") result.interruptions = bundle.snapshot.turns.slice(1).map((_, index) => analyzeInterruption(bundle, index));
    if (process.argv[4] === "stages") result.stages = analyzeStages(bundle, Number(process.argv[3] || 0));
    await writeFile(process.argv[2].replace(/\.json$/, "-analysis.json"), JSON.stringify(result, null, 2));
    console.log(JSON.stringify({ ...result, rows: undefined }, null, 2));
}
