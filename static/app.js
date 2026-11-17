import { AudioReceiver, MAX_BUFFER_SAMPLES } from "./audio-core.mjs";

class VoxClient {
    constructor() {
        this.ui = Object.fromEntries(["auto", "connect", "ptt", "ptt-label", "stop", "status", "error", "connection", "mode", "messages", "empty", "level"].map((id) => [id, document.getElementById(id)]));
        this.inputMode = "ptt";
        this.socket = null;
        this.context = null;
        this.stream = null;
        this.capture = null;
        this.playback = null;
        this.receiver = new AudioReceiver();
        this.ready = false;
        this.connecting = false;
        this.holding = false;
        this.recording = false;
        this.awaitingStart = false;
        this.flushing = false;
        this.turn = null;
        this.generating = false;
        this.receivedSamples = 0;
        this.acknowledgedSamples = 0;
        this.played = new Map();
        this.pendingProgress = [];
        this.assistantSegments = new Map();
        this.userMessage = null;
        this.assistantMessage = null;
        this.captureStartedAt = 0;
        this.bindControls();
        this.progressTimer = setInterval(() => this.reportPlayed(), 25);
    }

    bindControls() {
        this.ui.connect.addEventListener("click", () => this.socket || this.context ? this.disconnect() : this.connect());
        this.ui.auto.addEventListener("change", () => { this.inputMode = this.ui.auto.checked ? "auto" : "ptt"; this.updateControls(); });
        this.ui.stop.addEventListener("click", () => this.cancelAnswer());
        this.ui.ptt.addEventListener("pointerdown", (event) => {
            if (event.button !== 0 || this.ui.ptt.disabled) return;
            event.preventDefault();
            this.ui.ptt.focus({ preventScroll: true });
            this.ui.ptt.setPointerCapture(event.pointerId);
            this.startTurn();
        });
        for (const event of ["pointerup", "pointercancel", "lostpointercapture"]) this.ui.ptt.addEventListener(event, () => this.finishTurn());
        this.ui.ptt.addEventListener("keydown", (event) => {
            if (event.code !== "Enter") return;
            event.preventDefault();
            if (!event.repeat) this.startTurn();
        });
        // Пробел управляет PTT на странице, сохраняя обычный ввод в полях и других кнопках.
        window.addEventListener("keydown", (event) => {
            if (event.code !== "Space" || this.ui.ptt.disabled || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey || event.isComposing) return;
            const control = event.target instanceof Element ? event.target.closest("input, textarea, select, button, a, [contenteditable]") : null;
            if (control && control !== this.ui.ptt) return;
            event.preventDefault();
            if (!event.repeat) this.startTurn();
        });
        window.addEventListener("keyup", (event) => {
            if (["Space", "Enter"].includes(event.code) && this.holding) {
                event.preventDefault();
                this.finishTurn();
            }
        });
        window.addEventListener("blur", () => this.finishTurn());
        document.addEventListener("visibilitychange", () => { if (document.hidden) this.finishTurn(); });
        window.addEventListener("pagehide", () => this.disconnect());
    }

    status(message) {
        this.ui.status.textContent = message;
    }

    error(message) {
        this.ui.error.textContent = message;
        this.ui.error.hidden = false;
    }

    updateControls() {
        this.ui.connect.disabled = this.connecting;
        this.ui.connect.textContent = this.connecting ? "Подключение…" : (this.socket || this.context ? "Отключить" : "Подключить микрофон");
        this.ui.auto.disabled = this.connecting || this.ready;
        this.ui.ptt.disabled = !this.ready || this.flushing || this.inputMode === "auto";
        this.ui.ptt.setAttribute("aria-pressed", String(this.holding));
        this.ui.ptt.setAttribute("aria-label", this.holding ? "Идёт запись; отпустите для отправки" : "Удерживайте, чтобы говорить");
        this.ui["ptt-label"].innerText = this.inputMode === "auto" ? "Говорите\nбез нажатия" : (this.holding ? "Говорите…\nОтпустите для отправки" : "Удерживайте,\nчтобы говорить");
        this.ui.stop.disabled = !this.ready || (this.recording && this.inputMode !== "auto") || !(this.generating || this.receivedSamples > this.acknowledgedSamples);
        this.ui.connection.dataset.state = this.ready ? "online" : (this.connecting ? "connecting" : "offline");
        this.ui.connection.textContent = this.ready ? "Подключено" : (this.connecting ? "Подключение…" : "Не подключено");
    }

    async connect() {
        if (this.connecting || this.socket) return;
        console.debug("[vox:session.connect] Подключение микрофона и аудиоустройства");
        this.connecting = true;
        this.ui.error.hidden = true;
        this.status("Разрешите доступ к микрофону в браузере.");
        this.updateControls();
        try {
            if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) throw new Error("Микрофон доступен через localhost или защищённое HTTPS-соединение.");
            // Сначала получаем разрешение и открываем аудио; серверная сессия пока не занята.
            const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
            this.stream = stream;
            const context = new AudioContext({ latencyHint: "interactive" });
            this.context = context;
            await context.audioWorklet.addModule("/static/audio-worklet.js");
            await context.resume();
            const capture = new AudioWorkletNode(context, "vox-capture", { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1], channelCount: 1, channelCountMode: "explicit" });
            const playback = new AudioWorkletNode(context, "vox-playback", { numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [1] });
            this.capture = capture;
            this.playback = playback;
            const source = context.createMediaStreamSource(stream);
            // Беззвучная ветка поддерживает обработку микрофона без локального эха в динамиках.
            const silent = context.createGain();
            silent.gain.value = 0;
            source.connect(capture).connect(silent).connect(context.destination);
            playback.connect(context.destination);
            capture.port.onmessage = ({ data }) => { if (capture === this.capture) this.onCapture(data); };
            playback.port.onmessage = ({ data }) => { if (playback === this.playback) this.onPlayback(data); };
            for (const track of stream.getAudioTracks()) track.addEventListener("ended", () => {
                if (stream === this.stream) {
                    console.warn("[vox:microphone.ended] Устройство отключено или разрешение отозвано");
                    this.fail("Микрофон отключён или доступ отозван.");
                }
            });
            this.openSocket();
        } catch (error) {
            // В консоль передаём только фиксированные причины, без содержимого исключения.
            if (error.name === "NotAllowedError") console.warn("[vox:microphone.permission_denied] Доступ к микрофону запрещён");
            else console.error("[vox:session.connect_failed] Не удалось подготовить микрофон или аудиоустройство");
            const message = error.name === "NotAllowedError" ? "Доступ к микрофону запрещён. Разрешите его в настройках сайта и подключитесь снова."
                : error.name === "NotFoundError" ? "Микрофон не найден. Подключите устройство и повторите."
                    : error.name === "NotReadableError" ? "Не удалось открыть микрофон. Возможно, его использует другое приложение." : error.message;
            this.fail(message);
        }
    }

    openSocket() {
        const socket = new WebSocket(`${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/api/v1/session`);
        this.socket = socket;
        socket.binaryType = "arraybuffer";
        socket.onopen = () => {
            if (socket !== this.socket) return;
            console.debug("[vox:session.socket_open] Соединение открыто; согласование формата аудио");
            this.status("Согласование голосовой сессии…");
            this.send("session.start", { input_mode: this.inputMode, language: "ru", input_sample_rate: 16000, output_sample_rate: 24000, channels: 1, sample_width_bytes: 2 });
        };
        socket.onmessage = ({ data }) => {
            if (socket !== this.socket) return;
            try {
                if (typeof data === "string") this.onEvent(JSON.parse(data));
                else this.onAudio(data);
            } catch (error) {
                console.error("[vox:protocol.invalid_message] Получено недопустимое событие или аудиокадр");
                this.fail(`Ошибка аудиопротокола: ${error.message}`);
            }
        };
        socket.onerror = () => {
            if (socket === this.socket) {
                console.warn("[vox:session.socket_error] Ошибка соединения с сервером");
                this.error("Не удалось установить соединение с сервером.");
            }
        };
        socket.onclose = () => {
            if (socket === this.socket) this.disconnect("Соединение закрыто. Для нового разговора подключитесь снова.");
        };
        this.connectTimeout = setTimeout(() => {
            if (!this.ready && socket === this.socket) {
                console.warn("[vox:session.handshake_timeout] Сервер не подтвердил формат сессии вовремя");
                this.fail("Сервер не подтвердил сессию за 10 секунд.");
            }
        }, 10000);
    }

    send(type, fields = {}) {
        if (this.socket?.readyState !== WebSocket.OPEN) return false;
        this.socket.send(JSON.stringify({ version: 1, type, ...fields }));
        return true;
    }

    onEvent(event) {
        if (this.receiver.pending) throw new Error("Между аудиозаголовком и PCM получено JSON-событие");
        if (event.version !== 1 || typeof event.type !== "string") throw new Error("Неподдерживаемая версия события");
        if (event.type === "session.ready") {
            if (this.ready || (event.input_mode ?? "ptt") !== this.inputMode || !["cascade", "e2e"].includes(event.mode) || event.input_sample_rate !== 16000 || event.output_sample_rate !== 24000 || event.channels !== 1 || event.sample_width_bytes !== 2) {
                throw new Error("Сервер подтвердил неподдерживаемый формат сессии");
            }
            clearTimeout(this.connectTimeout);
            for (const entry of this.ui.messages.querySelectorAll("article")) entry.remove();
            this.ui.empty.hidden = false;
            this.userMessage = null;
            this.assistantMessage = null;
            this.assistantSegments.clear();
            // Только подтверждённый формат разрешает запись и команды разговора.
            this.connecting = false;
            this.ready = true;
            console.debug("[vox:session.ready] Формат голосовой сессии согласован");
            this.ui.mode.textContent = event.mode === "cascade" ? "Каскад" : "Сквозной режим";
            this.status("Можно говорить. Удерживайте кнопку микрофона.");
            if (this.inputMode === "auto") {
                this.recording = true;
                this.capture.port.postMessage({ type: "capture.start" });
                this.status("Слушаю автоматически. Пауза завершает реплику.");
            }
            this.updateControls();
            if (this.inputMode === "ptt") this.ui.ptt.focus({ preventScroll: true });
            return;
        }
        if (event.type === "error") {
            console.warn("[vox:session.command_rejected] Сервер отклонил команду или обработку реплики");
            this.holding = false;
            this.recording = false;
            this.flushing = false;
            this.awaitingStart = false;
            clearTimeout(this.turnTimeout);
            this.capture?.port.postMessage({ type: "capture.clear" });
            this.clearPlayback();
            this.error(event.message || "Сервер отклонил запрос.");
            this.status("Реплика не завершена. Можно попробовать снова.");
            if (this.inputMode === "auto" && this.ready) {
                this.recording = true;
                this.capture?.port.postMessage({ type: "capture.start" });
            }
            this.updateControls();
            return;
        }
        if (event.type === "input.started") {
            if (this.inputMode === "auto") {
                this.clearPlayback(false);
                this.userMessage = null;
                this.assistantMessage = null;
                this.assistantSegments.clear();
                this.turn = { turn_id: event.turn_id, generation_id: event.generation_id };
                this.receiver.start(event.generation_id);
                this.playback.port.postMessage({ type: "playback.start", generation_id: event.generation_id });
                this.status("Слушаю…");
                this.updateControls();
                return;
            }
            if (!this.awaitingStart) throw new Error("Неожиданное начало реплики");
            this.awaitingStart = false;
            this.turn = { turn_id: event.turn_id, generation_id: event.generation_id };
            this.receiver.start(event.generation_id);
            this.playback.port.postMessage({ type: "playback.start", generation_id: event.generation_id });
            if (!this.holding) {
                this.send("response.cancel", { generation_id: event.generation_id });
                this.clearPlayback();
                this.status("Нажатие слишком короткое. Удерживайте кнопку во время речи.");
            } else {
                this.recording = true;
                this.captureStartedAt = performance.now();
                this.capture.port.postMessage({ type: "capture.start" });
                this.status("Слушаю…");
                this.turnTimeout = setTimeout(() => { this.finishTurn(); this.status("Достигнут предел реплики — 30 секунд."); }, 30000);
            }
            this.updateControls();
            return;
        }
        if (event.type.startsWith("transcript.")) {
            if (event.generation_id !== this.turn?.generation_id || typeof event.text !== "string") return;
            if (!this.userMessage) this.userMessage = this.addMessage("user", "Вы");
            this.userMessage.content.textContent = event.text;
            this.userMessage.article.classList.toggle("partial", event.type !== "transcript.final");
            this.scrollMessages();
            return;
        }
        // Отменённый ответ больше не воспроизводим; его заголовок всё равно связывается с последующим PCM.
        if (event.generation_id !== this.receiver.generation) {
            if (event.type === "response.audio") this.receiver.header(event);
            return;
        }
        if (event.type === "response.started") {
            this.generating = true;
            this.status("Ассистент готовит ответ…");
        } else if (event.type === "response.text") {
            if (typeof event.text !== "string") throw new Error("Неверный текст ответа");
            this.assistantSegments.set(event.segment_id, event.text);
            if (!this.assistantMessage) this.assistantMessage = this.addMessage("assistant", "Ассистент");
            this.assistantMessage.content.textContent = [...this.assistantSegments.values()].join(" ");
            this.scrollMessages();
        } else if (event.type === "response.audio") {
            this.receiver.header(event);
        } else if (event.type === "response.segment.end") {
            this.receiver.endSegment(event);
        } else if (event.type === "response.done") {
            this.generating = false;
            this.status(this.receivedSamples > this.acknowledgedSamples ? "Воспроизведение ответа…" : "Ответ завершён. Можно говорить.");
        } else if (event.type === "response.cancelled") {
            this.clearPlayback(false);
            this.status("Ответ остановлен. Можно говорить.");
        } else {
            throw new Error(`Неизвестное событие ${event.type}`);
        }
        this.updateControls();
    }

    onAudio(buffer) {
        const item = this.receiver.binary(buffer);
        if (!item) return;
        const count = buffer.byteLength / 2;
        if (this.receivedSamples - this.acknowledgedSamples + count > MAX_BUFFER_SAMPLES) {
            console.warn("[vox:audio.playback_overflow] Исчерпан предел ожидающего воспроизведения аудио");
            this.cancelAnswer();
            this.error("Ответ остановлен: переполнен двухсекундный аудиобуфер.");
            return;
        }
        this.receivedSamples += count;
        this.playback.port.postMessage({ type: "playback.audio", header: item.header, buffer }, [buffer]);
        this.status("Ассистент отвечает…");
        this.updateControls();
    }

    onCapture(data) {
        if (data.type === "capture.frame" && this.recording) {
            if (this.socket?.readyState !== WebSocket.OPEN) return;
            if (this.socket.bufferedAmount + data.buffer.byteLength > 32000) {
                console.warn("[vox:audio.capture_backpressure] Отправка микрофона не успевает за записью");
                this.fail("Медленное соединение: микрофон остановлен, чтобы не накапливать задержку.");
                return;
            }
            this.socket.send(data.buffer);
        } else if (data.type === "capture.flushed" && this.flushing) {
            // Commit следует за последним кадром worklet, иначе сервер потеряет конец реплики.
            this.recording = false;
            this.flushing = false;
            this.send("input.commit", this.turn);
            this.generating = true;
            this.status("Реплика отправлена. Ожидаю ответ…");
            this.ui.level.value = 0;
            this.updateControls();
        } else if (data.type === "capture.level") this.ui.level.value = data.value;
    }

    onPlayback(data) {
        if (data.type === "playback.error") {
            console.error("[vox:audio.worklet_error] Аудиопроцессор отклонил данные воспроизведения");
            this.cancelAnswer();
            this.error(data.message);
            return;
        }
        if (data.type === "playback.progress") {
            for (const item of data.progress) if (item.generation_id === this.receiver.generation) this.pendingProgress.push({ ...item, context_time: data.context_time });
            if (this.pendingProgress.length > 2048) {
                console.warn("[vox:audio.progress_stalled] Не подтверждается продвижение воспроизведения");
                this.cancelAnswer();
                this.error("Браузер не подтверждает воспроизведение. Ответ остановлен.");
            }
        }
    }

    reportPlayed() {
        if (!this.context || !this.ready || !this.pendingProgress.length) return;
        // Подтверждаем услышанные отсчёты по времени устройства, а не по факту выдачи в worklet.
        const timestamp = this.context.getOutputTimestamp?.();
        const audibleTime = timestamp?.contextTime ?? Math.max(0, this.context.currentTime - (this.context.baseLatency || 0) - (this.context.outputLatency || 0));
        const latest = new Map();
        while (this.pendingProgress.length && this.pendingProgress[0].context_time <= audibleTime) {
            const item = this.pendingProgress.shift();
            if (item.generation_id === this.receiver.generation) latest.set(item.segment_id, item);
        }
        for (const item of latest.values()) {
            const previous = this.played.get(item.segment_id) ?? 0;
            if (item.played_samples <= previous) continue;
            this.played.set(item.segment_id, item.played_samples);
            this.acknowledgedSamples += item.played_samples - previous;
            this.send("playback.progress", { generation_id: item.generation_id, segment_id: item.segment_id, played_samples: item.played_samples });
        }
        if (!this.generating && this.receivedSamples && this.receivedSamples === this.acknowledgedSamples) this.status("Ответ завершён. Можно говорить.");
        this.updateControls();
    }

    async startTurn() {
        if (this.inputMode === "auto" || !this.ready || this.holding || this.awaitingStart || this.recording || this.flushing) return;
        this.holding = true;
        this.ui.error.hidden = true;
        this.updateControls();
        try {
            await this.context.resume();
            if (!this.holding || !this.ready) return;
            this.cancelAnswer(false);
            this.userMessage = null;
            this.assistantMessage = null;
            this.assistantSegments.clear();
            this.awaitingStart = true;
            this.send("input.start");
            this.status("Начинаю реплику…");
        } catch (error) {
            console.error("[vox:audio.resume_failed] Браузер не возобновил аудиоустройство");
            this.fail(`Не удалось включить аудио: ${error.message}`);
        }
    }

    finishTurn() {
        if (!this.holding) return;
        this.holding = false;
        clearTimeout(this.turnTimeout);
        if (this.recording) {
            this.flushing = true;
            this.capture.port.postMessage({ type: "capture.stop" });
        }
        this.updateControls();
    }

    clearPlayback(report = true) {
        if (report) this.reportPlayed();
        // Последнее подтверждение сохраняется до очистки, чтобы история учитывала услышанные фразы.
        this.receiver.cancel();
        this.playback?.port.postMessage({ type: "playback.clear" });
        this.pendingProgress = [];
        this.played.clear();
        this.receivedSamples = 0;
        this.acknowledgedSamples = 0;
        this.generating = false;
    }

    cancelAnswer(showStatus = true) {
        const generation = this.receiver.generation;
        this.clearPlayback();
        if (generation) this.send("response.cancel", { generation_id: generation });
        if (showStatus) this.status("Ответ остановлен. Можно говорить.");
        this.updateControls();
    }

    addMessage(role, label) {
        this.ui.empty.hidden = true;
        const article = document.createElement("article");
        article.className = `message ${role}`;
        const heading = document.createElement("div");
        heading.className = "message-label";
        heading.textContent = label;
        const content = document.createElement("div");
        content.className = "message-content";
        article.append(heading, content);
        this.ui.messages.append(article);
        const entries = this.ui.messages.querySelectorAll("article");
        if (entries.length > 50) entries[0].remove();
        return { article, content };
    }

    scrollMessages() {
        this.ui.messages.scrollTop = this.ui.messages.scrollHeight;
    }

    fail(message) {
        this.disconnect("Подключение завершено. Устраните причину и подключитесь снова.");
        this.error(message);
    }

    disconnect(message = "Сессия завершена. Микрофон отключён.") {
        if (this.socket || this.context || this.stream) console.debug("[vox:session.disconnect] Закрытие соединения и аудиоустройства");
        this.cancelAnswer(false);
        this.ready = false;
        this.connecting = false;
        this.holding = false;
        this.recording = false;
        this.flushing = false;
        this.awaitingStart = false;
        clearTimeout(this.turnTimeout);
        clearTimeout(this.connectTimeout);
        // Сначала отвязываем ресурсы от клиента; поздние callbacks прежней сессии будут проигнорированы.
        const socket = this.socket;
        this.socket = null;
        if (socket && socket.readyState < WebSocket.CLOSING) socket.close(1000, "Клиент завершил сессию");
        this.capture?.port.postMessage({ type: "capture.clear" });
        this.capture?.disconnect();
        this.playback?.disconnect();
        this.capture = null;
        this.playback = null;
        const stream = this.stream;
        this.stream = null;
        for (const track of stream?.getTracks() ?? []) track.stop();
        const context = this.context;
        this.context = null;
        if (context && context.state !== "closed") context.close().catch(() => {
            console.warn("[vox:audio.close_failed] Браузер не подтвердил закрытие аудиоустройства");
            this.error("Браузер не подтвердил закрытие аудиоустройства.");
        });
        this.turn = null;
        this.receiver.reset();
        this.ui.level.value = 0;
        this.status(message);
        this.updateControls();
    }
}

new VoxClient();
