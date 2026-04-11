// =====================================================================
// Hey Cook — Web Voice Assistant  (frontend logic)
// =====================================================================
// Records audio from the browser microphone, sends WAV to the Flask
// backend, renders the conversation, and plays TTS audio responses.
// =====================================================================

(function () {
  "use strict";

  // ─── DOM refs ───────────────────────────────────────────────
  const messagesEl  = document.getElementById("messages");
  const btnMic      = document.getElementById("btn-mic");
  const statusText  = document.getElementById("status-text");
  const btnClear    = document.getElementById("btn-clear");

  // sidebar
  const sideRecipe   = document.getElementById("sidebar-recipe");
  const sideStep     = document.getElementById("sidebar-step");
  const sideSection  = document.getElementById("sidebar-section");
  const sideProgress = document.querySelector("#sidebar-progress .progress-bar");

  // ─── State ──────────────────────────────────────────────────
  let mediaRecorder  = null;
  let audioChunks    = [];
  let recording      = false;
  let processing     = false;
  let currentAudio   = null;   // currently playing Audio element
  let msgCounter     = 0;
  let chunkQueue     = [];
  let isPlayingChunks = false;
  let activeSSE      = null;

  // ─── Welcome message ───────────────────────────────────────
  addMessage("assistant", "Welcome! Click the microphone and say something like \"How do I make pasta?\" to get started.");

  // ─── Mic button ─────────────────────────────────────────────
  btnMic.addEventListener("click", async () => {
    if (processing) return;

    // Allow interrupting audio playback to start a new recording
    if (isPlayingChunks || activeSSE) {
      stopAnyPlayback();
      finishStreaming();
    }

    if (recording) {
      stopRecording();
    } else {
      await startRecording();
    }
  });

  btnClear.addEventListener("click", async () => {
    stopAnyPlayback();
    messagesEl.innerHTML = "";
    addMessage("assistant", "Chat cleared. Click the microphone to start a new conversation.");

    // Reset sidebar
    sideRecipe.textContent   = "No recipe loaded";
    sideStep.textContent     = "—";
    sideSection.textContent  = "—";
    sideProgress.style.width = "0%";

    // Tell backend to reset session
    try { await fetch("/api/reset", { method: "POST" }); } catch (_) {}
  });

  // ─── Recording ──────────────────────────────────────────────

  function pickMimeType() {
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/mp4",
    ];
    for (const mt of candidates) {
      if (MediaRecorder.isTypeSupported(mt)) return mt;
    }
    return "";  // browser default
  }

  function stopAnyPlayback() {
    if (activeSSE) { activeSSE.close(); activeSSE = null; }
    chunkQueue = [];
    isPlayingChunks = false;
    if (currentAudio) {
      currentAudio.pause();
      currentAudio.currentTime = 0;
      currentAudio = null;
    }
    document.querySelectorAll(".msg-audio.playing").forEach((b) => b.classList.remove("playing"));
  }

  async function startRecording() {
    // Stop any audio that is currently playing
    stopAnyPlayback();

    // Secure-context guard (mic requires HTTPS or localhost)
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      statusText.textContent = "Microphone not available — use HTTPS or localhost";
      statusText.className   = "";
      console.error("mediaDevices API not available (insecure context?)");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks = [];

      const mimeType = pickMimeType();
      const opts = mimeType ? { mimeType } : {};
      mediaRecorder = new MediaRecorder(stream, opts);
      console.log("MediaRecorder using mimeType:", mediaRecorder.mimeType);

      mediaRecorder.addEventListener("dataavailable", (e) => {
        if (e.data.size > 0) audioChunks.push(e.data);
      });

      mediaRecorder.addEventListener("stop", () => {
        stream.getTracks().forEach((t) => t.stop());
        handleRecordingComplete();
      });

      mediaRecorder.start(250);  // fire dataavailable every 250ms
      recording = true;
      btnMic.classList.add("recording");
      statusText.textContent = "Recording… click again to stop";
      statusText.className   = "recording";
    } catch (err) {
      console.error("Mic / MediaRecorder error:", err);
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        statusText.textContent = "Microphone permission denied — allow it in your browser";
      } else if (err.name === "NotFoundError") {
        statusText.textContent = "No microphone found on this device";
      } else {
        statusText.textContent = "Microphone error: " + err.message;
      }
      statusText.className = "";
    }
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
    recording = false;
    btnMic.classList.remove("recording");
  }

  // ─── Process recorded audio ─────────────────────────────────

  async function handleRecordingComplete() {
    if (audioChunks.length === 0) return;

    processing = true;
    btnMic.classList.add("processing");
    statusText.textContent = "Processing…";
    statusText.className   = "processing";

    // Convert recorded blob to WAV (16 kHz mono PCM) for ASR compatibility
    const recMime = (mediaRecorder && mediaRecorder.mimeType) || "audio/webm";
    const rawBlob = new Blob(audioChunks, { type: recMime });
    console.log("Recorded blob:", rawBlob.size, "bytes,", recMime);
    let wavBlob;
    try {
      wavBlob = await convertToWav(rawBlob);
      console.log("Converted to WAV:", wavBlob.size, "bytes");
    } catch (e) {
      console.error("WAV conversion failed, sending raw recording:", e);
      wavBlob = rawBlob;
    }

    const thinkingId = addMessage("assistant", "Thinking…", { thinking: true });

    const formData = new FormData();
    formData.append("audio", wavBlob, "recording.wav");

    try {
      const resp = await fetch("/api/process", { method: "POST", body: formData });
      const data = await resp.json();

      removeMessage(thinkingId);

      if (data.transcript) {
        addMessage("user", data.transcript);
      }

      addMessage("assistant", data.response, { intent: data.intent });
      updateSidebar(data.sidebar);

      // Unlock mic now — audio playback should not block new commands
      processing = false;
      btnMic.classList.remove("processing");
      statusText.textContent = "Click the microphone and speak";
      statusText.className   = "";

      // Stream TTS chunks via SSE and play them sequentially
      if (data.tts_job_id) {
        streamAndPlayChunks(data.tts_job_id);
      }
    } catch (err) {
      console.error("API error:", err);
      removeMessage(thinkingId);
      addMessage("assistant", "Something went wrong. Please try again.");
      processing = false;
      btnMic.classList.remove("processing");
      statusText.textContent = "Click the microphone and speak";
      statusText.className   = "";
    }
  }

  // ─── Streaming TTS playback ───────────────────────────────────
  // Receives chunk URLs via SSE, queues them, and plays sequentially.

  function streamAndPlayChunks(jobId) {
    chunkQueue = [];
    isPlayingChunks = false;

    if (activeSSE) { activeSSE.close(); activeSSE = null; }

    const es = new EventSource(`/api/tts-stream/${jobId}`);
    activeSSE = es;

    es.addEventListener("chunk", (e) => {
      const { url } = JSON.parse(e.data);
      chunkQueue.push(url);
      if (!isPlayingChunks) playNextChunk();
    });

    es.addEventListener("done", () => {
      es.close();
      activeSSE = null;
    });

    es.onerror = () => {
      es.close();
      activeSSE = null;
      finishStreaming();
    };
  }

  function playNextChunk() {
    if (chunkQueue.length === 0) {
      isPlayingChunks = false;
      if (!activeSSE) finishStreaming();
      return;
    }

    isPlayingChunks = true;
    const url = chunkQueue.shift();
    const audio = new Audio(url);
    currentAudio = audio;
    audio.play().catch((e) => console.warn("Auto-play blocked:", e));
    audio.addEventListener("ended", () => {
      currentAudio = null;
      playNextChunk();
    });
    audio.addEventListener("error", () => {
      currentAudio = null;
      playNextChunk();
    });
  }

  function finishStreaming() {
    isPlayingChunks = false;
  }

  // ─── WAV conversion ─────────────────────────────────────────
  // Decodes the webm audio to PCM via OfflineAudioContext and
  // encodes it as a 16-bit mono WAV for Deepgram.

  async function convertToWav(blob) {
    const arrayBuf = await blob.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const decoded  = await audioCtx.decodeAudioData(arrayBuf);
    audioCtx.close();

    const sampleRate  = 16000;
    const numChannels = 1;

    // Resample to 16 kHz mono
    const offCtx   = new OfflineAudioContext(numChannels, decoded.duration * sampleRate, sampleRate);
    const source   = offCtx.createBufferSource();
    source.buffer  = decoded;
    source.connect(offCtx.destination);
    source.start(0);
    const rendered = await offCtx.startRendering();

    const pcm = rendered.getChannelData(0);

    // Encode WAV
    const buffer = new ArrayBuffer(44 + pcm.length * 2);
    const view   = new DataView(buffer);

    // RIFF header
    writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + pcm.length * 2, true);
    writeString(view, 8, "WAVE");
    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);                         // PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * 2, true);
    view.setUint16(32, numChannels * 2, true);
    view.setUint16(34, 16, true);                         // 16-bit
    writeString(view, 36, "data");
    view.setUint32(40, pcm.length * 2, true);

    // PCM samples
    let offset = 44;
    for (let i = 0; i < pcm.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, pcm[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }

    return new Blob([buffer], { type: "audio/wav" });
  }

  function writeString(view, offset, str) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  }

  // ─── Message rendering ──────────────────────────────────────

  function addMessage(role, text, opts = {}) {
    const id  = "msg-" + (++msgCounter);
    const div = document.createElement("div");
    div.className = `msg ${role}` + (opts.thinking ? " thinking" : "");
    div.id = id;

    let html = `<div class="msg-label">${role === "user" ? "You" : "Hey Cook"}</div>`;
    html += `<div class="msg-body">${escapeHtml(text)}</div>`;

    if (opts.audioUrl) {
      html += `<button class="msg-audio" data-url="${opts.audioUrl}" onclick="window.__playAudio(this)">
                 <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                 Play audio
               </button>`;
    }


    div.innerHTML = html;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return id;
  }

  function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function escapeHtml(str) {
    const d = document.createElement("div");
    d.textContent = str;
    return d.innerHTML;
  }

  // ─── Audio playback ─────────────────────────────────────────

  function playAudio(url) {
    if (currentAudio) {
      currentAudio.pause();
      currentAudio = null;
      document.querySelectorAll(".msg-audio.playing").forEach((b) => b.classList.remove("playing"));
    }
    const audio = new Audio(url);
    currentAudio = audio;
    audio.play().catch((e) => console.warn("Auto-play blocked:", e));
    audio.addEventListener("ended", () => { currentAudio = null; });
  }

  // Exposed globally so inline onclick works
  window.__playAudio = function (btn) {
    const url = btn.dataset.url;

    // If this button's audio is already playing, pause it
    if (btn.classList.contains("playing") && currentAudio) {
      currentAudio.pause();
      currentAudio = null;
      btn.classList.remove("playing");
      return;
    }

    // Stop any other playing audio
    if (currentAudio) {
      currentAudio.pause();
      currentAudio = null;
    }
    document.querySelectorAll(".msg-audio.playing").forEach((b) => b.classList.remove("playing"));

    const audio = new Audio(url);
    currentAudio = audio;
    btn.classList.add("playing");
    audio.play();
    audio.addEventListener("ended", () => {
      currentAudio = null;
      btn.classList.remove("playing");
    });
  };

  // ─── Sidebar updates ───────────────────────────────────────

  function updateSidebar(sidebar) {
    if (!sidebar) return;
    sideRecipe.textContent  = sidebar.recipe_title || "No recipe loaded";
    sideSection.textContent = sidebar.current_section || "—";

    const step  = sidebar.current_step || 0;
    const total = sidebar.total_steps  || 0;

    if (total > 0) {
      sideStep.textContent    = `Step ${step} of ${total}`;
      sideProgress.style.width = `${(step / total) * 100}%`;
    } else {
      sideStep.textContent    = "—";
      sideProgress.style.width = "0%";
    }
  }
})();
