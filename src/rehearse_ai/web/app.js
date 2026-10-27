const startButton = document.querySelector('#start');
const endButton = document.querySelector('#end');
const connectionLabel = document.querySelector('#connection');
const microphoneLabel = document.querySelector('#microphone');
const statusDot = document.querySelector('#status-dot');
const phaseLabel = document.querySelector('#phase-label');
const questionLabel = document.querySelector('#question');
const questionCount = document.querySelector('#question-count');
const guidance = document.querySelector('#guidance');
const transcript = document.querySelector('#transcript');
const jumpLatest = document.querySelector('#jump-latest');
const elapsed = document.querySelector('#elapsed');
const progressTrack = document.querySelector('#question-progress');
const errorLabel = document.querySelector('#error');
const audio = document.querySelector('#remote-audio');
let peer = null;
let microphone = null;
let channel = null;
const utterances = new Map();
const prepared = new Map();
const answers = new Map();
let attempt = 0;
let offerAbort = null;
let sessionClosed = true;
let terminalPending = false;
let cleanupTimeout = null;
let timerInterval = null;
let timerStartedAt = null;

function renderElapsed() {
  const seconds = timerStartedAt === null ? 0 : Math.floor((Date.now() - timerStartedAt) / 1000);
  elapsed.textContent = `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
}

function stopTimer() {
  if (timerInterval !== null) clearInterval(timerInterval);
  timerInterval = null;
  if (timerStartedAt !== null) renderElapsed();
  timerStartedAt = null;
}

function startTimer() {
  if (timerInterval !== null || timerStartedAt !== null) return;
  timerStartedAt = Date.now();
  renderElapsed();
  timerInterval = setInterval(renderElapsed, 1000);
}

function setProgress(questionNumber = 0, complete = false) {
  const segments = progressTrack.querySelectorAll('span');
  segments.forEach((segment, index) => {
    const state = complete || index + 1 < questionNumber ? 'complete'
      : index + 1 === questionNumber ? 'active' : '';
    if (state) segment.dataset.state = state;
    else delete segment.dataset.state;
  });
  progressTrack.setAttribute('aria-label', complete ? 'All four questions complete'
    : questionNumber ? `Question ${questionNumber} of 4` : 'Interview has not started');
}

function microphoneState() {
  return microphone?.getAudioTracks().some((track) => track.readyState === 'live')
    ? 'Microphone on' : 'Microphone off';
}

function setStatus(label) {
  connectionLabel.textContent = label;
  microphoneLabel.textContent = microphoneState();
  statusDot.dataset.state = label.toLowerCase();
}

function setInterviewPhase(phase, text, hint = '') {
  phaseLabel.textContent = phase;
  questionLabel.textContent = text;
  guidance.textContent = hint;
}

function showError(message) {
  errorLabel.textContent = message;
  errorLabel.hidden = false;
  errorLabel.dataset.kind = 'error';
  setStatus('Error');
}

function showWarning(message) {
  errorLabel.textContent = message;
  errorLabel.hidden = false;
  errorLabel.dataset.kind = 'warning';
  setStatus('Error');
}

function clearWarning() {
  if (errorLabel.dataset.kind === 'warning') {
    errorLabel.hidden = true;
    delete errorLabel.dataset.kind;
  }
}

function nearTranscriptBottom() {
  return transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 72;
}

function scrollIfNear(nearBottom) {
  if (nearBottom) {
    transcript.scrollTop = transcript.scrollHeight;
    jumpLatest.hidden = true;
  } else {
    jumpLatest.hidden = false;
  }
}

transcript.addEventListener('scroll', () => {
  jumpLatest.hidden = nearTranscriptBottom();
});
jumpLatest.addEventListener('click', () => {
  transcript.scrollTop = transcript.scrollHeight;
  jumpLatest.hidden = true;
  transcript.focus();
});

function addTurn(speaker, text, id = null) {
  const nearBottom = nearTranscriptBottom();
  transcript.querySelector('.empty')?.remove();
  const item = document.createElement('li');
  item.className = speaker === 'candidate' ? 'candidate' : 'agent';
  const message = document.createElement('div');
  message.className = 'message';
  const label = document.createElement('span');
  label.className = 'speaker';
  label.textContent = speaker === 'candidate' ? 'You' : 'RehearseAI';
  const copy = document.createElement('span');
  copy.className = 'message-copy';
  copy.textContent = text;
  message.append(label, copy);
  item.append(message);
  transcript.append(item);
  scrollIfNear(nearBottom);
  if (id !== null) utterances.set(id, item);
  return item;
}

function markInterrupted(item) {
  if (!item || item.querySelector('.annotation')) return;
  const annotation = document.createElement('span');
  annotation.className = 'annotation';
  annotation.textContent = 'Interrupted';
  item.querySelector('.message').append(annotation);
}

function handleEvent(message) {
  if (message.type !== 'rehearseai') return;
  const entry = message.event;
  if (!entry || !entry.event) return;
  if (terminalPending || sessionClosed && attempt > 0) return;

  if (entry.event === 'session_started') {
    document.querySelector('#session-id').textContent = `Session ${entry.session_id}`;
    startTimer();
  } else if (entry.event === 'candidate_started') {
    if (errorLabel.hidden) setStatus('Listening');
  } else if (entry.event === 'candidate_turn_completed') {
    if (errorLabel.hidden) setStatus('Thinking');
  } else if (entry.event === 'agent_prepared') {
    prepared.set(entry.utterance_id, entry);
    if (errorLabel.hidden) setStatus('Thinking');
  } else if (entry.event === 'agent_output_started') {
    clearWarning();
    const pending = prepared.get(entry.utterance_id);
    if (!pending) return;
    if (!utterances.has(entry.utterance_id)) {
      addTurn('agent', pending.text, entry.utterance_id);
    }
    if (pending.utterance_kind === 'question') {
      setInterviewPhase('Current question', pending.text, 'Take your time with your answer.');
      questionCount.textContent = `${String(pending.question_number).padStart(2, '0')} / 04`;
      questionCount.hidden = false;
      setProgress(pending.question_number);
    }
    setStatus('Speaking');
  } else if (entry.event === 'agent_interrupted') {
    prepared.delete(entry.utterance_id);
    markInterrupted(utterances.get(entry.utterance_id));
    if (errorLabel.hidden) setStatus('Listening');
  } else if (entry.event === 'agent_delivery_reported') {
    prepared.delete(entry.utterance_id);
    if (errorLabel.hidden) setStatus('Listening');
  } else if (entry.event === 'candidate_final' && entry.text) {
    const answerNumber = entry.answer_number ?? entry.sequence;
    const previous = answers.get(answerNumber);
    if (previous) {
      const nearBottom = nearTranscriptBottom();
      previous.querySelector('.message-copy').append(document.createTextNode(` ${entry.text}`));
      scrollIfNear(nearBottom);
    } else {
      answers.set(answerNumber, addTurn('candidate', entry.text));
    }
  } else if (entry.event === 'awaiting_continuation') {
    guidance.textContent = 'Take your time; keep going when you’re ready.';
    if (errorLabel.hidden) setStatus('Listening');
  } else if (entry.event === 'interview_complete') {
    setInterviewPhase('Complete', 'Your feedback is in the conversation.', 'Finishing the closing audio.');
    questionCount.hidden = true;
    setProgress(4, true);
    scheduleAutomaticEnd();
  } else if (entry.event === 'interview_ended_early') {
    setInterviewPhase('Ended early', 'Interview ended early.', 'Finishing the closing audio.');
    questionCount.hidden = true;
    scheduleAutomaticEnd();
  } else if (entry.event === 'clarification_limit') {
    guidance.textContent = 'Answer the current question, or say “end the interview” to stop.';
    if (errorLabel.hidden) setStatus('Listening');
  } else if (entry.event === 'opening_repeat_limit') {
    guidance.textContent = entry.opening_phase === 'greeting'
      ? 'Tell RehearseAI how you’re doing to continue.'
      : 'Say “start” when ready, or “end the interview” to stop.';
    if (errorLabel.hidden) setStatus('Listening');
  } else if (entry.event === 'opening_waiting_for_readiness') {
    guidance.textContent = 'Say “start” when ready, or “end the interview” to stop.';
    if (errorLabel.hidden) setStatus('Listening');
  } else if (entry.event === 'no_project_waiting') {
    guidance.textContent = 'Return with a technical example, or say “end the interview”.';
    if (errorLabel.hidden) setStatus('Listening');
  } else if (entry.event === 'no_project_repeat_limit' || entry.event === 'project_example_required') {
    guidance.textContent = 'Use a technical example, or say “end the interview”.';
    if (errorLabel.hidden) setStatus('Listening');
  } else if (entry.event === 'response_error') {
    if (entry.error_code === 'tts_context_no_audio') {
      showWarning('The voice service returned no audio. You can speak to retry the prompt.');
    } else if (entry.failure_stage === 'opening_inference' || entry.failure_stage === 'opening_intent') {
      showWarning('The coach is recovering from a planning error. You can keep speaking.');
    } else if (entry.failure_stage === 'pipeline' && ['OpenAITTSService', 'CartesiaTTSService'].includes(entry.processor)) {
      showError('The voice provider could not generate audio. Try speaking again; end the session if it persists.');
    } else {
      showError('The coach could not complete a response. Try speaking again or end the session.');
    }
  }
}

function finishSession({ manual = false } = {}) {
  if (sessionClosed) return;
  sessionClosed = true;
  terminalPending = true;
  attempt += 1;
  if (cleanupTimeout !== null) clearTimeout(cleanupTimeout);
  cleanupTimeout = null;
  stopTimer();
  offerAbort?.abort();
  offerAbort = null;
  channel?.close();
  channel = null;
  peer?.close();
  peer = null;
  microphone?.getTracks().forEach((track) => track.stop());
  microphone = null;
  audio.pause();
  audio.srcObject = null;
  startButton.disabled = false;
  startButton.textContent = 'Start again';
  endButton.disabled = true;
  if (manual) {
    setInterviewPhase('Ended', 'Session ended.');
    questionCount.hidden = true;
  } else {
    guidance.textContent = 'Your microphone is off. You can start again.';
  }
  setStatus('Ended');
}

function endSession() { finishSession({ manual: true }); }

function scheduleAutomaticEnd() {
  terminalPending = true;
  setStatus('Finishing');
  const endingAttempt = attempt;
  // Output-stop is server-side. Allow a bounded interval for browser playout to drain.
  cleanupTimeout = setTimeout(() => {
    if (endingAttempt === attempt) finishSession();
  }, 1200);
}

async function waitForIce(pc) {
  if (pc.iceGatheringState === 'complete') return;
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      pc.removeEventListener('icegatheringstatechange', check);
      reject(new Error('WebRTC setup timed out.'));
    }, 12000);
    function check() {
      if (pc.iceGatheringState === 'complete') {
        clearTimeout(timeout);
        pc.removeEventListener('icegatheringstatechange', check);
        resolve();
      }
    }
    pc.addEventListener('icegatheringstatechange', check);
  });
}

async function startSession() {
  if (!sessionClosed) return;
  const thisAttempt = ++attempt;
  sessionClosed = false;
  terminalPending = false;
  stopTimer();
  elapsed.textContent = '00:00';
  errorLabel.hidden = true;
  delete errorLabel.dataset.kind;
  const empty = document.createElement('li');
  empty.className = 'empty';
  empty.textContent = 'Your conversation will appear here when you begin.';
  transcript.replaceChildren(empty);
  utterances.clear();
  prepared.clear();
  answers.clear();
  jumpLatest.hidden = true;
  document.querySelector('#session-id').textContent = 'Connecting to a session';
  setInterviewPhase('Opening', 'We’ll start with a quick hello.', 'Allow microphone access when prompted.');
  questionCount.hidden = true;
  setProgress();
  startButton.disabled = true;
  endButton.disabled = false;
  setStatus('Connecting');

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('Microphone access requires a supported browser on localhost.');
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true },
    });
    if (thisAttempt !== attempt) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    microphone = stream;
    microphone.getAudioTracks().forEach((track) => {
      track.addEventListener('ended', () => {
        if (thisAttempt === attempt && microphone === stream) {
          showError('Microphone access stopped. End the session, then allow access and restart.');
        }
      });
    });
    setStatus('Connecting');
    peer = new RTCPeerConnection();
    microphone.getAudioTracks().forEach((track) => peer.addTrack(track, microphone));
    channel = peer.createDataChannel('rehearseai');
    const activeChannel = channel;
    channel.onmessage = (event) => {
      if (thisAttempt !== attempt) return;
      try { handleEvent(JSON.parse(event.data)); } catch { /* Ignore unrelated data. */ }
    };
    channel.onclose = () => {
      if (thisAttempt === attempt && channel === activeChannel && peer) {
        showError('The session connection closed. End and restart the session.');
      }
    };
    peer.ontrack = (event) => {
      if (thisAttempt !== attempt) return;
      audio.srcObject = event.streams[0] || new MediaStream([event.track]);
      audio.play().catch(() => {
        if (thisAttempt === attempt) {
          showError('Audio playback was blocked. Allow audio in your browser and retry.');
        }
      });
    };
    peer.onconnectionstatechange = () => {
      if (thisAttempt !== attempt || !peer) return;
      if (peer.connectionState === 'connected' && connectionLabel.textContent === 'Connecting') {
        setStatus('Listening');
      }
      if (['failed', 'disconnected'].includes(peer.connectionState)) {
        showError('Connection lost. End and restart the session.');
      }
    };
    await peer.setLocalDescription(await peer.createOffer());
    if (thisAttempt !== attempt) return;
    await waitForIce(peer);
    if (thisAttempt !== attempt) return;
    offerAbort = new AbortController();
    const response = await fetch('/api/offer', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sdp: peer.localDescription.sdp, type: peer.localDescription.type }),
      signal: offerAbort.signal,
    });
    if (thisAttempt !== attempt) return;
    if (!response.ok) throw new Error('The local voice server rejected the connection.');
    const answer = await response.json();
    if (thisAttempt !== attempt) return;
    await peer.setRemoteDescription({ type: answer.type, sdp: answer.sdp });
    offerAbort = null;
  } catch (error) {
    if (thisAttempt !== attempt) return;
    endSession();
    showError(error.name === 'NotAllowedError'
      ? 'Microphone permission was denied. Allow access in your browser and try again.'
      : error.message);
  }
}

startButton.addEventListener('click', startSession);
endButton.addEventListener('click', endSession);
window.addEventListener('pagehide', endSession);
