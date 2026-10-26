const startButton = document.querySelector('#start');
const endButton = document.querySelector('#end');
const connectionLabel = document.querySelector('#connection');
const microphoneLabel = document.querySelector('#microphone');
const statusDot = document.querySelector('#status-dot');
const questionLabel = document.querySelector('#question');
const questionCount = document.querySelector('#question-count');
const transcript = document.querySelector('#transcript');
const errorLabel = document.querySelector('#error');
const audio = document.querySelector('#remote-audio');
let peer = null;
let microphone = null;
let channel = null;
const utterances = new Map();
const answers = new Map();

function status(message, mic = 'Microphone off', active = false) {
  connectionLabel.textContent = message;
  microphoneLabel.textContent = mic;
  statusDot.classList.toggle('active', active);
}

function showError(message) {
  errorLabel.textContent = message;
  errorLabel.hidden = false;
  status('Connection needs attention');
}

function addTurn(speaker, text, id = null) {
  transcript.querySelector('.empty')?.remove();
  const item = document.createElement('li');
  item.className = speaker === 'candidate' ? 'candidate' : 'agent';
  const label = document.createElement('span');
  label.className = 'speaker';
  label.textContent = speaker === 'candidate' ? 'You' : 'Coach';
  item.append(label, document.createTextNode(text));
  transcript.append(item);
  transcript.scrollTop = transcript.scrollHeight;
  if (id !== null) {
    item.classList.add('prepared');
    utterances.set(id, item);
  }
  return item;
}

function handleEvent(message) {
  if (message.type !== 'rehearseai') return;
  const entry = message.event;
  if (!entry || !entry.event) return;
  if (entry.event === 'session_started') {
    document.querySelector('#session-id').textContent = `Session ${entry.session_id}`;
  } else if (entry.event === 'agent_prepared') {
    addTurn('agent', entry.text, entry.utterance_id);
    if (entry.utterance_kind === 'question') {
      questionLabel.textContent = entry.text;
      questionCount.textContent = `${String(entry.question_number).padStart(2, '0')} / 04`;
    }
  } else if (entry.event === 'agent_interrupted') {
    const item = utterances.get(entry.utterance_id);
    item?.classList.remove('prepared');
    item?.classList.add('interrupted');
  } else if (entry.event === 'agent_delivery_reported') {
    utterances.get(entry.utterance_id)?.classList.remove('prepared');
  } else if (entry.event === 'candidate_final' && entry.text) {
    const answerNumber = entry.answer_number ?? entry.sequence;
    const previous = answers.get(answerNumber);
    if (previous) {
      previous.append(document.createTextNode(` ${entry.text}`));
    } else {
      answers.set(answerNumber, addTurn('candidate', entry.text));
    }
    transcript.scrollTop = transcript.scrollHeight;
  } else if (entry.event === 'awaiting_continuation') {
    status('Keep going when you’re ready', 'Microphone on', true);
  } else if (entry.event === 'interview_complete') {
    status('Interview complete', 'Microphone still connected', true);
  } else if (entry.event === 'clarification_limit') {
    status('Please answer the current question', 'Microphone on', true);
  } else if (entry.event === 'response_error') {
    showError('The coach could not prepare a reply. End and restart this session.');
  }
}

function endSession() {
  channel?.close();
  channel = null;
  peer?.close();
  peer = null;
  microphone?.getTracks().forEach((track) => track.stop());
  microphone = null;
  audio.srcObject = null;
  startButton.disabled = false;
  endButton.disabled = true;
  status('Session ended');
}

async function waitForIce(pc) {
  if (pc.iceGatheringState === 'complete') return;
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => { pc.removeEventListener('icegatheringstatechange', check); reject(new Error('WebRTC setup timed out.')); }, 12000);
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
  errorLabel.hidden = true;
  transcript.innerHTML = '<li class="empty">Your conversation will appear here as it happens.</li>';
  utterances.clear();
  answers.clear();
  questionLabel.textContent = 'Your first question will appear here.';
  questionCount.textContent = '01 / 04';
  startButton.disabled = true;
  status('Requesting microphone…');
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Microphone access requires a supported browser on localhost.');
    microphone = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    status('Connecting…', 'Microphone on', true);
    peer = new RTCPeerConnection();
    microphone.getAudioTracks().forEach((track) => peer.addTrack(track, microphone));
    channel = peer.createDataChannel('rehearseai');
    channel.onmessage = (event) => {
      try { handleEvent(JSON.parse(event.data)); } catch { /* Ignore unrelated data. */ }
    };
    peer.ontrack = (event) => {
      audio.srcObject = event.streams[0] || new MediaStream([event.track]);
      audio.play().catch(() => showError('Audio playback was blocked. Allow audio in your browser and retry.'));
    };
    peer.onconnectionstatechange = () => {
      if (peer?.connectionState === 'connected') status('Interview in progress', 'Microphone on', true);
      if (peer && ['failed', 'disconnected'].includes(peer.connectionState)) showError('Connection lost. End and restart the session.');
    };
    await peer.setLocalDescription(await peer.createOffer());
    await waitForIce(peer);
    const response = await fetch('/api/offer', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sdp: peer.localDescription.sdp, type: peer.localDescription.type }),
    });
    if (!response.ok) throw new Error('The local voice server rejected the connection.');
    const answer = await response.json();
    await peer.setRemoteDescription({ type: answer.type, sdp: answer.sdp });
    endButton.disabled = false;
  } catch (error) {
    endSession();
    showError(error.name === 'NotAllowedError' ? 'Microphone permission was denied. Allow access in your browser and try again.' : error.message);
  }
}

startButton.addEventListener('click', startSession);
endButton.addEventListener('click', endSession);
window.addEventListener('pagehide', endSession);
