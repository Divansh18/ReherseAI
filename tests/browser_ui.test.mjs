import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../src/rehearse_ai/web/app.js', import.meta.url), 'utf8');

class TextNode {
  constructor(value) { this.textContent = value; this.parent = null; }
}

class Element {
  constructor(tag = 'div') {
    this.tag = tag;
    this.children = [];
    this.parent = null;
    this._text = '';
    this.className = '';
    this.dataset = {};
    this.hidden = false;
    this.disabled = false;
    this.scrollHeight = 400;
    this.clientHeight = 100;
    this.scrollTop = 0;
    this.classList = {
      add: (value) => { this.className = [...new Set([...this.className.split(' '), value])].filter(Boolean).join(' '); },
      remove: (value) => { this.className = this.className.split(' ').filter((item) => item !== value).join(' '); },
    };
  }
  get textContent() { return this._text + this.children.map((child) => child.textContent).join(''); }
  set textContent(value) { this._text = value; this.children = []; }
  append(...nodes) {
    for (const node of nodes) { node.parent = this; this.children.push(node); }
  }
  replaceChildren(...nodes) { this.children = []; this._text = ''; this.append(...nodes); }
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter((child) => child !== this);
  }
  querySelector(selector) {
    for (const child of this.children) {
      if (child instanceof Element) {
        if (child.className.split(' ').includes(selector.slice(1))) return child;
        const nested = child.querySelector(selector);
        if (nested) return nested;
      }
    }
    return null;
  }
  addEventListener() {}
}

function fixture() {
  const ids = ['start', 'end', 'connection', 'microphone', 'status-dot', 'phase-label',
    'question', 'question-count', 'guidance', 'transcript', 'error', 'remote-audio', 'session-id'];
  const nodes = Object.fromEntries(ids.map((id) => [id, new Element()]));
  nodes['question-count'].hidden = true;
  nodes.error.hidden = true;
  nodes.start.disabled = false;
  nodes.end.disabled = true;
  nodes.connection.textContent = 'Ready';
  nodes.microphone.textContent = 'Microphone off';
  const empty = new Element('li');
  empty.className = 'empty';
  nodes.transcript.append(empty);
  nodes['remote-audio'].pause = () => {};
  const document = {
    querySelector: (selector) => nodes[selector.slice(1)],
    createElement: (tag) => new Element(tag),
    createTextNode: (text) => new TextNode(text),
  };
  const context = vm.createContext({ document, window: { addEventListener() {} }, navigator: {} });
  vm.runInContext(source, context);
  return {
    nodes,
    context,
    send: (event) => context.handleEvent({ type: 'rehearseai', event }),
  };
}

test('queued speech stays out of transcript; output, interruption, and candidate continuations stay ordered', () => {
  const { nodes, send } = fixture();
  send({ event: 'agent_prepared', utterance_id: 0, utterance_kind: 'greeting', text: 'Unplayed.' });
  send({ event: 'agent_interrupted', utterance_id: 0 });
  assert.equal(nodes.transcript.children.length, 1);
  send({ event: 'agent_prepared', utterance_id: 1, utterance_kind: 'greeting', text: 'Hello.' });
  assert.equal(nodes.transcript.children.length, 1);
  assert.equal(nodes['question-count'].hidden, true);
  assert.equal(nodes.connection.textContent, 'Thinking');
  send({ event: 'agent_output_started', utterance_id: 1 });
  assert.equal(nodes.transcript.children[0].querySelector('.speaker').textContent, 'RehearseAI');
  assert.equal(nodes.connection.textContent, 'Speaking');
  send({ event: 'agent_interrupted', utterance_id: 1 });
  assert.equal(nodes.transcript.children[0].querySelector('.annotation').textContent, 'Interrupted');
  assert.equal(nodes.connection.textContent, 'Listening');
  send({ event: 'agent_prepared', utterance_id: 2, utterance_kind: 'question', question_number: 1, text: 'Tell me about a project.' });
  send({ event: 'agent_output_started', utterance_id: 2 });
  assert.equal(nodes['question-count'].hidden, false);
  assert.equal(nodes['question-count'].textContent, '01 / 04');
  send({ event: 'agent_delivery_reported', utterance_id: 2 });
  assert.equal(nodes.connection.textContent, 'Listening');
  send({ event: 'candidate_final', answer_number: 1, text: 'I built a parser.' });
  send({ event: 'candidate_final', answer_number: 1, text: 'It reads config files.' });
  assert.equal(nodes.transcript.children.length, 3);
  assert.equal(nodes.transcript.children[2].querySelector('.speaker').textContent, 'You');
  assert.match(nodes.transcript.children[2].textContent, /I built a parser\. It reads config files\./);
});

test('transcript does not snap back when reader has scrolled up', () => {
  const { nodes, send } = fixture();
  send({ event: 'candidate_final', answer_number: 1, text: 'First segment.' });
  assert.equal(nodes.transcript.scrollTop, 0);
  nodes.transcript.scrollTop = 295;
  send({ event: 'candidate_final', answer_number: 1, text: 'Second segment.' });
  assert.equal(nodes.transcript.scrollTop, 400);
});

test('no-audio warning stays visible during preparation and clears when server output starts', () => {
  const { nodes, send } = fixture();
  send({ event: 'response_error', error_code: 'tts_context_no_audio' });
  assert.equal(nodes.connection.textContent, 'Error');
  assert.equal(nodes.error.hidden, false);
  assert.match(nodes.error.textContent, /returned no audio/);
  send({ event: 'agent_prepared', utterance_id: 3, utterance_kind: 'question', question_number: 1, text: 'Question one.' });
  assert.equal(nodes.error.hidden, false);
  send({ event: 'agent_output_started', utterance_id: 3 });
  assert.equal(nodes.error.hidden, true);
  assert.equal(nodes.connection.textContent, 'Speaking');
  send({ event: 'interview_complete' });
  assert.equal(nodes.connection.textContent, 'Ended');
});

test('connecting opens without question progress and microphone denial shows an actionable error', async () => {
  const { nodes, context } = fixture();
  let rejectPermission;
  context.navigator.mediaDevices = {
    getUserMedia: () => new Promise((_, reject) => { rejectPermission = reject; }),
  };
  const attempt = context.startSession();
  assert.equal(nodes.connection.textContent, 'Connecting');
  assert.equal(nodes['phase-label'].textContent, 'Opening');
  assert.equal(nodes['question-count'].hidden, true);
  rejectPermission(Object.assign(new Error('denied'), { name: 'NotAllowedError' }));
  await attempt;
  assert.equal(nodes.connection.textContent, 'Error');
  assert.match(nodes.error.textContent, /Microphone permission was denied/);
  assert.equal(nodes.start.disabled, false);
  assert.equal(nodes.end.disabled, true);
});

test('End session releases tracks and connections immediately', () => {
  const { nodes, context } = fixture();
  const closed = [];
  const track = { readyState: 'live', stop: () => { track.readyState = 'ended'; } };
  context.fakeStream = { getTracks: () => [track], getAudioTracks: () => [track] };
  context.fakePeer = { close: () => closed.push('peer') };
  context.fakeChannel = { close: () => closed.push('channel') };
  nodes['remote-audio'].pause = () => closed.push('audio');
  vm.runInContext('microphone = fakeStream; peer = fakePeer; channel = fakeChannel;', context);
  context.endSession();
  assert.deepEqual(closed, ['channel', 'peer', 'audio']);
  assert.equal(track.readyState, 'ended');
  assert.equal(nodes.microphone.textContent, 'Microphone off');
  assert.equal(nodes.connection.textContent, 'Ended');
  assert.equal(nodes.start.disabled, false);
  assert.equal(nodes.end.disabled, true);
});
