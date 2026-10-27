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
    this.attributes = {};
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
  querySelectorAll(selector) {
    return this.children.filter((child) => child instanceof Element && child.tag === selector);
  }
  setAttribute(name, value) { this.attributes[name] = value; }
  focus() { this.focused = true; }
  addEventListener(name, handler) { this[`on${name}`] = handler; }
}

function fixture({ active = true } = {}) {
  const ids = ['start', 'end', 'connection', 'microphone', 'status-dot', 'phase-label',
    'question', 'question-count', 'question-progress', 'guidance', 'transcript',
    'jump-latest', 'elapsed', 'error', 'remote-audio', 'session-id'];
  const nodes = Object.fromEntries(ids.map((id) => [id, new Element()]));
  for (let i = 0; i < 4; i += 1) nodes['question-progress'].append(new Element('span'));
  nodes['question-count'].hidden = true;
  nodes['jump-latest'].hidden = true;
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
  const timeouts = new Map();
  const intervals = new Map();
  let nextTimer = 1;
  const context = vm.createContext({
    document, window: { addEventListener() {} }, navigator: {},
    setTimeout: (fn) => { const id = nextTimer++; timeouts.set(id, fn); return id; },
    clearTimeout: (id) => timeouts.delete(id),
    setInterval: (fn) => { const id = nextTimer++; intervals.set(id, fn); return id; },
    clearInterval: (id) => intervals.delete(id),
  });
  vm.runInContext(source, context);
  if (active) vm.runInContext('sessionClosed = false;', context);
  return {
    nodes,
    context,
    timeouts,
    intervals,
    flushTimeouts: () => { for (const [id, fn] of [...timeouts]) { timeouts.delete(id); fn(); } },
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
  assert.equal(nodes.transcript.children[0].children[0].className, 'message');
  assert.equal(nodes.transcript.children[0].children[0].children[0].className, 'speaker');
  assert.equal(nodes.connection.textContent, 'Speaking');
  send({ event: 'agent_interrupted', utterance_id: 1 });
  assert.equal(nodes.transcript.children[0].querySelector('.annotation').textContent, 'Interrupted');
  assert.equal(nodes.connection.textContent, 'Listening');
  send({ event: 'agent_prepared', utterance_id: 2, utterance_kind: 'question', question_number: 1, text: 'Tell me about a project.' });
  send({ event: 'agent_output_started', utterance_id: 2 });
  assert.equal(nodes['question-count'].hidden, false);
  assert.equal(nodes['question-count'].textContent, '01 / 04');
  assert.equal(nodes['question-progress'].children[0].dataset.state, 'active');
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
  nodes.transcript.onscroll();
  assert.equal(nodes['jump-latest'].hidden, false);
  nodes.transcript.scrollTop = 100;
  send({ event: 'candidate_final', answer_number: 1, text: 'Second segment.' });
  assert.equal(nodes.transcript.scrollTop, 100);
  assert.equal(nodes['jump-latest'].hidden, false);
  nodes['jump-latest'].onclick();
  assert.equal(nodes.transcript.scrollTop, 400);
  assert.equal(nodes['jump-latest'].hidden, true);
  assert.equal(nodes.transcript.focused, true);
});

test('a long transcript keeps chronological turns while the reader stays scrolled back', () => {
  const { nodes, send } = fixture();
  for (let index = 1; index <= 50; index += 1) {
    send({ event: 'candidate_final', answer_number: index, text: `Answer ${index}` });
  }
  assert.equal(nodes.transcript.children.length, 50);
  assert.match(nodes.transcript.children[0].textContent, /Answer 1/);
  assert.match(nodes.transcript.children[49].textContent, /Answer 50/);
  nodes.transcript.scrollTop = 80;
  send({ event: 'candidate_final', answer_number: 51, text: 'Answer 51' });
  assert.equal(nodes.transcript.scrollTop, 80);
  assert.equal(nodes['jump-latest'].hidden, false);
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
  assert.equal(nodes.connection.textContent, 'Finishing');
});

test('connecting opens without question progress and microphone denial shows an actionable error', async () => {
  const { nodes, context } = fixture({ active: false });
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
  assert.equal(nodes.start.textContent, 'Start again');
});

test('closing feedback drains before automatic cleanup, then freezes timer and preserves transcript', () => {
  const { nodes, context, send, timeouts, intervals, flushTimeouts } = fixture();
  const closed = [];
  const track = { readyState: 'live', stop: () => { track.readyState = 'ended'; } };
  context.fakeStream = { getTracks: () => [track], getAudioTracks: () => [track] };
  context.fakePeer = { close: () => closed.push('peer') };
  context.fakeChannel = { close: () => closed.push('channel') };
  nodes['remote-audio'].pause = () => closed.push('audio');
  vm.runInContext('microphone = fakeStream; peer = fakePeer; channel = fakeChannel;', context);
  send({ event: 'session_started', session_id: 'session-a' });
  assert.equal(intervals.size, 1);
  send({ event: 'agent_prepared', utterance_id: 8, utterance_kind: 'feedback', text: 'Good work.' });
  send({ event: 'agent_output_started', utterance_id: 8 });
  send({ event: 'agent_delivery_reported', utterance_id: 8 });
  send({ event: 'interview_complete' });
  assert.equal(nodes['phase-label'].textContent, 'Interview complete');
  assert.equal(track.readyState, 'live');
  assert.equal(timeouts.size, 1);
  assert.equal(nodes['question-progress'].children[3].dataset.state, 'complete');
  flushTimeouts();
  assert.deepEqual(closed, ['channel', 'peer', 'audio']);
  assert.equal(track.readyState, 'ended');
  assert.equal(nodes.connection.textContent, 'Ended');
  assert.equal(nodes.microphone.textContent, 'Microphone off');
  assert.equal(nodes.start.textContent, 'Start again');
  assert.equal(nodes['phase-label'].textContent, 'Interview complete');
  assert.equal(nodes.transcript.children[0].textContent.includes('Good work.'), true);
  assert.equal(intervals.size, 0);
  send({ event: 'agent_prepared', utterance_id: 9, text: 'Late question.' });
  send({ event: 'agent_output_started', utterance_id: 9 });
  assert.equal(nodes.transcript.children.length, 1);
});

test('manual End cancels pending automatic cleanup and a new session resets presentation', async () => {
  const { nodes, context, send, timeouts, flushTimeouts } = fixture();
  send({ event: 'interview_ended_early' });
  assert.equal(timeouts.size, 1);
  context.endSession();
  assert.equal(timeouts.size, 0);
  flushTimeouts();
  assert.equal(nodes.connection.textContent, 'Ended');
  let rejectPermission;
  context.navigator.mediaDevices = {
    getUserMedia: () => new Promise((_, reject) => { rejectPermission = reject; }),
  };
  const next = context.startSession();
  assert.equal(nodes.connection.textContent, 'Connecting');
  assert.equal(nodes.elapsed.textContent, '00:00');
  assert.equal(nodes['question-progress'].children[0].dataset.state, undefined);
  assert.equal(nodes.transcript.children.length, 1);
  rejectPermission(Object.assign(new Error('denied'), { name: 'NotAllowedError' }));
  await next;
  assert.equal(nodes.connection.textContent, 'Error');
});
