async function loadModels() {
  const res = await fetch('/models');
  const models = await res.json();
  const sel = document.getElementById('model-select');
  models.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m;
    sel.appendChild(opt);
  });
}

function renderBoard(board) {
  const grid = document.getElementById('board');
  grid.innerHTML = '';
  board.flat().forEach(val => {
    const cell = document.createElement('div');
    cell.className = 'tile';
    cell.textContent = val === 0 ? '' : val;
    grid.appendChild(cell);
  });
}

document.getElementById('start-btn').onclick = async () => {
  const model = document.getElementById('model-select').value;
  const res = await fetch('/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model })
  });
  const data = await res.json();
  renderBoard(data.board);
  document.getElementById('ai-btn').disabled = false;
};

document.getElementById('ai-btn').onclick = async () => {
  const res = await fetch('/ai_move', { method: 'POST' });
  const data = await res.json();
  renderBoard(data.board);
};

document.addEventListener('keydown', async e => {
  const map = { ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right' };
  if (map[e.key]) {
    const res = await fetch('/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ direction: map[e.key] })
    });
    const data = await res.json();
    renderBoard(data.board);
    e.preventDefault();
  }
});

loadModels();
