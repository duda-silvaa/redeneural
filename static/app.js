const button = document.querySelector('#trainBtn');
const status = document.querySelector('#status');
const canvas = document.querySelector('#lossChart');

function drawLoss(loss) {
  const ctx = canvas.getContext('2d');
  const width = canvas.clientWidth * devicePixelRatio;
  const height = canvas.clientHeight * devicePixelRatio;
  canvas.width = width; canvas.height = height;
  ctx.clearRect(0, 0, width, height);
  const max = Math.max(...loss), min = Math.min(...loss);
  const x = i => 12 + i * (width - 24) / Math.max(loss.length - 1, 1);
  const y = value => 10 + (max - value) * (height - 24) / Math.max(max - min, 0.001);
  ctx.strokeStyle = '#e6ebe3'; ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) { ctx.beginPath(); ctx.moveTo(0, i * height / 4); ctx.lineTo(width, i * height / 4); ctx.stroke(); }
  ctx.beginPath(); loss.forEach((value, i) => i ? ctx.lineTo(x(i), y(value)) : ctx.moveTo(x(i), y(value)));
  ctx.strokeStyle = '#86bd59'; ctx.lineWidth = 3; ctx.stroke();
  ctx.lineTo(x(loss.length - 1), height); ctx.lineTo(x(0), height); ctx.closePath();
  ctx.fillStyle = 'rgba(168,228,111,.15)'; ctx.fill();
}

button.addEventListener('click', async () => {
  button.disabled = true; button.innerHTML = 'Treinando... <span>↻</span>'; status.textContent = 'Baixando dados e ajustando pesos...';
  try {
    const response = await fetch('/api/train');
    if (!response.ok) throw new Error('Falha ao treinar');
    const data = await response.json();
    ['accuracy', 'precision', 'f1'].forEach(key => document.querySelector(`#${key}`).textContent = `${(data[key] * 100).toFixed(1)}%`);
    document.querySelector('#date').textContent = `último dado · ${data.latest_date}`;
    document.querySelector('#samples').textContent = `${data.samples.train.toLocaleString('pt-BR')} treino · ${data.samples.validation.toLocaleString('pt-BR')} validação · ${data.samples.test.toLocaleString('pt-BR')} teste`;
    document.querySelector('#trainingDetails').textContent = `${data.samples.validation.toLocaleString('pt-BR')} validação · ${data.parameters} parâmetros · ${data.epochs_trained} épocas`;
    drawLoss(data.loss); status.textContent = `Concluído · limiar de alto risco: ${data.threshold} óbitos/dia`;
  } catch (error) { status.textContent = 'Não foi possível carregar o dataset. Tente novamente.'; }
  button.disabled = false; button.innerHTML = 'Treinar novamente <span>→</span>';
});
