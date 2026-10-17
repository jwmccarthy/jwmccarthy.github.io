document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('div.highlighter-rouge').forEach((block) => {
    const code = block.querySelector('code');
    if (!code) return;

    block.classList.add('code-block');
    const footer = document.createElement('div');
    footer.className = 'code-block-footer';
    const button = document.createElement('button');
    button.className = 'copy-code-button';
    button.type = 'button';
    button.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M16 1H4a2 2 0 0 0-2 2v14h2V3h12V1Zm3 4H8a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2Zm0 16H8V7h11v14Z"/></svg><span>Copy</span>';
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(code.innerText);
        button.querySelector('span').textContent = 'Copied';
      } catch {
        button.querySelector('span').textContent = 'Copy failed';
      }
      window.setTimeout(() => { button.querySelector('span').textContent = 'Copy'; }, 1500);
    });
    footer.append(button);
    block.append(footer);
  });
});
