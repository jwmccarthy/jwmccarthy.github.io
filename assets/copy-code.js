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
    button.textContent = 'Copy';
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(code.innerText);
        button.textContent = 'Copied';
      } catch {
        button.textContent = 'Copy failed';
      }
      window.setTimeout(() => { button.textContent = 'Copy'; }, 1500);
    });
    footer.append(button);
    block.append(footer);
  });
});
