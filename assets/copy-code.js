document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('div.highlighter-rouge').forEach((block) => {
    const code = block.querySelector('code');
    if (!code) return;

    const button = document.createElement('button');
    button.className = 'copy-code-button';
    button.type = 'button';
    button.textContent = 'Copy';
    button.addEventListener('click', async () => {
      await navigator.clipboard.writeText(code.innerText);
      button.textContent = 'Copied';
      window.setTimeout(() => { button.textContent = 'Copy'; }, 1500);
    });
    block.append(button);
  });
});
