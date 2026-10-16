/* Add new posts at the top. No build step or framework required. */
const posts = [
  {
    date: '2026-10-16',
    title: 'A small place to think in public',
    excerpt: 'Why I am making room for short notes, unfinished ideas, and the things I learn while building.',
    url: 'post.html?post=a-small-place-to-think-in-public',
    file: 'posts/a-small-place-to-think-in-public.md'
  }
];

const postList = document.querySelector('#posts');

postList.innerHTML = posts.map((post) => `
  <a class="timeline-item" href="${post.url}">
    <time class="post-date" datetime="${post.date}">${new Date(`${post.date}T12:00:00`).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</time>
    <span class="timeline-dot" aria-hidden="true"></span>
    <span class="post-title">${post.title}</span>
  </a>`).join('');
