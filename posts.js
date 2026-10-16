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
const postCount = document.querySelector('#post-count');

postCount.textContent = `${posts.length} ${posts.length === 1 ? 'note' : 'notes'}`;
postList.innerHTML = posts.map((post) => `
  <a class="post" href="${post.url}">
    <time class="post-date" datetime="${post.date}">${new Date(`${post.date}T12:00:00`).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</time>
    <div><h3 class="post-title">${post.title}</h3><p class="post-excerpt">${post.excerpt}</p></div>
    <span class="post-arrow" aria-hidden="true">↗</span>
  </a>`).join('');
