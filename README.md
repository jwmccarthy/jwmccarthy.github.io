# jack.notes

A small, dependency-free personal blog template.

## Add a post

Open `posts.js` and add an object to the **top** of the `posts` array:

```js
{
  date: '2026-10-20',
  title: 'Your post title',
  excerpt: 'One or two sentences to draw readers in.',
  url: 'posts/your-post.html'
}
```

The home page is plain HTML, CSS, and JavaScript, so it works on GitHub Pages without a build step. Create individual post pages in `posts/` as you need them and point `url` at each page.

## Test locally

From this directory, run `python3 -m http.server 8000`, then open <http://localhost:8000>.
