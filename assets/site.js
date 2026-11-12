// Page behaviour: scroll progress, section highlighting, reveal-on-scroll, stat count-up, figure zoom.

const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// scroll progress bar and active nav link
const bar = document.querySelector("#scroll-progress");
const links = [...document.querySelectorAll("#nav a")];
const targets = links.map((a) => document.querySelector(a.getAttribute("href")));
function onScroll() {
  const max = document.documentElement.scrollHeight - innerHeight;
  bar.style.width = `${max > 0 ? (scrollY / max) * 100 : 0}%`;
  let current = -1;
  targets.forEach((el, i) => { if (el && el.getBoundingClientRect().top < innerHeight * 0.4) current = i; });
  links.forEach((a, i) => a.classList.toggle("on", i === current));
}
addEventListener("scroll", onScroll, { passive: true });
addEventListener("resize", onScroll);
onScroll();

// reveal on scroll
const revealables = document.querySelectorAll(".reveal");
if (reduced || !("IntersectionObserver" in window)) {
  revealables.forEach((el) => el.classList.add("in"));
} else {
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
  }, { rootMargin: "0px 0px -8% 0px" });
  revealables.forEach((el) => io.observe(el));
}

// stat count-up
function format(value, el) {
  const decimals = +(el.dataset.decimals || 0);
  const text = decimals ? value.toFixed(decimals) : Math.round(value).toLocaleString("en-GB");
  return text + (el.dataset.suffix || "");
}
const counters = document.querySelectorAll("[data-count]");
if (!reduced && "IntersectionObserver" in window) {
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      io.unobserve(e.target);
      const el = e.target, end = +el.dataset.count, t0 = performance.now(), ms = 1100;
      const step = (t) => {
        const k = Math.min(1, (t - t0) / ms), eased = 1 - (1 - k) ** 3;
        el.textContent = format(end * eased, el);
        if (k < 1) requestAnimationFrame(step); else el.textContent = format(end, el);
      };
      requestAnimationFrame(step);
    }
  }, { threshold: 0.6 });
  counters.forEach((el) => io.observe(el));
}

// project clock on the timeline: months since January 2026 out of 48
const start = new Date(2026, 0, 1), now = new Date();
const months = (now.getFullYear() - start.getFullYear()) * 12 + now.getMonth() + (now.getDate() - 1) / 30;
document.querySelector("#timeline")?.style.setProperty("--now", Math.max(0, Math.min(1, months / 48)).toFixed(4));

// figure zoom
const box = document.querySelector("#lightbox"), boxImg = box.querySelector("img");
document.querySelectorAll("img[data-zoom]").forEach((img) => img.addEventListener("click", () => {
  boxImg.src = img.currentSrc || img.src; boxImg.alt = img.alt; box.hidden = false;
}));
box.addEventListener("click", () => { box.hidden = true; });
addEventListener("keydown", (e) => { if (e.key === "Escape") box.hidden = true; });
