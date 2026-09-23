(() => {
  'use strict';

  const hero = document.querySelector('[data-hero]');
  const sticky = document.querySelector('[data-hero-sticky]');
  const content = document.querySelector('[data-hero-content]');
  const media = document.querySelector('[data-hero-media]');
  const collapsedHeader = document.querySelector('[data-collapsed-header]');
  const collapsedTitle = document.querySelector('[data-collapsed-title]');

  if (!hero || !sticky || !content || !media || !collapsedHeader || !collapsedTitle) {
    return;
  }

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let animationFrame = 0;

  const clamp = (value, min = 0, max = 1) => Math.min(Math.max(value, min), max);

  const render = () => {
    animationFrame = 0;

    const heroTop = hero.getBoundingClientRect().top;
    const transitionDistance = Math.max(hero.offsetHeight - sticky.offsetHeight, 1);
    const progress = clamp(-heroTop / transitionDistance);
    const hasReducedMotion = reducedMotion.matches;
    const opacity = hasReducedMotion ? (progress < 0.72 ? 1 : 0) : 1 - progress;
    const headerProgress = hasReducedMotion
      ? (progress >= 0.72 ? 1 : 0)
      : clamp((progress - 0.56) / 0.44);
    const headerIsActive = headerProgress >= 0.98;

    content.style.opacity = opacity.toFixed(3);
    content.style.transform = hasReducedMotion
      ? 'none'
      : `translate3d(0, ${(-112 * progress).toFixed(2)}px, 0) scale(${(1 - 0.14 * progress).toFixed(4)})`;
    content.style.pointerEvents = opacity < 0.05 ? 'none' : '';

    media.style.transform = hasReducedMotion
      ? 'scale(1.06)'
      : `translate3d(0, ${(30 * progress).toFixed(2)}px, 0) scale(${(1.06 + 0.08 * progress).toFixed(4)})`;

    if (media instanceof HTMLVideoElement) {
      const shouldPlay = !hasReducedMotion && !document.hidden && hero.getBoundingClientRect().bottom > 0;
      if (shouldPlay && media.paused) {
        media.play().catch(() => {});
      } else if (!shouldPlay && !media.paused) {
        media.pause();
      }
    }

    collapsedHeader.style.opacity = headerProgress.toFixed(3);
    collapsedHeader.style.transform = `translate3d(0, ${(-100 * (1 - headerProgress)).toFixed(2)}%, 0)`;
    collapsedHeader.style.visibility = headerProgress > 0 ? 'visible' : 'hidden';
    collapsedHeader.style.pointerEvents = headerIsActive ? 'auto' : 'none';
    collapsedHeader.setAttribute('aria-hidden', headerIsActive ? 'false' : 'true');
    collapsedTitle.tabIndex = headerIsActive ? 0 : -1;
  };

  const scheduleRender = () => {
    if (!animationFrame) {
      animationFrame = window.requestAnimationFrame(render);
    }
  };

  window.addEventListener('scroll', scheduleRender, {passive: true});
  window.addEventListener('resize', scheduleRender);
  document.addEventListener('visibilitychange', scheduleRender);
  reducedMotion.addEventListener('change', scheduleRender);
  render();
})();

(() => {
  'use strict';

  const carousel = document.querySelector('#results-carousel');
  if (!carousel || typeof window.bulmaCarousel === 'undefined') {
    return;
  }

  window.bulmaCarousel.attach(carousel, {
    slidesToScroll: 1,
    slidesToShow: 3,
    loop: true,
    infinite: true,
    autoplay: false,
    autoplaySpeed: 3000
  });
})();
