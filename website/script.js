const revealItems = Array.from(document.querySelectorAll(".reveal"));

if (revealItems.length > 0) {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("show");
          observer.unobserve(entry.target);
        }
      });
    },
    {
      threshold: 0.14,
      rootMargin: "0px 0px -6% 0px",
    },
  );

  revealItems.forEach((item, idx) => {
    item.style.transitionDelay = `${Math.min(idx * 45, 300)}ms`;
    observer.observe(item);
  });
}

const yearElements = Array.from(document.querySelectorAll("#year"));
yearElements.forEach((el) => {
  el.textContent = String(new Date().getFullYear());
});

const modeTabs = Array.from(document.querySelectorAll(".mode-tab"));
const modePanels = Array.from(document.querySelectorAll(".mode-panel"));

if (modeTabs.length > 0 && modePanels.length > 0) {
  const activateMode = (mode) => {
    modeTabs.forEach((tab) => {
      const isActive = tab.dataset.mode === mode;
      tab.classList.toggle("is-active", isActive);
      tab.setAttribute("aria-selected", String(isActive));
    });

    modePanels.forEach((panel) => {
      panel.classList.toggle("is-active", panel.dataset.panel === mode);
    });
  };

  modeTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      activateMode(tab.dataset.mode);
    });
  });
}

const faqSearchInput = document.querySelector("[data-faq-search]");
const faqItems = Array.from(document.querySelectorAll("[data-faq-item]"));
const faqCount = document.querySelector("[data-faq-count]");

if (faqItems.length > 0) {
  const updateFaqCount = () => {
    const visibleItems = faqItems.filter((item) => !item.classList.contains("is-hidden"));
    if (faqCount) {
      faqCount.textContent = String(visibleItems.length);
    }
  };

  const filterFaqs = (query) => {
    const normalized = query.trim().toLowerCase();

    faqItems.forEach((item) => {
      const text = item.textContent.toLowerCase();
      const matches = normalized.length === 0 || text.includes(normalized);
      item.classList.toggle("is-hidden", !matches);
      if (!matches) {
        item.removeAttribute("open");
      }
    });

    updateFaqCount();
  };

  updateFaqCount();

  if (faqSearchInput) {
    faqSearchInput.addEventListener("input", (event) => {
      filterFaqs(event.target.value);
    });
  }
}
