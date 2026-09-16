// Add a "Live explorer" link to the Material header, pointing at the published explorer.
document.addEventListener("DOMContentLoaded", () => {
  const header = document.querySelector(".md-header__inner");
  const config = document.getElementById("__config");
  if (!header || !config || header.querySelector(".tapestry-explorer-link")) return;
  const base = JSON.parse(config.textContent).base || ".";
  const link = document.createElement("a");
  link.className = "tapestry-explorer-link";
  link.href = `${base}/explorer/`;
  link.textContent = "Live explorer";
  link.title = "Tapestry surveillance data explorer (published snapshot)";
  header.insertBefore(link, header.querySelector(".md-search") || null);
});
