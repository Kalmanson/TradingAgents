"use strict";

const form = document.querySelector("#purchase-form");
if (form) {
  const ticker = document.querySelector("#ticker");
  const language = document.querySelector("#language");
  const button = document.querySelector("#purchase-button");
  const error = document.querySelector("#purchase-error");
  document.querySelectorAll("[data-ticker]").forEach((example) => {
    example.addEventListener("click", () => { ticker.value = example.dataset.ticker; ticker.focus(); });
  });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (button.disabled) return;
    const body = { ticker: ticker.value.trim().toUpperCase(), language: language.value };
    const fingerprint = JSON.stringify(body);
    let saved;
    try { saved = JSON.parse(sessionStorage.getItem("report-checkout") || "null"); } catch (_) { /* storage unavailable */ }
    const key = saved && saved.fingerprint === fingerprint ? saved.key : crypto.randomUUID();
    try { sessionStorage.setItem("report-checkout", JSON.stringify({ fingerprint, key })); } catch (_) { /* optional storage */ }
    error.hidden = true;
    button.disabled = true;
    const label = button.textContent;
    button.textContent = "Preparing secure checkout…";
    try {
      const response = await fetch("/api/orders", {
        method: "POST", headers: { "Content-Type": "application/json", "Idempotency-Key": key },
        body: fingerprint,
      });
      const result = await response.json();
      if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "We couldn't prepare checkout. Please try again.");
      window.location.assign(result.checkoutUrl);
    } catch (failure) {
      error.textContent = failure.message || "Connection interrupted. Please try again.";
      error.hidden = false;
      button.disabled = false;
      button.textContent = label;
    }
  });
}

const statusCard = document.querySelector("[data-status-token]");
if (statusCard) {
  const terminal = new Set(["COMPLETED", "FAILED", "REFUNDED"]);
  const messages = {
    PENDING_PAYMENT: ["Waiting for payment confirmation.", "This page updates after our payment service confirms your purchase."],
    PAID: ["Payment received. Preparing your report.", "Your report is in the research queue. You can close this page; we'll email your private link when it's ready."],
    RUNNING: ["Your committee is at work.", "The analysts are reviewing the evidence. We'll email your private report link when the research is complete."],
    COMPLETED: ["Your report is ready.", "Your report is saved. We will also send the link to your checkout email."],
    FAILED: ["We couldn't complete your report.", "Our team will review the order and arrange a refund. You do not need to pay again."],
    REFUNDED: ["This order has been refunded.", "Report access is no longer available for this order."],
  };
  async function poll() {
    try {
      const response = await fetch(`/api/orders/status/${encodeURIComponent(statusCard.dataset.statusToken)}`, { cache: "no-store" });
      if (!response.ok) throw new Error("Status temporarily unavailable.");
      const order = await response.json();
      if (!messages[order.status]) throw new Error("Status temporarily unavailable.");
      document.querySelector("#status-title").textContent = messages[order.status][0];
      document.querySelector("#status-description").textContent = messages[order.status][1];
      document.querySelector("#poll-notice").textContent = "";
      if (order.reportUrl) {
        const link = document.querySelector("#view-report");
        link.href = order.reportUrl;
        link.hidden = false;
      }
      if (terminal.has(order.status)) return;
    } catch (_) {
      document.querySelector("#poll-notice").textContent = "Reconnecting for an update. Your report continues in the background.";
    }
    setTimeout(poll, 5000);
  }
  if (!terminal.has(statusCard.dataset.orderState)) setTimeout(poll, 1500);
}
