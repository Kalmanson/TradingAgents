"use strict";

const form = document.querySelector("#purchase-form");
if (form) {
  const ticker = document.querySelector("#ticker");
  const language = document.querySelector("#language");
  const button = document.querySelector("#purchase-button");
  const error = document.querySelector("#purchase-error");
  const captcha = document.querySelector("#purchase-captcha");
  let captchaToken = "";
  let captchaWidget;
  if (captcha) {
    const notice = document.querySelector("#captcha-notice");
    window.onPurchaseCaptchaLoad = () => {
      captchaWidget = window.grecaptcha.render(captcha, {
        sitekey: captcha.dataset.sitekey,
        size: captcha.clientWidth < 304 ? "compact" : "normal",
        callback: (token) => {
          captchaToken = token;
          notice.textContent = "Verification complete. You can continue to checkout.";
        },
        "expired-callback": () => {
          captchaToken = "";
          notice.textContent = "Verification expired. Please complete it again.";
        },
        "error-callback": () => {
          captchaToken = "";
          notice.textContent = "Verification unavailable. Check your connection and reload this page.";
        },
      });
      notice.textContent = "Please complete the verification before continuing.";
    };
    const script = document.createElement("script");
    script.src = "https://www.google.com/recaptcha/api.js?onload=onPurchaseCaptchaLoad&render=explicit";
    script.async = true;
    script.onerror = () => {
      notice.textContent = "Verification could not load. Check your connection and reload this page.";
    };
    document.head.appendChild(script);
  }
  document.querySelectorAll("[data-ticker]").forEach((example) => {
    example.addEventListener("click", () => { ticker.value = example.dataset.ticker; ticker.focus(); });
  });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (button.disabled) return;
    if (captcha && !captchaToken) {
      error.textContent = "Please complete the human verification before continuing.";
      error.hidden = false;
      return;
    }
    const body = { ticker: ticker.value.trim().toUpperCase(), language: language.value };
    // Only purchase inputs identify a retry; the one-use CAPTCHA token must not
    // be persisted or cause a new idempotency key when verification is repeated.
    const fingerprint = JSON.stringify(body);
    let saved;
    try { saved = JSON.parse(sessionStorage.getItem("report-checkout") || "null"); } catch (_) { /* storage unavailable */ }
    const key = saved && saved.fingerprint === fingerprint ? saved.key : crypto.randomUUID();
    try { sessionStorage.setItem("report-checkout", JSON.stringify({ fingerprint, key })); } catch (_) { /* optional storage */ }
    error.hidden = true;
    button.disabled = true;
    const label = button.innerHTML;
    let retryAfter = 0;
    button.textContent = "Preparing secure checkout…";
    try {
      const response = await fetch("/api/orders", {
        method: "POST", headers: { "Content-Type": "application/json", "Idempotency-Key": key },
        body: JSON.stringify(captcha ? { ...body, recaptcha_token: captchaToken } : body),
      });
      if (response.status === 429) {
        retryAfter = Math.max(1, Math.min(3600, Number(response.headers.get("Retry-After")) || 60));
      }
      const result = await response.json();
      if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "We couldn't prepare checkout. Please try again.");
      window.location.assign(result.checkoutUrl);
    } catch (failure) {
      error.textContent = failure.message || "Connection interrupted. Please try again.";
      error.hidden = false;
      if (captcha && captchaWidget !== undefined) {
        captchaToken = "";
        window.grecaptcha.reset(captchaWidget);
        document.querySelector("#captcha-notice").textContent = "Please complete a new verification before trying again.";
      }
      if (retryAfter) {
        button.textContent = "Please wait before retrying…";
        setTimeout(() => { button.disabled = false; button.innerHTML = label; }, retryAfter * 1000);
      } else {
        button.disabled = false;
        button.innerHTML = label;
      }
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
    let delay = 5000;
    try {
      const response = await fetch(`/api/orders/status/${encodeURIComponent(statusCard.dataset.statusToken)}`, { cache: "no-store" });
      if (response.status === 429) {
        delay = Math.max(5, Math.min(3600, Number(response.headers.get("Retry-After")) || 60)) * 1000;
      }
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
    setTimeout(poll, delay);
  }
  if (!terminal.has(statusCard.dataset.orderState)) setTimeout(poll, 1500);
}
