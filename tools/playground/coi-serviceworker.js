// Adds the cross-origin isolation headers a static host cannot set, so the multi-threaded WebAssembly build can
// use SharedArrayBuffer. The page registers this worker once and reloads under its control.

const COOP = ["Cross-Origin-Opener-Policy", "same-origin"];
const COEP = ["Cross-Origin-Embedder-Policy", "require-corp"];

self.addEventListener("install", () => self.skipWaiting());

self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  if (event.request.cache === "only-if-cached" && event.request.mode !== "same-origin") return;
  event.respondWith(
    fetch(event.request).then((response) => {
      if (response.status === 0) return response;
      const headers = new Headers(response.headers);
      headers.set(COOP[0], COOP[1]);
      headers.set(COEP[0], COEP[1]);
      return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
    }),
  );
});
