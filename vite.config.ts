import { defineConfig } from 'vite';

// The demo harness is index.html at the repo root; the reusable algorithm
// lives in src/core and pulls in no Vite/React/Electron APIs, so it can be
// lifted into portals-projector-agent untouched.
export default defineConfig({
  server: {
    port: 5180,
    strictPort: true,
    // Listen on all interfaces (IPv4 0.0.0.0 + IPv6) so it's reachable
    // whether the browser resolves `localhost` to 127.0.0.1 or ::1 — the
    // Windows dual-stack mismatch that makes Firefox "can't connect" when
    // the server is pinned to IPv4-only. getUserMedia still gets a secure
    // context via localhost/127.0.0.1.
    host: true,
  },
});
