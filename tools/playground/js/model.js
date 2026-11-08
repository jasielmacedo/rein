// The model on the visitor's machine: download with progress into the browser's cache, load into WebGPU (or the
// CPU build), report what is cached, delete it, download it again.

import {
  BATCH_TOKENS,
  CONTEXT_TOKENS,
  MODEL_BYTES,
  MODEL_FILE,
  MODEL_REPO,
  PARALLEL_DOWNLOADS,
  WLLAMA_ESM,
  WLLAMA_WASM,
} from "./config.js";

const REPO_MARK = MODEL_REPO.split("/")[1];

export const PHASES = {
  idle: "idle",
  downloading: "downloading",
  loading: "loading",
  ready: "ready",
  deleting: "deleting",
  error: "error",
};

export class ModelStore {
  constructor(onChange) {
    this.onChange = onChange;
    this.wllama = null;
    this.phase = PHASES.idle;
    this.progress = { loaded: 0, total: MODEL_BYTES };
    this.cachedBytes = 0;
    this.webgpu = false;
    this.threads = 1;
    this.error = null;
    this.controller = null;
  }

  async init() {
    const { Wllama } = await import(WLLAMA_ESM);
    this.wllama = new Wllama({ default: WLLAMA_WASM }, { parallelDownloads: PARALLEL_DOWNLOADS, suppressNativeLog: true });
    this.webgpu = typeof this.wllama.isSupportWebGPU === "function" && this.wllama.isSupportWebGPU();
    await this.refreshCache();
  }

  async refreshCache() {
    const entries = await this.wllama.cacheManager.list();
    this.cachedBytes = entries
      .filter((entry) => (entry.metadata?.originalURL ?? "").includes(REPO_MARK))
      .reduce((sum, entry) => sum + (entry.size > 0 ? entry.size : 0), 0);
    this.emit();
  }

  get isCached() {
    return this.cachedBytes >= MODEL_BYTES * 0.99;
  }

  get isReady() {
    return this.phase === PHASES.ready;
  }

  emit() {
    this.onChange(this);
  }

  setPhase(phase, error = null) {
    this.phase = phase;
    this.error = error;
    this.emit();
  }

  // Downloads what is missing (the cache keeps finished shards) and loads the model; one call serves both the
  // first download and a later visit.
  async load() {
    if (this.phase === PHASES.downloading || this.phase === PHASES.loading) return;
    this.controller = new AbortController();
    this.setPhase(PHASES.downloading);
    try {
      await this.wllama.loadModelFromHF(
        { repo: MODEL_REPO, file: MODEL_FILE },
        {
          n_ctx: CONTEXT_TOKENS,
          n_batch: BATCH_TOKENS,
          useCache: true,
          signal: this.controller.signal,
          progressCallback: ({ loaded, total }) => {
            this.progress = { loaded, total: total || MODEL_BYTES };
            if (loaded >= this.progress.total) this.phase = PHASES.loading;
            this.emit();
          },
        },
      );
      this.threads = this.wllama.getNumThreads();
      await this.refreshCache();
      this.setPhase(PHASES.ready);
    } catch (error) {
      if (this.controller.signal.aborted) {
        await this.refreshCache();
        this.setPhase(PHASES.idle);
        return;
      }
      this.setPhase(PHASES.error, error);
    }
  }

  cancel() {
    this.controller?.abort();
  }

  async unload() {
    if (this.wllama.isModelLoaded()) await this.wllama.exit();
  }

  async deleteCache() {
    this.setPhase(PHASES.deleting);
    try {
      await this.unload();
      await this.wllama.cacheManager.deleteMany((entry) => (entry.metadata?.originalURL ?? "").includes(REPO_MARK));
      await this.refreshCache();
      this.setPhase(PHASES.idle);
    } catch (error) {
      this.setPhase(PHASES.error, error);
    }
  }

  async redownload() {
    await this.deleteCache();
    await this.load();
  }

  backendLabel() {
    if (this.phase !== PHASES.ready) return this.webgpu ? "WebGPU available" : "no WebGPU: CPU build";
    return this.webgpu ? "running on WebGPU" : `running on the CPU, ${this.threads} thread${this.threads > 1 ? "s" : ""}`;
  }
}
